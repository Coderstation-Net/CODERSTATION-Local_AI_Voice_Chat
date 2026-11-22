let ws = null;
  let currentAssistantBubble = null;
  let localMicAvailable = false;
  let handsFree = true;
  let muted = false;
  let browserRecognition = null;

  // Web Audio + HTML5 Audio Queue for smooth, zero-gap playback directly in the browser
  let audioCtx = null;
  let avatarAnalyser = null;
  let avatarFreqData = null;
  let avatarTimeData = null;
  let liveMicRms = 0;
  let liveMicRmsTs = 0;
  let liveTtsRms = 0;
  let liveTtsRmsTs = 0;
  let isCapturingVoice = false;
  let voiceCaptureMode = 'standby';
  let audioQueue = [];
  let isPlayingBrowserAudio = false;
  let currentAudioSource = null;
  let currentHtmlAudio = null;

  function ensureAudioContext() {
    try {
      if (!audioCtx) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        audioCtx = new AudioContextClass();
        avatarAnalyser = audioCtx.createAnalyser();
        avatarAnalyser.fftSize = 1024;
        avatarAnalyser.smoothingTimeConstant = 0.78;
        avatarAnalyser.connect(audioCtx.destination);
        avatarFreqData = new Uint8Array(avatarAnalyser.frequencyBinCount);
        avatarTimeData = new Uint8Array(avatarAnalyser.fftSize);
      }
      if (audioCtx.state === 'suspended') {
        audioCtx.resume();
      }
    } catch (_) {}
  }

  // Unlock browser Web Audio on every click or keypress
  window.addEventListener('pointerdown', ensureAudioContext);
  window.addEventListener('keydown', ensureAudioContext);

  async function enqueueBrowserWav(base64Wav, fallbackText) {
    if (muted) return;
    ensureAudioContext();
    audioQueue.push({ b64: base64Wav, text: fallbackText || '' });
    if (!isPlayingBrowserAudio) {
      playNextBrowserWav();
    }
  }

  async function playNextBrowserWav() {
    if (audioQueue.length === 0 || muted) {
      isPlayingBrowserAudio = false;
      currentAudioSource = null;
      currentHtmlAudio = null;
      if (document.body.getAttribute('data-state') === 'speaking') {
        setUiState('idle');
      }
      return;
    }
    isPlayingBrowserAudio = true;
    setUiState('speaking');
    const item = audioQueue.shift();

    // Primary path: Web Audio API decodeAudioData routed through avatarAnalyser for real-time 60FPS formant sync
    try {
      ensureAudioContext();
      if (audioCtx && audioCtx.state === 'running' && avatarAnalyser) {
        const binary = atob(item.b64);
        const len = binary.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
        const audioBuffer = await audioCtx.decodeAudioData(bytes.buffer);
        const source = audioCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(avatarAnalyser);
        currentAudioSource = source;
        source.onended = () => {
          currentAudioSource = null;
          playNextBrowserWav();
        };
        source.start(0);
        return;
      }
    } catch (e) {
      console.warn('WebAudio playback failed, trying HTML5 Audio element:', e);
    }

    // Secondary path: HTML5 Audio Element with Data URL (works reliably across all browsers)
    try {
      const audioEl = new Audio('data:audio/wav;base64,' + item.b64);
      currentHtmlAudio = audioEl;
      audioEl.onended = () => {
        currentHtmlAudio = null;
        playNextBrowserWav();
      };
      audioEl.onerror = () => {
        currentHtmlAudio = null;
        if (item.text) speakBrowserFallback(item.text);
        playNextBrowserWav();
      };
      await audioEl.play();
      return;
    } catch (err) {
      console.warn('HTML5 Audio play blocked or failed, trying SpeechSynthesis fallback:', err);
      if (item.text) speakBrowserFallback(item.text);
      playNextBrowserWav();
    }
  }

  function stopBrowserAudio() {
    audioQueue = [];
    if (currentHtmlAudio) {
      try {
        currentHtmlAudio.pause();
        currentHtmlAudio.currentTime = 0;
      } catch (_) {}
      currentHtmlAudio = null;
    }
    if (currentAudioSource) {
      try { currentAudioSource.stop(); } catch (_) {}
      currentAudioSource = null;
    }
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    isPlayingBrowserAudio = false;
  }

  function speakBrowserFallback(text) {
    if (muted || !('speechSynthesis' in window)) return;
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1.08;
    utter.onstart = () => setUiState('speaking');
    utter.onend = () => setUiState('idle');
    window.speechSynthesis.speak(utter);
  }

  // Setup Web Speech API fallback if Windows native mic access is blocked
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    browserRecognition = new SpeechRecognition();
    browserRecognition.lang = 'en-US';
    browserRecognition.interimResults = false;
    browserRecognition.maxAlternatives = 1;

    browserRecognition.onstart = () => {
      setUiState('listening');
      setBanner('Listening via Browser Mic...');
    };
    browserRecognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript.trim();
      if (transcript && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'chat', text: transcript }));
      }
    };
    browserRecognition.onerror = (e) => {
      setUiState('idle');
      setBanner('Voice error: ' + e.error + '. Check Windows Microphone Privacy Settings.');
    };
    browserRecognition.onend = () => {
      if (document.body.getAttribute('data-state') === 'listening') {
        setUiState('idle');
      }
      if (handsFree && !localMicAvailable) {
        setTimeout(() => { try { browserRecognition.start(); } catch (_) {} }, 400);
      }
    };
  }

  function connectWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/ws`);

    ws.onopen = () => {
      // Disable duplicate host speaker playback when browser audio is active to prevent echo,
      // while keeping host speaker selectable if user switches device
      ws.send(JSON.stringify({ action: 'set_host_speaker', enabled: true }));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleServerEvent(data);
    };

    ws.onclose = () => {
      setBanner('Connection lost. Reconnecting...');
      setTimeout(connectWebSocket, 1500);
    };
  }

  function handleServerEvent(data) {
    switch (data.type) {
      case 'init':
        setUiState(data.state || 'idle');
        localMicAvailable = !!data.mic_available;
        muted = !!data.muted;
        updateMicBadge();
        break;

      case 'state':
        if (data.state === 'idle' && isPlayingBrowserAudio) {
          // Keep speaking animation while browser audio finishes
          return;
        }
        setUiState(data.state);
        break;

      case 'audio_chunk':
        if (data.wav_base64) {
          enqueueBrowserWav(data.wav_base64, data.text);
        }
        break;

      case 'tts_speaking':
        if (data.fallback_browser_tts && data.text) {
          speakBrowserFallback(data.text);
        }
        break;

      case 'stop_audio':
        stopBrowserAudio();
        break;

      case 'mic_level':
        liveMicRms = Math.min(1.0, Math.max(0, Number(data.rms) || 0));
        liveMicRmsTs = performance.now();
        if (typeof data.capturing === 'boolean') {
          isCapturingVoice = data.capturing;
          document.body.setAttribute('data-capturing', isCapturingVoice ? 'true' : 'false');
        }
        break;

      case 'voice_capture':
        isCapturingVoice = !!data.active;
        voiceCaptureMode = data.mode || 'command';
        document.body.setAttribute('data-capturing', isCapturingVoice ? 'true' : 'false');
        if (typeof data.rms === 'number') {
          liveMicRms = Math.min(1.0, Math.max(0, data.rms));
          liveMicRmsTs = performance.now();
        }
        updateStateLabelText();
        break;

      case 'tts_level':
        liveTtsRms = Math.min(1.0, Math.max(0, Number(data.rms) || 0));
        liveTtsRmsTs = performance.now();
        break;

      case 'stt_result':
        setBanner(`Heard: "${data.text}" (${data.confidence || 92}% • ${data.latency_ms || 0}ms • ${data.model || 'STT'})`);
        break;

      case 'user_message':
        if (!document.getElementById('statusBanner').textContent.startsWith('Heard:')) {
          setBanner('Heard: "' + data.text + '"');
        }
        break;

      case 'tool_call':
        setBanner(`⚡ Executing tool: ${data.name}(${JSON.stringify(data.args)})`);
        break;

      case 'assistant_start':
        setLastAnswer('');
        break;

      case 'assistant_token':
        setLastAnswer(data.full_text);
        break;

      case 'assistant_done':
        if (data.text) setLastAnswer(data.text);
        break;

      case 'status_note':
        setBanner(data.message);
        break;

      case 'error':
        setBanner(data.message);
        setLastAnswer(`⚠️ ${data.message}`);
        break;

      case 'history_cleared':
        setLastAnswer('Conversation memory cleared. Ready for your next command, Sir.');
        break;
    }
  }

  function updateStateLabelText() {
    const lbl = document.getElementById('stateLabel');
    if (!lbl) return;
    const st = document.body.getAttribute('data-state') || 'idle';
    if (isCapturingVoice && st !== 'speaking' && st !== 'thinking') {
      lbl.textContent = 'CAPTURING VOICE';
    } else {
      lbl.textContent = st.toUpperCase();
    }
  }

  function setUiState(state) {
    document.body.setAttribute('data-state', state);
    if (state === 'thinking' || state === 'speaking') {
      isCapturingVoice = false;
      document.body.setAttribute('data-capturing', 'false');
    }
    if (state !== 'listening' && !isCapturingVoice) {
      liveMicRms = 0;
    }
    updateStateLabelText();
  }

  function setBanner(msg) {
    document.getElementById('statusBanner').textContent = msg;
  }

  function setLastAnswer(text) {
    const contentEl = document.getElementById('lastAnswerContent');
    if (contentEl) {
      contentEl.textContent = text;
    }
    scrollChat();
  }

  function toggleSettingsPanel() {
    const wsEl = document.getElementById('mainWorkspace');
    const btn = document.getElementById('btnToggleSettings');
    const hidden = wsEl.classList.toggle('settings-hidden');
    btn.textContent = hidden ? '⚙️ Show Settings' : '⚙️ Hide Settings';
  }

  function toggleChatInput() {
    const bar = document.getElementById('chatInputBar');
    const btn = document.getElementById('btnToggleInput');
    const hidden = bar.classList.toggle('input-hidden');
    btn.textContent = hidden ? '💬 Show Input' : '💬 Hide Input';
    if (!hidden) {
      document.getElementById('chatInput').focus();
    }
  }

  function scrollChat() {
    const log = document.getElementById('chatLog');
    log.scrollTop = log.scrollHeight;
  }

  function sendTextCommand() {
    ensureAudioContext();
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ action: 'chat', text }));
    input.value = '';
  }

  document.getElementById('chatInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendTextCommand();
  });

  function testVoiceOutput() {
    ensureAudioContext();
    setBanner('Synthesizing test voice...');
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'test_voice' }));
    }
  }

  function toggleHandsFree() {
    ensureAudioContext();
    handsFree = !handsFree;
    const btn = document.getElementById('btnHandsFree');
    if (btn) {
      btn.textContent = handsFree ? '🔁 Wake "Coderstation": ON' : '🔁 Wake "Coderstation": OFF';
      btn.classList.toggle('active-toggle', handsFree);
    }
    setBanner(handsFree ? 'Say "Coderstation" anytime to speak' : 'Wake phrase paused');
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'toggle_continuous', enabled: handsFree }));
    }
    if (handsFree && !localMicAvailable && browserRecognition) {
      try { browserRecognition.start(); } catch (_) {}
    }
  }

  function stopSpeaking() {
    stopBrowserAudio();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'stop_speaking' }));
    }
  }

  function toggleMute() {
    muted = !muted;
    if (muted) stopBrowserAudio();
    const btn = document.getElementById('btnMute');
    if (btn) {
      btn.textContent = muted ? '🔇 Voice: MUTED' : '🔊 Voice: ON';
      btn.classList.toggle('active-toggle', muted);
    }
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'set_mute', muted }));
    }
  }

  function changeModel(model) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'set_model', model }));
    }
  }

  function changeVoice(voice) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'set_voice', voice }));
    }
  }

  function changeWhisperModel(model) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'set_whisper_model', model }));
    }
  }

  function triggerVoiceCapture() {
    ensureAudioContext();
    if (localMicAvailable && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'listen_now' }));
    } else if (browserRecognition) {
      try { browserRecognition.start(); } catch (_) {}
    } else {
      setBanner('Microphone unavailable — use Text Input or enable Windows mic permissions.');
    }
  }

  function changeOutputDevice(deviceId) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'set_output_device', device_id: deviceId }));
    }
  }

  function clearHistory() {
    stopBrowserAudio();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'clear_history' }));
    }
  }

  function updateMicBadge() {
    const el = document.getElementById('micVal');
    if (localMicAvailable) {
      el.textContent = 'READY (CUDA)';
      el.style.color = '#00e676';
    } else if (browserRecognition) {
      el.textContent = 'WEB SPEECH';
      el.style.color = '#BBF2FC';
    } else {
      el.textContent = 'BLOCKED';
      el.style.color = '#ff2a6d';
    }
  }

  let speakersPopulated = false;
  const _chartHistory = { cpu: [], gpu: [], ram: [], gpu_vram: [] };
  function pushAndDrawChart(key, canvasId, value, strokeColor, fillColor) {
    const arr = _chartHistory[key];
    arr.push(Math.max(0, Math.min(100, Number(value) || 0)));
    if (arr.length > 28) arr.shift();
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Subtle grid line
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h * 0.5);
    ctx.lineTo(w, h * 0.5);
    ctx.stroke();

    if (arr.length < 2) return;
    const step = w / 27;
    ctx.beginPath();
    arr.forEach((v, i) => {
      const x = i * step;
      const y = h - (v / 100) * (h - 6) - 3;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.lineTo((arr.length - 1) * step, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, fillColor);
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = grad;
    ctx.fill();
  }

  function getDarkMetricColor(pct, normalColor) {
    if (pct >= 85) return '#f87171';
    if (pct >= 60) return '#BBF2FC';
    return normalColor;
  }

  async function pollTelemetry() {
    try {
      const res = await fetch('/api/telemetry');
      const data = await res.json();
      const cpuEl = document.getElementById('cpuVal');
      const ramEl = document.getElementById('ramVal');
      cpuEl.textContent = `${Number(data.cpu_percent).toFixed(1)}% (${data.cpu_threads || '--'}T)`;
      cpuEl.style.color = getDarkMetricColor(data.cpu_percent, '#38bdf8');
      ramEl.textContent = `${Number(data.ram_percent).toFixed(1)}% (${data.ram_used_gb}/${data.ram_total_gb}G)`;
      ramEl.style.color = getDarkMetricColor(data.ram_percent, '#c4b5fd');
      pushAndDrawChart('cpu', 'cpuChart', data.cpu_percent, '#38bdf8', 'rgba(56, 189, 248, 0.28)');
      pushAndDrawChart('ram', 'ramChart', data.ram_percent, '#a78bfa', 'rgba(167, 139, 250, 0.28)');

      // Update GPU telemetry
      const gpu = data.gpu || {};
      const gpuValEl = document.getElementById('gpuVal');
      const gpuNameEl = document.getElementById('gpuNameLabel');
      const gpuTempEl = document.getElementById('gpuTempBadge');
      const gpuUtilEl = document.getElementById('gpuUtilText');
      const gpuVramEl = document.getElementById('gpuVramText');
      const gpuBarEl = document.getElementById('gpuVramBar');
      if (gpu.available) {
        gpuValEl.textContent = `${Number(gpu.util_percent).toFixed(1)}% (${gpu.mem_used_gb}/${gpu.mem_total_gb}G)`;
        gpuValEl.style.color = getDarkMetricColor(gpu.util_percent, '#34d399');
        gpuNameEl.textContent = `GPU // ${gpu.name.replace('NVIDIA GeForce ', '')}`;
        gpuTempEl.textContent = `${gpu.temp_c}°C`;
        gpuTempEl.style.color = gpu.temp_c >= 80 ? '#f87171' : (gpu.temp_c >= 65 ? '#BBF2FC' : '#34d399');
        gpuUtilEl.textContent = `${gpu.util_percent}%`;
        gpuUtilEl.style.color = getDarkMetricColor(gpu.util_percent, '#34d399');
        gpuVramEl.textContent = `${gpu.mem_used_gb} / ${gpu.mem_total_gb} GB`;
        gpuBarEl.style.width = `${Math.min(100, Math.max(0, gpu.mem_percent || 0))}%`;
        pushAndDrawChart('gpu', 'gpuChart', gpu.util_percent, '#10b981', 'rgba(16, 185, 129, 0.28)');
        pushAndDrawChart('gpu_vram', 'gpuVramChart', gpu.mem_percent || 0, '#38bdf8', 'rgba(56, 189, 248, 0.25)');
      } else {
        gpuValEl.textContent = 'N/A';
        gpuNameEl.textContent = 'GPU // UNAVAILABLE';
        gpuTempEl.textContent = '--°C';
        gpuUtilEl.textContent = '--%';
        gpuVramEl.textContent = '-- / -- GB';
        gpuBarEl.style.width = '0%';
        pushAndDrawChart('gpu', 'gpuChart', 0, '#10b981', 'rgba(16, 185, 129, 0.28)');
        pushAndDrawChart('gpu_vram', 'gpuVramChart', 0, '#38bdf8', 'rgba(56, 189, 248, 0.25)');
      }

      // Update Ollama PS active models
      const psList = Array.isArray(data.ollama_ps) ? data.ollama_ps : [];
      const psCountEl = document.getElementById('ollamaPsCount');
      const psListEl = document.getElementById('ollamaPsList');
      psCountEl.textContent = `${psList.length} ACTIVE`;
      if (psList.length === 0) {
        psListEl.innerHTML = '<div class="ollama-ps-empty">Standby — No active models loaded in VRAM</div>';
      } else {
        psListEl.innerHTML = psList.map(m => `
          <div class="ollama-ps-item">
            <span class="ollama-ps-name" title="${m.name}">${m.name}</span>
            <span>${m.size_gb} GB</span>
            <span>${m.processor}</span>
            <span>CTX ${m.context}</span>
          </div>
        `).join('');
      }
      document.getElementById('sttBadge').textContent = `STT: ${data.current_whisper_model || 'small.en'} ${data.stt_backend} | TTS: ${data.tts_backend || 'CUDA'}`;
      const wSel = document.getElementById('whisperSelect');
      if (wSel && data.current_whisper_model && wSel.value !== data.current_whisper_model) {
        wSel.value = data.current_whisper_model;
      }
      const vSel = document.getElementById('voiceSelect');
      if (vSel && data.current_voice && vSel.value !== data.current_voice) {
        vSel.value = data.current_voice;
      }
      localMicAvailable = !!data.mic_available;
      updateMicBadge();

      if (Array.isArray(data.available_models) && data.available_models.length > 0) {
        const sel = document.getElementById('modelSelect');
        const existing = Array.from(sel.options).map(o => o.value);
        data.available_models.forEach(m => {
          if (m && !existing.includes(m)) {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            sel.appendChild(opt);
          }
        });
        sel.value = data.current_model;
      }

      if (!speakersPopulated && Array.isArray(data.output_devices) && data.output_devices.length > 0) {
        const spkSel = document.getElementById('speakerSelect');
        data.output_devices.forEach(d => {
          const opt = document.createElement('option');
          opt.value = d.id;
          opt.textContent = d.name;
          spkSel.appendChild(opt);
        });
        if (data.selected_output_device !== null && data.selected_output_device !== undefined) {
          spkSel.value = String(data.selected_output_device);
        }
        speakersPopulated = true;
      }
    } catch (_) {}
  }

  function updateHudClock() {
    const el = document.getElementById('hudClock');
    if (el) {
      el.textContent = new Date().toLocaleTimeString([], { hour12: false });
    }
  }
  updateHudClock();
  setInterval(updateHudClock, 1000);

  connectWebSocket();
  pollTelemetry();
  setInterval(pollTelemetry, 2500);
  // === Animated Glowing Network Wiring & Running Data Packets ===
  (function initNetworkWiring() {
    const canvas = document.getElementById('matrixBgCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const palette = ['#0284c7', '#0ea5e9', '#10b981', '#6366f1'];
    let wires = [];
    let packets = [];

    function buildNetwork() {
      const w = canvas.width = window.innerWidth;
      const h = canvas.height = window.innerHeight;
      wires = [];
      packets = [];
      const count = 6;

      for (let i = 0; i < count; i++) {
        const horizontal = i % 2 === 0;
        const pts = [];
        if (horizontal) {
          let x = -20;
          let y = ((i + 0.5) / count) * h + (Math.random() - 0.5) * 40;
          pts.push({ x, y });
          const segs = 4 + Math.floor(Math.random() * 4);
          for (let s = 1; s <= segs; s++) {
            const nx = (s / segs) * (w + 40) - 20;
            const midX = x + (nx - x) * (0.35 + Math.random() * 0.3);
            pts.push({ x: midX, y });
            const dy = (Math.random() - 0.5) * 90;
            y = Math.max(20, Math.min(h - 20, y + dy));
            pts.push({ x: midX + Math.abs(dy) * 0.4, y });
            pts.push({ x: nx, y });
            x = nx;
          }
        } else {
          let x = ((i + 0.5) / count) * w + (Math.random() - 0.5) * 40;
          let y = -20;
          pts.push({ x, y });
          const segs = 3 + Math.floor(Math.random() * 3);
          for (let s = 1; s <= segs; s++) {
            const ny = (s / segs) * (h + 40) - 20;
            const midY = y + (ny - y) * (0.35 + Math.random() * 0.3);
            pts.push({ x, y: midY });
            const dx = (Math.random() - 0.5) * 90;
            x = Math.max(20, Math.min(w - 20, x + dx));
            pts.push({ x, y: midY + Math.abs(dx) * 0.4 });
            pts.push({ x, y: ny });
            y = ny;
          }
        }

        // Compute cumulative segment lengths
        let totalLen = 0;
        const segLens = [];
        for (let j = 0; j < pts.length - 1; j++) {
          const d = Math.hypot(pts[j + 1].x - pts[j].x, pts[j + 1].y - pts[j].y);
          segLens.push(d);
          totalLen += d;
        }
        const color = palette[i % palette.length];
        wires.push({ pts, segLens, totalLen, color });

        // 1 slow-moving data packet per wire
        packets.push({
          wireIdx: i,
          dist: Math.random() * totalLen,
          speed: 0.35 + Math.random() * 0.45,
          dir: Math.random() > 0.2 ? 1 : -1,
          color,
        });
      }
    }

    function pointAtDist(wire, dist) {
      let d = dist;
      for (let j = 0; j < wire.segLens.length; j++) {
        const sl = wire.segLens[j];
        if (d <= sl || j === wire.segLens.length - 1) {
          const t = sl > 0 ? Math.max(0, Math.min(1, d / sl)) : 0;
          const a = wire.pts[j], b = wire.pts[j + 1];
          return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
        }
        d -= sl;
      }
      return wire.pts[0];
    }

    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw static glowing network traces & junction nodes
      for (const wire of wires) {
        ctx.save();
        ctx.beginPath();
        wire.pts.forEach((p, idx) => (idx === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
        ctx.strokeStyle = 'rgba(14, 165, 233, 0.22)';
        ctx.lineWidth = 1.25;
        ctx.shadowColor = wire.color;
        ctx.shadowBlur = 6;
        ctx.stroke();

        // Junction nodes
        ctx.fillStyle = 'rgba(56, 189, 248, 0.35)';
        for (let j = 1; j < wire.pts.length - 1; j += 2) {
          ctx.beginPath();
          ctx.arc(wire.pts[j].x, wire.pts[j].y, 2.1, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }

      // Draw running data packets with glowing trails
      for (const pkt of packets) {
        const wire = wires[pkt.wireIdx];
        if (!wire || wire.totalLen <= 0) continue;
        pkt.dist += pkt.speed * pkt.dir;
        if (pkt.dist > wire.totalLen) pkt.dist = 0;
        if (pkt.dist < 0) pkt.dist = wire.totalLen;

        const head = pointAtDist(wire, pkt.dist);
        const tail = pointAtDist(wire, Math.max(0, Math.min(wire.totalLen, pkt.dist - pkt.dir * 26)));

        ctx.save();
        const grad = ctx.createLinearGradient(tail.x, tail.y, head.x, head.y);
        grad.addColorStop(0, 'rgba(56, 189, 248, 0)');
        grad.addColorStop(1, pkt.color);
        ctx.strokeStyle = grad;
        ctx.lineWidth = 2.8;
        ctx.lineCap = 'round';
        ctx.shadowColor = pkt.color;
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.moveTo(tail.x, tail.y);
        ctx.lineTo(head.x, head.y);
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(head.x, head.y, 2.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      requestAnimationFrame(animate);
    }

    window.addEventListener('resize', buildNetwork);
    buildNetwork();
    requestAnimationFrame(animate);
  })();
  // === Animated Random Line Chart Background for Inference Engine ===
  (function initInferenceBgChart() {
    const canvas = document.getElementById('inferenceBgChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const maxPts = 36;
    const vals = Array.from({ length: maxPts }, () => 25 + Math.random() * 45);
    let lastVal = vals[vals.length - 1];

    function resize() {
      const parent = canvas.parentElement;
      if (!parent) return;
      canvas.width = parent.clientWidth || 320;
      canvas.height = parent.clientHeight || 140;
    }
    window.addEventListener('resize', resize);
    resize();

    setInterval(() => {
      if (canvas.width === 0 || canvas.height === 0) resize();
      const delta = (Math.random() - 0.5) * 26;
      lastVal = Math.max(12, Math.min(88, lastVal + delta));
      vals.push(lastVal);
      if (vals.length > maxPts) vals.shift();

      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const step = w / (maxPts - 1);
      ctx.beginPath();
      vals.forEach((v, i) => {
        const x = i * step;
        const y = h - (v / 100) * (h * 0.72) - 8;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.65)';
      ctx.lineWidth = 1.6;
      ctx.stroke();

      ctx.lineTo(w, h);
      ctx.lineTo(0, h);
      ctx.closePath();
      const grad = ctx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, 'rgba(56, 189, 248, 0.24)');
      grad.addColorStop(1, 'rgba(56, 189, 248, 0.0)');
      ctx.fillStyle = grad;
      ctx.fill();
    }, 320);
  })();
  // === 3D Holographic Pure-Dot Circuit Avatar Engine (Coderstation Voice-Clone Design) ===
  (function initCoderstation3DAvatar() {
    const canvas = document.getElementById('coderstationAvatarCanvas');
    const wrapper = document.getElementById('avatarStageWrapper');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const avState = {
      phase: 0,
      smoothRms: 0,
      smoothBass: 0,
      smoothMid: 0,
      smoothHigh: 0,
      thinkAlpha: 0,
      captureAlpha: 0,
      captureBars: new Float32Array(24),
      avatarDots: [],
      bodyCircuits: [],
      brainCircuits: [],
      envMotes3D: [],
      logicalW: 840,
      logicalH: 430,
    };

    function resizeAvatarCanvas() {
      const rect = (wrapper || canvas).getBoundingClientRect();
      const w = Math.max(320, Math.round(rect.width || 840));
      const h = Math.max(260, Math.round(rect.height || 430));
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      avState.logicalW = w;
      avState.logicalH = h;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    if (typeof ResizeObserver !== 'undefined' && wrapper) {
      new ResizeObserver(resizeAvatarCanvas).observe(wrapper);
    }
    window.addEventListener('resize', resizeAvatarCanvas);
    resizeAvatarCanvas();

    function getAnatomicalHeadHalfW(dy) {
      const topY = -124;
      const botY = 92;
      if (dy <= topY || dy >= botY) return 0;
      const midY = -22;
      let hw = 0;
      if (dy <= midY) {
        const u = (dy - midY) / (midY - topY);
        hw = 64 * Math.pow(Math.max(0, 1 - u * u), 0.48);
      } else {
        const u = (dy - midY) / (botY - midY);
        const oval = Math.sqrt(Math.max(0, 1 - u * u));
        hw = 64 * oval * (1 - 0.10 * Math.pow(u, 1.3) - 0.14 * Math.pow(u, 2.3));
      }
      if (dy >= -20 && dy <= 28) {
        const earT = (dy + 20) / 48;
        const earCurve = Math.sin(earT * Math.PI) * (0.85 + 0.25 * Math.sin(earT * Math.PI * 0.7));
        hw += earCurve * 7.2;
      }
      return hw;
    }

    function getAnatomicalTorsoHalfW(dy) {
      if (dy < 64) return 0;
      if (dy <= 122) {
        const t = (dy - 64) / 58;
        const inward = Math.sin(t * Math.PI) * 5.2;
        const baseFlare = Math.pow(Math.max(0, (t - 0.62) / 0.38), 2.0) * 14.0;
        return 36.5 - inward + baseFlare;
      }
      if (dy <= 176) {
        const t = (dy - 122) / 54;
        const smoothS = t * t * (3 - 2 * t);
        return 50.5 + smoothS * 97.5;
      }
      if (dy <= 226) {
        const t = (dy - 176) / 50;
        const dome = Math.sin(t * Math.PI * 0.5);
        return 148 + dome * 30;
      }
      const t = Math.min(1.0, (dy - 226) / 39);
      return 178 + Math.sin(t * Math.PI * 0.5) * 4;
    }

    function getDotFacialReliefY(nx, ny) {
      const r2 = Math.min(1, nx * nx + ny * ny);
      const zSphere = Math.sqrt(Math.max(0, 1 - r2));
      let archY = ny * (1 - zSphere) * 9.0;
      const nose =
        Math.exp(-Math.pow(nx / 0.14, 2)) * Math.exp(-Math.pow((ny - 0.04) / 0.22, 2));
      archY -= nose * 3.8;
      return archY;
    }

    function initAnimatedAvatarDots() {
      const dots = [];

      // 1. High-Resolution 3D Head Outer Silhouette & Inner Micro-Bevel Rim
      const rimSteps = 44;
      for (let i = 0; i <= rimSteps; i++) {
        const t = i / rimSteps;
        const dy = -122 + t * 218;
        const hw = getAnatomicalHeadHalfW(dy);
        if (hw < 3) {
          dots.push({
            x: 0, y: dy, role: 'rim', group: 'head_rim_center',
            isSecondary: false, size: 1.05, row: i, u: 0, phaseOff: i * 0.22,
          });
        } else {
          [-1, 1].forEach((side) => {
            dots.push({
              x: side * hw, y: dy, role: 'rim', group: `head_rim_${side}`,
              isSecondary: false, size: 1.0, row: i, u: side, phaseOff: i * 0.2 + side,
            });
            if (i % 2 === 0 && hw > 14) {
              dots.push({
                x: side * (hw - 4.5), y: dy, role: 'face_contour', group: `head_bevel_${side}`,
                isSecondary: false, size: 0.72, row: i, u: side * 0.92, phaseOff: i * 0.24 + side * 1.3,
              });
            }
          });
        }
      }

      // 2. High-Density 3D Sculpted Head Surface Point Cloud
      const numFaceRows = 27;
      for (let r = 0; r < numFaceRows; r++) {
        const tRow = (r + 0.6) / (numFaceRows + 0.2);
        const dy = -117 + tRow * 208;
        const hw = getAnatomicalHeadHalfW(dy);
        if (hw < 10) continue;

        const ny = (dy + 13) / 110;
        const numCols = Math.max(8, Math.round((hw * 2) / 8.2));
        const normCoreY = (dy - 14) / 56;

        for (let c = 1; c < numCols; c++) {
          const u = (c / numCols) * 2 - 1;
          const archY = getDotFacialReliefY(u, ny);
          const radialCoreSq = (u * u) / 0.44 + normCoreY * normCoreY;
          const inCore = radialCoreSq <= 1.0;
          const coreWeight = Math.max(0, 1 - radialCoreSq);

          dots.push({
            x: u * hw,
            y: dy + archY,
            role: inCore ? 'face_core' : 'face_contour',
            group: `face_row_${r}`,
            isSecondary: false,
            coreWeight,
            size: inCore ? 0.78 + coreWeight * 0.34 : 0.68,
            row: r,
            u,
            phaseOff: r * 0.34 + u * 2.6,
          });
        }
      }

      // 2B. Cylindrical 3D Neck Micro-Dot Rings
      for (let nr = 0; nr < 7; nr++) {
        const dy = 86 + nr * 6.2;
        const hw = getAnatomicalTorsoHalfW(dy);
        const cols = 9;
        for (let c = 1; c < cols; c++) {
          const u = (c / cols) * 2 - 1;
          const px = u * hw * 0.90;
          if (dy < 94 && Math.abs(px) < getAnatomicalHeadHalfW(dy) * 0.72) continue;
          dots.push({
            x: px, y: dy, role: 'shoulder_contour', group: `neck_ring_${nr}`,
            isSecondary: false, size: 0.66, row: nr, u, phaseOff: nr * 0.45 + u * 2.1,
          });
        }
      }

      // 3. Neck, Sloping Trapezius & Wide Rounded Shoulders Double-Bevel Rim
      const torsoRimSteps = 50;
      for (let i = 0; i <= torsoRimSteps; i++) {
        const t = i / torsoRimSteps;
        const dy = 66 + t * 194;
        const hw = getAnatomicalTorsoHalfW(dy);
        [-1, 1].forEach((side) => {
          if (dy < 94 && hw < getAnatomicalHeadHalfW(dy) + 1) return;
          dots.push({
            x: side * hw, y: dy, role: 'shoulder_rim', group: `torso_rim_${side}`,
            isSecondary: false, size: 1.0, row: i, u: side, phaseOff: i * 0.18 + side * 0.5,
          });
          if (dy >= 118 && i % 2 === 0) {
            dots.push({
              x: side * (hw - 5.5), y: dy + 1.5, role: 'shoulder_contour', group: `torso_bevel_${side}`,
              isSecondary: false, size: 0.72, row: i, u: side * 0.92, phaseOff: i * 0.22 + side,
            });
          }
        });
      }

      // 3B. Sculpted Clavicle, Rounded Deltoid Shoulder Dome & Upper-Torso Micro-Dot Mesh
      [-1, 1].forEach((side) => {
        for (let c = 0; c <= 18; c++) {
          const f = c / 18;
          const cx = side * (22 + f * 124);
          const cy = 136 + Math.sin(f * Math.PI * 0.65) * 10 + f * 32;
          if (Math.abs(cx) > getAnatomicalTorsoHalfW(cy) - 4) continue;
          dots.push({
            x: cx, y: cy, role: 'shoulder_rim', isSecondary: false,
            size: c % 4 === 0 ? 0.96 : 0.74, row: c, u: side * f, phaseOff: c * 0.28 + side * 0.4,
          });
        }

        for (let layer = 0; layer < 6; layer++) {
          const baseY = 166 + layer * 16;
          const steps = 11;
          for (let s = 0; s <= steps; s++) {
            const f = s / steps;
            const px = side * (108 + f * 66);
            const py = baseY + Math.sin(f * Math.PI * 0.55) * 12;
            if (py > 256 || Math.abs(px) > getAnatomicalTorsoHalfW(py) - 5) continue;
            dots.push({
              x: px, y: py, role: 'shoulder_contour', isSecondary: false,
              size: 0.70, row: layer, u: side * (0.66 + f * 0.34), phaseOff: layer * 0.42 + s * 0.25,
            });
          }
        }

        for (let pr = 0; pr < 7; pr++) {
          const py = 144 + pr * 16;
          const maxW = getAnatomicalTorsoHalfW(py) - 10;
          for (let pc = 0; pc < 6; pc++) {
            const px = side * (26 + pc * 17);
            if (Math.abs(px) > maxW) continue;
            if (Math.hypot(px, py - 204) < 30) continue;
            dots.push({
              x: px, y: py, role: 'shoulder_contour', isSecondary: false,
              size: 0.64, row: pr, u: side * (px / Math.max(1, maxW)), phaseOff: pr * 0.38 + pc * 0.31 + side,
            });
          }
        }
      });

      // 4. Body Circuit-Board Microchip Arrays
      [-1, 1].forEach((side) => {
        const chips = [
          { cx: side * 78, cy: 176, cols: 5, rows: 3, step: 5.2, isAccent: false },
          { cx: side * 124, cy: 184, cols: 3, rows: 3, step: 5.2, isAccent: true },
          { cx: side * 66, cy: 232, cols: 5, rows: 2, step: 5.2, isAccent: false },
        ];
        chips.forEach((chip, cIdx) => {
          for (let r = 0; r < chip.rows; r++) {
            for (let c = 0; c < chip.cols; c++) {
              const px = chip.cx + (c - (chip.cols - 1) * 0.5) * chip.step;
              const py = chip.cy + (r - (chip.rows - 1) * 0.5) * chip.step;
              if (Math.abs(px) > getAnatomicalTorsoHalfW(py) - 6) continue;
              const isEdge = r === 0 || r === chip.rows - 1 || c === 0 || c === chip.cols - 1;
              dots.push({
                x: px, y: py,
                role: chip.isAccent && !isEdge ? 'throat_core' : 'shoulder_contour',
                isSecondary: chip.isAccent && !isEdge,
                size: chip.isAccent && !isEdge ? 0.68 : (isEdge ? 0.82 : 0.95),
                row: r, u: side * 0.5, phaseOff: cIdx * 0.9 + r * 0.4 + c * 0.3,
              });
            }
          }
        });
      });

      // 5. Orthogonal & 45-Degree Neck-to-Collarbone Circuit Bus Tiny-Dot Matrix
      [-1, 1].forEach((side) => {
        const neckTracks = [
          [[14, 96], [14, 134], [36, 156], [62, 156]],
          [[24, 88], [24, 120], [48, 144], [92, 144], [116, 168]],
          [[32, 98], [32, 114], [58, 130], [110, 130], [136, 156]],
        ];
        neckTracks.forEach((track, tIdx) => {
          for (let s = 0; s < track.length - 1; s++) {
            const [x0, y0] = track[s];
            const [x1, y1] = track[s + 1];
            const segLen = Math.hypot(x1 - x0, y1 - y0);
            const count = Math.max(3, Math.round(segLen / 5.5));
            for (let k = 0; k <= count; k++) {
              const frac = k / count;
              const px = side * (x0 + (x1 - x0) * frac);
              const py = y0 + (y1 - y0) * frac;
              if (py < 100 && Math.abs(px) < getAnatomicalHeadHalfW(py) + 2) continue;
              if (Math.abs(px) > getAnatomicalTorsoHalfW(py) - 5) continue;
              const isBendNode = k === 0 || k === count;
              dots.push({
                x: px, y: py, role: 'shoulder_contour', isSecondary: false,
                size: isBendNode ? 1.05 : 0.72, row: tIdx, u: side * frac, phaseOff: tIdx * 0.7 + k * 0.35,
              });
            }
          }
        });
      });

      avState.avatarDots = dots;
    }

    function initBodyCircuits() {
      const halfTraces = [
        // 1. Upper Collarbone -> Outer Deltoid Shoulder
        { pts: [[16, 188], [16, 162], [38, 140], [108, 140], [138, 170], [138, 220]], isAccent: false, speed: 0.44 },
        // 2. Heart -> Mid Pectoral -> Outer Deltoid Cap
        { pts: [[22, 200], [56, 200], [78, 178], [122, 178], [156, 204], [166, 238]], isAccent: true, speed: 0.56 },
        // 3. Heart -> Upper Trapezius -> High Shoulder Crest
        { pts: [[16, 190], [34, 172], [62, 152], [104, 152], [134, 174], [154, 192]], isAccent: false, speed: 0.52 },
        // 4. Heart -> Direct Diagonal Shoulder Power Conduit
        { pts: [[20, 196], [48, 184], [86, 168], [126, 168], [152, 188], [168, 214]], isAccent: true, speed: 0.48 },
        // 5. Heart -> Clavicle Ridge -> Outer Shoulder Edge
        { pts: [[22, 192], [44, 170], [74, 170], [96, 148], [126, 156], [148, 178]], isAccent: false, speed: 0.50 },
        // 6. Heart -> Mid-Lower Chest -> Outer Arm / Shoulder Base
        { pts: [[22, 212], [58, 212], [84, 196], [128, 196], [158, 222], [170, 248]], isAccent: false, speed: 0.42 },
        // 7. Heart -> Lower Pectoral Rail -> Outer Bicep Bus
        { pts: [[22, 216], [48, 216], [64, 230], [110, 230], [136, 246]], isAccent: false, speed: 0.38 },
        // 8. Heart -> Deep Shoulder Sub-Bus
        { pts: [[18, 206], [42, 206], [68, 190], [104, 190], [136, 212], [154, 236]], isAccent: true, speed: 0.60 },
        // 9. Lower Torso Horizontal Bus
        { pts: [[14, 226], [34, 246], [88, 246], [104, 230]], isAccent: false, speed: 0.42 },
      ];

      const circuits = [];
      [-1, 1].forEach((side) => {
        halfTraces.forEach((tr, idx) => {
          const mirroredPts = tr.pts.map(([x, y]) => [side * x, y]);
          const segLens = [];
          let totalLen = 0;
          for (let i = 0; i < mirroredPts.length - 1; i++) {
            const dx = mirroredPts[i + 1][0] - mirroredPts[i][0];
            const dy = mirroredPts[i + 1][1] - mirroredPts[i][1];
            const len = Math.hypot(dx, dy);
            segLens.push(len);
            totalLen += len;
          }

          const traceDots = [];
          for (let i = 0; i < mirroredPts.length - 1; i++) {
            const [x0, y0] = mirroredPts[i];
            const [x1, y1] = mirroredPts[i + 1];
            const count = Math.max(2, Math.round(segLens[i] / 6.5));
            for (let k = 0; k <= count; k++) {
              const f = k / count;
              traceDots.push({
                x: x0 + (x1 - x0) * f,
                y: y0 + (y1 - y0) * f,
                isNode: k === 0 || k === count,
              });
            }
          }

          circuits.push({
            pts: mirroredPts,
            traceDots,
            segLens,
            totalLen,
            isAccent: tr.isAccent,
            speed: tr.speed,
            phaseOffset: (idx * 0.19 + (side > 0 ? 0.08 : 0)) % 1.0,
          });
        });
      });

      avState.bodyCircuits = circuits;
    }

    /**
     * Defines strictly 2 continuous orange (#B9F2FD) Heart-to-Brain lines (1 left, 1 right):
     * Heart (Arc Reactor) -> outer edge of shoulder -> outer edge of neck & head -> into the brain,
     * plus crisp 0°/45°/90° PCB circuit branches strictly inside the brain cranium (y <= -24).
     */
    function initBrainCircuits() {
      const halfTraces = [
        // 1. Single Continuous Heart -> Shoulder Edge -> Head Edge -> Brain Line (1 Left + 1 Right = 2 lines)
        {
          pts: [
            [18, 200], [66, 200], [114, 174], [144, 174],
            [112, 152], [68, 134], [36, 114], [31, 92],
            [38, 66], [52, 36], [60, 6], [62, -24],
            [58, -56], [48, -82], [32, -96], [16, -96],
          ],
          speed: 0.52,
          isPrimary: true,
        },
        // 2. Brain Upper Cortex Circuit Branch (branches inward from head edge into frontal/crown lobe)
        { pts: [[58, -56], [42, -56], [30, -68], [14, -68], [14, -86]], speed: 0.56, isPrimary: true },
        // 3. Brain Mid Cortex Circuit Branch (branches inward from temple edge into central brain)
        { pts: [[62, -24], [46, -40], [28, -40], [18, -50], [18, -76]], speed: 0.50, isPrimary: true },
        // 4. Brain Crown Synapse Bus (inside upper brain only)
        { pts: [[48, -82], [32, -82], [22, -92], [8, -92]], speed: 0.58, isPrimary: false },
      ];

      const circuits = [];
      [-1, 1].forEach((side) => {
        halfTraces.forEach((tr, idx) => {
          const mirroredPts = tr.pts.map(([x, y]) => [side * x, y]);
          const segLens = [];
          let totalLen = 0;
          for (let i = 0; i < mirroredPts.length - 1; i++) {
            const dx = mirroredPts[i + 1][0] - mirroredPts[i][0];
            const dy = mirroredPts[i + 1][1] - mirroredPts[i][1];
            const len = Math.hypot(dx, dy);
            segLens.push(len);
            totalLen += len;
          }

          const traceDots = [];
          for (let i = 0; i < mirroredPts.length - 1; i++) {
            const [x0, y0] = mirroredPts[i];
            const [x1, y1] = mirroredPts[i + 1];
            const count = Math.max(2, Math.round(segLens[i] / 5.5));
            for (let k = 0; k <= count; k++) {
              const f = k / count;
              traceDots.push({
                x: x0 + (x1 - x0) * f,
                y: y0 + (y1 - y0) * f,
                isNode: k === 0 || k === count,
              });
            }
          }

          circuits.push({
            pts: mirroredPts,
            traceDots,
            segLens,
            totalLen,
            isPrimary: tr.isPrimary,
            speed: tr.speed,
            phaseOffset: (idx * 0.17 + (side > 0 ? 0.09 : 0)) % 1.0,
          });
        });
      });

      avState.brainCircuits = circuits;
    }

    /**
     * Renders the 3D Orange (#B9F2FD) Neck-to-Brain Neural Circuit Network inside the avatar's head.
     */
    function drawBrainCircuits(rotY, rotX, rms, mid, high) {
      ctx.save();

      const projHead = (x, y, zOffset = 0) => {
        // Smooth 3D transition from chest/heart (y >= 115) up through neck into head (y <= 60)
        const headBlend = Math.max(0, Math.min(1, (115 - y) / 55));
        const zTorso = computeAvatarSurfaceZ(x, Math.max(66, y), false);
        const zHead = computeAvatarSurfaceZ(x, Math.min(88, y), true);
        const zSurf = zTorso * (1 - headBlend) + zHead * headBlend + zOffset;
        const rY = rotY * (0.65 + 0.35 * headBlend);
        const rX = rotX * (0.50 + 0.50 * headBlend);
        return project3DPoint(x, y, zSurf, rY, rX);
      };

      // Soft warm #B9F2FD cortical energy bloom inside the brain
      const cortexCenter = projHead(0, -56, 4);
      const cortexGlow = ctx.createRadialGradient(
        cortexCenter.sx, cortexCenter.sy, 4,
        cortexCenter.sx, cortexCenter.sy, (58 + rms * 22) * cortexCenter.scale
      );
      cortexGlow.addColorStop(0, `rgba(185, 242, 253, ${0.18 + rms * 0.28 + mid * 0.14})`);
      cortexGlow.addColorStop(0.55, `rgba(185, 242, 253, ${0.09 + rms * 0.14})`);
      cortexGlow.addColorStop(1, 'rgba(185, 242, 253, 0)');
      ctx.fillStyle = cortexGlow;
      ctx.beginPath();
      ctx.arc(cortexCenter.sx, cortexCenter.sy, (62 + rms * 22) * cortexCenter.scale, 0, Math.PI * 2);
      ctx.fill();

      for (let c = 0; c < avState.brainCircuits.length; c++) {
        const tr = avState.brainCircuits[c];
        const pts = tr.pts;

        // 1. Pure-dot glowing pathway & synapse via pads (tiny orange pinpoint dots, ZERO stroke lines!)
        for (let dIdx = 0; dIdx < tr.traceDots.length; dIdx++) {
          const td = tr.traceDots[dIdx];
          if (!td.isNode && dIdx % 2 !== 0) continue;
          const pp = projHead(td.x, td.y, 4);
          const r = (td.isNode ? 0.95 : 0.55) * pp.scale;
          ctx.beginPath();
          ctx.arc(pp.sx, pp.sy, r, 0, Math.PI * 2);
          ctx.fillStyle = td.isNode
            ? `rgba(185, 242, 253, ${0.68 + rms * 0.30})`
            : `rgba(185, 242, 253, ${0.34 + rms * 0.28})`;
          if (td.isNode) {
            ctx.shadowColor = '#B9F2FD';
            ctx.shadowBlur = 4 + rms * 4;
          }
          ctx.fill();
          ctx.shadowBlur = 0;
        }

        // 2. Three racing glowing 6-dot orange (#B9F2FD) energy comet pulses surging from Heart -> Brain
        const sampleAtDist = (dist) => {
          let acc = 0;
          for (let s = 0; s < tr.segLens.length; s++) {
            const sl = tr.segLens[s];
            if (dist <= acc + sl || s === tr.segLens.length - 1) {
              const frac = sl > 0 ? Math.max(0, Math.min(1, (dist - acc) / sl)) : 0;
              return [
                pts[s][0] + (pts[s + 1][0] - pts[s][0]) * frac,
                pts[s][1] + (pts[s + 1][1] - pts[s][1]) * frac,
              ];
            }
            acc += sl;
          }
          return pts[0];
        };

        const pulseCount = tr.isPrimary ? 3 : 2;
        for (let p = 0; p < pulseCount; p++) {
          const progress =
            (avState.phase * 0.125 * tr.speed * (1 + rms * 2.6 + high * 0.8) + tr.phaseOffset + p / pulseCount) % 1.0;
          const headDist = progress * tr.totalLen;

          for (let trail = 0; trail < 6; trail++) {
            const dPos = Math.max(0, headDist - trail * 3.4);
            const [px, py] = sampleAtDist(dPos);
            const pp = projHead(px, py, 6);
            const fade = 1.0 - trail * 0.15;
            const r = (1.10 - trail * 0.11 + rms * 0.42) * pp.scale;

            ctx.beginPath();
            ctx.arc(pp.sx, pp.sy, Math.max(0.42, r), 0, Math.PI * 2);
            ctx.fillStyle = trail === 0
              ? `rgba(187, 242, 252, ${Math.min(1.0, 0.96 + rms * 0.04)})`
              : `rgba(185, 242, 253, ${fade * (0.88 + rms * 0.12)})`;
            ctx.shadowColor = '#B9F2FD';
            ctx.shadowBlur = (trail === 0 ? 14 : 8) + rms * 7;
            ctx.fill();
          }
          ctx.shadowBlur = 0;
        }
      }

      ctx.shadowBlur = 0;
      ctx.restore();
    }

    function init3DEnvironmentMotes() {
      const motes = [];
      for (let i = 0; i < 72; i++) {
        motes.push({
          x: (Math.random() - 0.5) * 980,
          y: (Math.random() - 0.5) * 520,
          z: -80 + Math.random() * 680,
          speedZ: 0.45 + Math.random() * 1.15,
          size: 0.8 + Math.random() * 1.6,
          isAccent: Math.random() < 0.22,
          phaseOff: Math.random() * Math.PI * 2,
        });
      }
      avState.envMotes3D = motes;
    }

    function project3DPoint(x, y, z, rotY, rotX, fov = 560) {
      const cosY = Math.cos(rotY);
      const sinY = Math.sin(rotY);
      const x1 = x * cosY - z * sinY;
      const z1 = x * sinY + z * cosY;

      const cosX = Math.cos(rotX);
      const sinX = Math.sin(rotX);
      const y2 = y * cosX - z1 * sinX;
      const z2 = y * sinX + z1 * cosX;

      const persp = fov / Math.max(120, fov - z2);
      return { sx: x1 * persp, sy: y2 * persp, z: z2, scale: persp };
    }

    function computeAvatarSurfaceZ(x, y, isHead) {
      if (isHead) {
        const hw = Math.max(12, getAnatomicalHeadHalfW(y));
        const nx = Math.max(-1, Math.min(1, x / hw));
        const ny = Math.max(-1, Math.min(1, (y + 13) / 112));
        const zBase = Math.sqrt(Math.max(0, 1 - nx * nx * 0.90 - ny * ny * 0.30)) * 78;
        const noseBump = Math.exp(-Math.pow(nx / 0.16, 2)) * Math.exp(-Math.pow((ny - 0.05) / 0.24, 2)) * 15;
        const cheekBump = Math.exp(-Math.pow((Math.abs(nx) - 0.50) / 0.26, 2)) * Math.exp(-Math.pow((ny - 0.14) / 0.28, 2)) * 9;
        const chinBump = Math.exp(-Math.pow(nx / 0.30, 2)) * Math.exp(-Math.pow((ny - 0.72) / 0.22, 2)) * 11;
        return zBase + noseBump + cheekBump + chinBump;
      } else {
        const hw = Math.max(20, getAnatomicalTorsoHalfW(y));
        const nx = Math.max(-1, Math.min(1, x / hw));
        const chestCurve = Math.sqrt(Math.max(0, 1 - nx * nx * 0.86)) * 60;
        const clavicleBump = Math.exp(-Math.pow((y - 126) / 14, 2)) * Math.exp(-Math.pow((Math.abs(nx) - 0.42) / 0.35, 2)) * 9;
        const pectoralBump = Math.exp(-Math.pow((y - 184) / 34, 2)) * Math.exp(-Math.pow((Math.abs(nx) - 0.36) / 0.32, 2)) * 11;
        const deltoidBump = Math.exp(-Math.pow((y - 168) / 36, 2)) * Math.exp(-Math.pow((Math.abs(nx) - 0.84) / 0.18, 2)) * 10;
        return chestCurve + clavicleBump + pectoralBump + deltoidBump - 6;
      }
    }

    function drawIronManBodyCircuitsAndArcReactor(rotY, rotX, rms, bass) {
      ctx.save();
      const projBody = (x, y, zOffset = 0) => {
        const zSurf = computeAvatarSurfaceZ(x, y, false) + zOffset;
        return project3DPoint(x, y, zSurf, rotY, rotX);
      };

      for (let c = 0; c < avState.bodyCircuits.length; c++) {
        const tr = avState.bodyCircuits[c];
        const pts = tr.pts;

        for (let dIdx = 0; dIdx < tr.traceDots.length; dIdx++) {
          const td = tr.traceDots[dIdx];
          const pp = projBody(td.x, td.y, 2);
          const r = tr.isAccent
            ? (td.isNode ? 0.95 : 0.55) * pp.scale
            : (td.isNode ? 1.7 : 0.95) * pp.scale;
          ctx.beginPath();
          ctx.arc(pp.sx, pp.sy, r, 0, Math.PI * 2);
          ctx.fillStyle = tr.isAccent
            ? `rgba(185, 242, 253, ${(td.isNode ? 0.62 : 0.34) + rms * 0.30})`
            : `rgba(42, 172, 255, ${(td.isNode ? 0.62 : 0.34) + rms * 0.28})`;
          ctx.fill();
        }

        const sampleAtDist = (dist) => {
          let acc = 0;
          for (let s = 0; s < tr.segLens.length; s++) {
            const sl = tr.segLens[s];
            if (dist <= acc + sl || s === tr.segLens.length - 1) {
              const frac = sl > 0 ? Math.max(0, Math.min(1, (dist - acc) / sl)) : 0;
              return [
                pts[s][0] + (pts[s + 1][0] - pts[s][0]) * frac,
                pts[s][1] + (pts[s + 1][1] - pts[s][1]) * frac,
              ];
            }
            acc += sl;
          }
          return pts[0];
        };

        for (let p = 0; p < 2; p++) {
          const progress =
            (avState.phase * 0.088 * tr.speed * (1 + rms * 2.5) + tr.phaseOffset + p * 0.5) % 1.0;
          const headDist = progress * tr.totalLen;

          for (let trail = 0; trail < 4; trail++) {
            const dPos = Math.max(0, headDist - trail * 4.5);
            const [px, py] = sampleAtDist(dPos);
            const pp = projBody(px, py, 4);
            const fade = 1.0 - trail * 0.22;
            const r = tr.isAccent
              ? (1.05 - trail * 0.14 + rms * 0.40) * pp.scale
              : (1.75 - trail * 0.28 + rms * 0.95) * pp.scale;

            ctx.beginPath();
            ctx.arc(pp.sx, pp.sy, Math.max(0.42, r), 0, Math.PI * 2);
            ctx.fillStyle = tr.isAccent
              ? `rgba(185, 242, 253, ${fade * (0.85 + rms * 0.15)})`
              : `rgba(${130 + (trail === 0 ? 110 : 0)}, 238, 255, ${fade * (0.85 + rms * 0.15)})`;
            ctx.shadowColor = tr.isAccent ? '#B9F2FD' : '#38d6ff';
            ctx.shadowBlur = (trail === 0 ? 12 : 6) + rms * 6;
            ctx.fill();
          }
          ctx.shadowBlur = 0;
        }
      }

      // 3D Pure-Dot Rotating Arc Reactor Core at (0, 204)
      const reactorPulse = 1.0 + rms * 0.14 + bass * 0.08;
      const arcCenter = projBody(0, 204, 6);
      const rx = arcCenter.sx;
      const ry = arcCenter.sy;
      const rScale = arcCenter.scale * reactorPulse;

      const arcGlow = ctx.createRadialGradient(rx, ry, 2, rx, ry, 32 * rScale);
      arcGlow.addColorStop(0, `rgba(200, 248, 255, ${0.58 + rms * 0.35})`);
      arcGlow.addColorStop(0.45, `rgba(38, 185, 255, ${0.32 + rms * 0.32})`);
      arcGlow.addColorStop(1, 'rgba(6, 46, 82, 0)');
      ctx.fillStyle = arcGlow;
      ctx.beginPath();
      ctx.arc(rx, ry, 34 * rScale, 0, Math.PI * 2);
      ctx.fill();

      const cosYaw = Math.max(0.55, Math.cos(rotY));
      const rotAngle = avState.phase * 0.42;

      for (let k = 0; k < 20; k++) {
        const a = rotAngle + (k / 20) * Math.PI * 2;
        const dx = Math.cos(a) * 18.5 * rScale * cosYaw;
        const dy = Math.sin(a) * 18.5 * rScale;
        const isMajor = k % 2 === 0;
        ctx.beginPath();
        ctx.arc(rx + dx, ry + dy, (isMajor ? 1.8 : 1.15) * rScale, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(135, 235, 255, ${(isMajor ? 0.92 : 0.65) + rms * 0.15})`;
        ctx.shadowColor = '#38d6ff';
        ctx.shadowBlur = 7 + rms * 6;
        ctx.fill();
      }

      for (let k = 0; k < 10; k++) {
        const a = -rotAngle * 1.4 + (k / 10) * Math.PI * 2;
        const dx = Math.cos(a) * 10.5 * rScale * cosYaw;
        const dy = Math.sin(a) * 10.5 * rScale;
        ctx.beginPath();
        ctx.arc(rx + dx, ry + dy, 1.35 * rScale, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(220, 252, 255, ${0.88 + rms * 0.12})`;
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(rx, ry, 4.2 * rScale, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(240, 253, 255, ${0.92 + rms * 0.08})`;
      ctx.shadowColor = '#7ce8ff';
      ctx.shadowBlur = 12 + rms * 8;
      ctx.fill();

      ctx.shadowBlur = 0;
      ctx.restore();
    }

    function drawBalancedSignalWaveBackground(W, H, rms, bass) {
      ctx.save();
      ctx.clearRect(0, 0, W, H);

      const vpX = W * 0.5;
      const vpY = H * 0.36;
      const fov = 460;

      // Clean, soft volumetric ambient glow behind the avatar (fully transparent edges)
      const glowR = Math.min(W, H) * 0.42;
      const horizonGlow = ctx.createRadialGradient(vpX, vpY, 12, vpX, vpY, glowR);
      horizonGlow.addColorStop(0, `rgba(38, 175, 255, ${0.14 + rms * 0.16})`);
      horizonGlow.addColorStop(0.55, `rgba(6, 46, 82, ${0.08 + bass * 0.06})`);
      horizonGlow.addColorStop(1, 'rgba(2, 11, 22, 0)');
      ctx.fillStyle = horizonGlow;
      ctx.fillRect(0, 0, W, H);

      // Minimal delicate floating depth particles (28 subtle light motes for clean 3D depth)
      const count = Math.min(28, avState.envMotes3D.length);
      for (let i = 0; i < count; i++) {
        const m = avState.envMotes3D[i];
        m.z -= m.speedZ * (0.65 + rms * 1.6);
        if (m.z < -40) {
          m.z = 620;
          m.x = (Math.random() - 0.5) * 760;
          m.y = (Math.random() - 0.5) * 420;
        }
        const p = fov / Math.max(90, fov + m.z);
        const sx = vpX + m.x * p;
        const sy = vpY + m.y * p;
        if (sx < 0 || sx > W || sy < 0 || sy > H) continue;

        const depthAlpha = Math.max(0.04, Math.min(0.45, (1 - m.z / 650) * (0.35 + rms * 0.3)));
        ctx.beginPath();
        ctx.arc(sx, sy, m.isAccent ? Math.max(0.4, m.size * p * 0.55) : Math.max(0.5, m.size * p * 1.05), 0, Math.PI * 2);
        ctx.fillStyle = m.isAccent
          ? `rgba(185, 242, 253, ${depthAlpha})`
          : `rgba(110, 215, 255, ${depthAlpha})`;
        ctx.fill();
      }

      ctx.restore();
    }

    function drawAnimatedDotAvatar(W, H, rms, bass, mid, high, isSpeaking) {
      ctx.save();

      // Responsive anatomical scale framed cleanly in the open center above the bottom chat dock
      const fitScale = Math.max(0.75, Math.min(1.58, Math.min((H * 0.76) / 390, (W * 0.44) / 365) * 1.06));
      const avatarScale = fitScale + Math.sin(avState.phase * 0.45) * 0.006 + rms * 0.022;
      const headX = W * 0.5;
      const headY = H * 0.34 + Math.sin(avState.phase * 0.45) * 1.6;

      const rotY = Math.sin(avState.phase * 0.28) * (0.24 + rms * 0.08);
      const rotX = Math.sin(avState.phase * 0.55) * 0.06 + (isSpeaking ? Math.sin(avState.phase * 1.6) * rms * 0.09 : 0);

      ctx.translate(headX, headY);
      ctx.scale(avatarScale, avatarScale);

      drawIronManBodyCircuitsAndArcReactor(rotY * 0.65, rotX * 0.5, rms, bass);

      const headGlowCenter = project3DPoint(0, 6, 40, rotY, rotX);
      const headGlowAlpha = 0.08 + rms * 0.42 + mid * 0.22;
      const headGlow = ctx.createRadialGradient(
        headGlowCenter.sx, headGlowCenter.sy, 4,
        headGlowCenter.sx, headGlowCenter.sy, 76 + rms * 28
      );
      headGlow.addColorStop(0, `rgba(110, 220, 255, ${headGlowAlpha})`);
      headGlow.addColorStop(0.5, `rgba(24, 145, 245, ${headGlowAlpha * 0.48})`);
      headGlow.addColorStop(1, 'rgba(6, 46, 82, 0)');
      ctx.fillStyle = headGlow;
      ctx.beginPath();
      ctx.arc(headGlowCenter.sx, headGlowCenter.sy, 98 + rms * 22, 0, Math.PI * 2);
      ctx.fill();

      const projectedDots = [];
      for (let i = 0; i < avState.avatarDots.length; i++) {
        const d = avState.avatarDots[i];
        const isHeadRole = d.role === 'face_core' || d.role === 'face_contour' || d.role === 'rim';

        let dx = Math.sin(avState.phase * 1.15 + d.phaseOff) * (0.75 + rms * 1.8);
        let dy = Math.cos(avState.phase * 0.95 + d.phaseOff * 0.9) * (0.75 + rms * 1.6);
        let dz = 0;
        let dotRadius = d.size;
        let speechFlash = 0;

        if (d.role === 'face_core' || d.role === 'face_contour') {
          const distFromCore = Math.hypot(d.x, d.y - 12);
          const radialWave = Math.sin(distFromCore * 0.078 - avState.phase * 3.4);
          const wavePush = radialWave * (rms * 5.2 + mid * 3.2);
          const normAngle = Math.atan2(d.y - 12, d.x);
          dx += Math.cos(normAngle) * wavePush;
          dy += Math.sin(normAngle) * wavePush;
          dz += radialWave * (rms * 12 + mid * 7);

          if (d.role === 'face_core') {
            const w = d.coreWeight || 0.5;
            dx += Math.sin(avState.phase * 2.8 + d.row * 0.65) * (rms * 5.5 + mid * 3.5) * w;
            dy += Math.sin(d.u * 6.5 - avState.phase * 3.2 + d.row * 0.6) * (0.9 + rms * 7.2 + mid * 4.5) * w;
            speechFlash = Math.min(1.0, (rms * 2.6 + mid * 1.3 + Math.max(0, radialWave) * rms * 1.5) * (0.48 + 0.52 * w));
            dotRadius = d.size * (1.0 + (0.15 + rms * 0.75 + mid * 0.38) * w);
          } else {
            dy += Math.sin(d.u * 5.5 - avState.phase * 2.0 + d.row * 0.45) * (0.8 + rms * 3.2);
            speechFlash = Math.min(1.0, Math.max(0, radialWave) * (rms * 2.1 + high * 0.75));
            dotRadius = d.size * (1.0 + rms * 0.38 + speechFlash * 0.28);
          }
        } else if (d.role === 'rim') {
          const rimWave = Math.sin(d.row * 0.5 - avState.phase * 3.0);
          const pulseOut = 1.0 + Math.sin(avState.phase * 1.2 + d.phaseOff) * 0.011 + rms * 0.052 + rimWave * rms * 0.022;
          dx += d.x * (pulseOut - 1.0);
          dy += d.y * (pulseOut - 1.0) * 0.42;
          speechFlash = Math.min(1.0, rms * 1.85 + high * 0.8 + Math.max(0, rimWave) * rms * 1.1);
          dotRadius = d.size * (1.0 + rms * 0.48 + high * 0.22);
        } else if (d.role === 'throat_core') {
          const travelWave = Math.sin(d.throatT * 9 - avState.phase * 2.8);
          dx += travelWave * (1.2 + rms * 4.6 + bass * 2.8);
          dy -= ((avState.phase * 9 * (1 + rms * 1.6) + d.row * 10) % 12) - 6;
          dz = 6;
          dotRadius = d.size * (1.0 + Math.max(0, travelWave) * (0.20 + rms * 0.60));
        } else if (d.role === 'shoulder_rim') {
          const pulseOut = 1.0 + Math.sin(avState.phase * 1.2 + d.phaseOff) * 0.010 + rms * 0.030;
          dx += d.x * (pulseOut - 1.0);
          dy += d.y * (pulseOut - 1.0) * 0.32;
          dotRadius = d.size * (1.0 + rms * 0.28 + high * 0.14);
        } else {
          dy += Math.sin(d.u * 5.2 - avState.phase * 1.8 + d.row * 0.45) * (0.9 + rms * 3.0);
          dotRadius = d.size * (1.0 + rms * 0.24);
        }

        const baseZ = computeAvatarSurfaceZ(d.x, d.y, isHeadRole) + dz;
        const p3 = project3DPoint(
          d.x + dx, d.y + dy, baseZ,
          isHeadRole ? rotY : rotY * 0.65,
          isHeadRole ? rotX : rotX * 0.5
        );

        projectedDots.push({
          px: p3.sx, py: p3.sy, z: p3.z,
          r: dotRadius * p3.scale,
          depthShade: Math.max(0.45, Math.min(1.15, (p3.z + 65) / 120)),
          speechFlash, d,
        });
      }

      projectedDots.sort((a, b) => a.z - b.z);

      for (let i = 0; i < projectedDots.length; i++) {
        const { px, py, r, depthShade, speechFlash, d } = projectedDots[i];
        const twinkle = 0.5 + 0.5 * Math.sin(avState.phase * 2.4 + d.phaseOff * 3.1);
        const glowBoost = twinkle * (0.28 + rms * 0.72);
        const haloR = r * (1.85 + glowBoost * 0.7 + speechFlash * 0.65);

        ctx.beginPath();
        ctx.arc(px, py, haloR, 0, Math.PI * 2);

        if (d.role === 'throat_core') {
          ctx.fillStyle = `rgba(185, 242, 253, ${(0.14 + glowBoost * 0.22 + rms * 0.18) * depthShade})`;
          ctx.fill();

          ctx.beginPath();
          ctx.arc(px, py, r, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(185, 242, 253, ${Math.min(1.0, (0.82 + rms * 0.18) * depthShade)})`;
          ctx.shadowColor = '#B9F2FD';
          ctx.shadowBlur = 5 + rms * 6 + glowBoost * 4;
          ctx.fill();

          if (twinkle > 0.55 || rms > 0.08) {
            ctx.shadowBlur = 0;
            ctx.beginPath();
            ctx.arc(px, py, r * 0.45, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(187, 242, 252, ${(0.55 + glowBoost * 0.45) * depthShade})`;
            ctx.fill();
          }
        } else if (d.role === 'face_core' || d.role === 'face_contour' || d.role === 'rim') {
          ctx.fillStyle = `rgba(${Math.round(70 + speechFlash * 150)}, ${Math.round(
            195 + speechFlash * 60
          )}, 255, ${(0.12 + glowBoost * 0.18 + speechFlash * 0.24) * depthShade})`;
          ctx.fill();

          ctx.beginPath();
          ctx.arc(px, py, r, 0, Math.PI * 2);
          const rCol = Math.min(255, Math.round((45 + speechFlash * 190 + glowBoost * 35) * depthShade));
          const gCol = Math.min(255, Math.round((170 + speechFlash * 85 + glowBoost * 30) * depthShade));
          const bCol = Math.min(255, Math.round(255 * Math.min(1, depthShade + 0.12)));
          const alpha = Math.min(1.0, ((d.role === 'rim' ? 0.90 : 0.78) + speechFlash * 0.22) * depthShade);
          ctx.fillStyle = `rgba(${rCol}, ${gCol}, ${bCol}, ${alpha})`;
          ctx.shadowColor = speechFlash > 0.30 || twinkle > 0.75 ? '#a5ecff' : '#1e96ff';
          ctx.shadowBlur = ((d.role === 'rim' ? 6 : 4) + speechFlash * 9 + glowBoost * 4) * depthShade;
          ctx.fill();

          if (twinkle > 0.62 || speechFlash > 0.28) {
            ctx.shadowBlur = 0;
            ctx.beginPath();
            ctx.arc(px, py, r * 0.48, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(235, 252, 255, ${Math.min(1.0, (0.45 + speechFlash * 0.55 + glowBoost * 0.35) * depthShade)})`;
            ctx.fill();
          }
        } else if (d.role === 'shoulder_rim') {
          ctx.fillStyle = `rgba(75, 195, 255, ${(0.13 + glowBoost * 0.18) * depthShade})`;
          ctx.fill();

          ctx.beginPath();
          ctx.arc(px, py, r, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(75, 195, 255, ${Math.min(1.0, (0.88 + rms * 0.12) * depthShade)})`;
          ctx.shadowColor = '#1e96ff';
          ctx.shadowBlur = 6 + rms * 5 + glowBoost * 3;
          ctx.fill();

          if (twinkle > 0.68) {
            ctx.shadowBlur = 0;
            ctx.beginPath();
            ctx.arc(px, py, r * 0.45, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(220, 248, 255, ${0.65 * depthShade})`;
            ctx.fill();
          }
        } else {
          ctx.fillStyle = `rgba(42, 168, 255, ${(0.10 + glowBoost * 0.15) * depthShade})`;
          ctx.fill();

          ctx.beginPath();
          ctx.arc(px, py, r, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(38, 158, 252, ${Math.min(1.0, (0.76 + rms * 0.22) * depthShade)})`;
          ctx.shadowColor = '#0b5da8';
          ctx.shadowBlur = 3 + glowBoost * 3;
          ctx.fill();
        }
      }

      ctx.shadowBlur = 0;

      // 3D Orange (#B9F2FD) Neck-to-Brain Neural Circuit Network inside the avatar's head
      drawBrainCircuits(rotY, rotX, rms, mid, high);

      ctx.restore();
    }

    /**
     * Clean Minimalist Voice-Capture Indicator: 16-bar reactive wave + crisp recording label
     */
    function drawVoiceCaptureIndicator(W, H, micRms, activelyCapturing) {
      ctx.save();
      ctx.globalAlpha = avState.captureAlpha;

      const cx = W * 0.5;
      const barCount = 16;
      const span = Math.min(200, W * 0.24);
      const baseY = Math.min(H - 134, H * 0.78);

      for (let i = 0; i < barCount; i++) {
        const normPos = (i / (barCount - 1)) * 2 - 1;
        const bell = Math.cos(normPos * Math.PI * 0.5);
        const target = activelyCapturing
          ? Math.max(0.15, (0.24 + micRms * 1.45) * bell * (0.55 + 0.45 * Math.sin(avState.phase * 5.8 + i * 0.8)))
          : Math.max(0.08, (0.12 + micRms * 0.8) * bell * (0.65 + 0.35 * Math.sin(avState.phase * 2.8 + i * 0.5)));
        avState.captureBars[i] += (target - avState.captureBars[i]) * 0.32;

        const bx = cx + normPos * (span * 0.5);
        const hBar = Math.min(26, avState.captureBars[i] * 28);
        ctx.beginPath();
        ctx.moveTo(bx, baseY - hBar * 0.5);
        ctx.lineTo(bx, baseY + hBar * 0.5);
        ctx.strokeStyle = activelyCapturing ? 'rgba(185, 242, 253, 0.92)' : 'rgba(56, 189, 248, 0.72)';
        ctx.lineWidth = 2.6;
        ctx.lineCap = 'round';
        ctx.shadowColor = activelyCapturing ? '#B9F2FD' : '#38bdf8';
        ctx.shadowBlur = 8;
        ctx.stroke();
      }
      ctx.shadowBlur = 0;

      const pulse = 0.55 + 0.45 * Math.sin(avState.phase * 5.5);
      const labelText = activelyCapturing ? 'CAPTURING VOICE' : 'LISTENING...';
      ctx.font = "700 10.5px 'Consolas', 'Courier New', monospace";
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const labelY = baseY + 20;
      const textW = ctx.measureText(labelText).width;

      ctx.beginPath();
      ctx.arc(cx - textW * 0.5 - 9, labelY, 3.0, 0, Math.PI * 2);
      ctx.fillStyle = activelyCapturing ? `rgba(185, 242, 253, ${0.65 + pulse * 0.35})` : `rgba(56, 189, 248, ${0.55 + pulse * 0.45})`;
      ctx.shadowColor = activelyCapturing ? '#B9F2FD' : '#38bdf8';
      ctx.shadowBlur = 9;
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.fillStyle = activelyCapturing ? '#BBF2FC' : '#bae6fd';
      ctx.fillText(labelText, cx + 4, labelY);

      ctx.restore();
    }

    initAnimatedAvatarDots();
    initBodyCircuits();
    initBrainCircuits();
    init3DEnvironmentMotes();

    let lastNow = performance.now();
    const energyBadge = document.getElementById('avatarEnergyBadge');

    function renderFrame(now) {
      requestAnimationFrame(renderFrame);
      const dt = Math.min(0.05, Math.max(0.004, (now - lastNow) / 1000.0));
      lastNow = now;

      const W = avState.logicalW;
      const H = avState.logicalH;
      const uiState = document.body.getAttribute('data-state') || 'idle';
      const isSpeaking = uiState === 'speaking';
      const isThinking = uiState === 'thinking';
      const isListening = uiState === 'listening';
      const showVoiceCapture = (isListening || isCapturingVoice) && !isSpeaking && !isThinking;

      const thinkTarget = isThinking ? 1.0 : 0.0;
      avState.thinkAlpha += (thinkTarget - avState.thinkAlpha) * Math.min(1, dt * 7.5);
      const captureTarget = showVoiceCapture ? 1.0 : 0.0;
      avState.captureAlpha += (captureTarget - avState.captureAlpha) * Math.min(1, dt * 9.0);

      let rms = 0;
      let bass = 0;
      let mid = 0;
      let high = 0;
      const freshMic = (now - liveMicRmsTs) < 550 ? liveMicRms : 0;

      if (isThinking) {
        rms = 0.14 + 0.06 * Math.sin(avState.phase * 2.6);
        bass = 0.18 + 0.08 * Math.sin(avState.phase * 1.9);
        mid = 0.22 + 0.08 * Math.cos(avState.phase * 2.3);
        high = 0.20 + 0.10 * Math.sin(avState.phase * 3.1);
      } else if (isSpeaking) {
        let gotLiveAnalyser = false;
        if (avatarAnalyser && avatarFreqData && avatarTimeData && currentAudioSource) {
          avatarAnalyser.getByteFrequencyData(avatarFreqData);
          avatarAnalyser.getByteTimeDomainData(avatarTimeData);
          let sumSq = 0;
          for (let i = 0; i < avatarTimeData.length; i++) {
            const norm = (avatarTimeData[i] - 128) / 128.0;
            sumSq += norm * norm;
          }
          const measuredRms = Math.sqrt(sumSq / avatarTimeData.length);
          if (measuredRms >= 0.012) {
            gotLiveAnalyser = true;
            rms = measuredRms;
            for (let i = 2; i < 16; i++) bass += avatarFreqData[i];
            for (let i = 16; i < 75; i++) mid += avatarFreqData[i];
            for (let i = 75; i < 190; i++) high += avatarFreqData[i];
            bass /= 14 * 255;
            mid /= 59 * 255;
            high /= 115 * 255;
          }
        }
        if (!gotLiveAnalyser && (now - liveTtsRmsTs) < 250 && liveTtsRms > 0.01) {
          gotLiveAnalyser = true;
          rms = Math.min(0.55, liveTtsRms * 0.55);
          bass = rms * 1.1;
          mid = rms * (1.05 + 0.25 * Math.cos(avState.phase * 4.0));
          high = rms * (0.85 + 0.3 * Math.sin(avState.phase * 5.5));
        }
        if (!gotLiveAnalyser) {
          const syl = Math.max(0.06, 0.22 + 0.14 * Math.sin(avState.phase * 3.1) + 0.09 * Math.sin(avState.phase * 5.7));
          rms = syl;
          bass = syl * 1.15;
          mid = syl * (1.1 + 0.3 * Math.cos(avState.phase * 4.2));
          high = syl * (0.85 + 0.35 * Math.sin(avState.phase * 6.4));
        }
      } else if (showVoiceCapture) {
        const micBoost = Math.min(0.48, freshMic * 1.65);
        rms = Math.max(isCapturingVoice ? 0.12 : 0.06 + 0.03 * Math.sin(avState.phase * 2.1), micBoost);
        bass = rms * 1.1;
        mid = rms * 0.95;
        high = rms * 0.85;
      }

      const smoothFollow = (cur, target, atkTau, relTau) => {
        const tau = target > cur ? atkTau : relTau;
        const alpha = 1.0 - Math.exp(-dt / tau);
        return cur + (target - cur) * alpha;
      };

      avState.smoothRms = smoothFollow(avState.smoothRms, rms, 0.045, 0.16);
      avState.smoothBass = smoothFollow(avState.smoothBass, bass, 0.055, 0.18);
      avState.smoothMid = smoothFollow(avState.smoothMid, mid, 0.045, 0.15);
      avState.smoothHigh = smoothFollow(avState.smoothHigh, high, 0.050, 0.16);

      const sRms = avState.smoothRms;
      const sBass = avState.smoothBass;
      const sMid = avState.smoothMid;
      const sHigh = avState.smoothHigh;

      if (energyBadge) {
        if (isSpeaking) {
          const estFreq = Math.round(115 + sMid * 245 + sHigh * 180);
          energyBadge.textContent = `VOCAL ENERGY: ${Math.min(100, Math.round(sRms * 265))}% • ~${estFreq} Hz`;
        } else if (isThinking) {
          energyBadge.textContent = 'NEURAL PROCESSING...';
        } else if (isCapturingVoice) {
          energyBadge.textContent = `CAPTURING VOICE • ${Math.min(100, Math.round(Math.max(freshMic, sRms) * 180))}%`;
        } else if (isListening) {
          energyBadge.textContent = `MIC STANDBY • ${Math.min(100, Math.round(sRms * 220))}%`;
        } else {
          energyBadge.textContent = 'NEURAL AVATAR // ONLINE';
        }
      }

      avState.phase += dt * (1.75 + sRms * 2.8);

      drawBalancedSignalWaveBackground(W, H, sRms, sBass);
      drawAnimatedDotAvatar(W, H, sRms, sBass, sMid, sHigh, isSpeaking);
      if (avState.captureAlpha > 0.01) {
        drawVoiceCaptureIndicator(W, H, freshMic, isCapturingVoice);
      }
    }

    requestAnimationFrame(renderFrame);
  })();
