import os
import re
import io
import sys
import time
import wave
import base64
import queue
import threading
import datetime
import argparse
import subprocess
import urllib.request
from pathlib import Path
from typing import Callable, Optional

# Ensure all NVIDIA CUDA 12 DLLs from pip packages are discoverable on Windows BEFORE importing onnxruntime/ctranslate2
_nvidia_root = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
if os.path.exists(_nvidia_root):
    for _pkg in os.listdir(_nvidia_root):
        _dll_path = os.path.join(_nvidia_root, _pkg, "bin")
        if os.path.exists(_dll_path):
            try:
                os.add_dll_directory(_dll_path)
            except Exception:
                pass
            os.environ["PATH"] = _dll_path + os.pathsep + os.environ["PATH"]

import numpy as np
import psutil
import sounddevice as sd
import ollama
import onnxruntime as ort
from math import gcd
from scipy import signal as sp_signal
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps
from openwakeword.model import Model as WakeWordModel

# Silence verbose ONNX Runtime CUDA warnings
ort.set_default_logger_severity(3)

# ================= CONFIGURATION =================
BASE_DIR = Path(__file__).resolve().parent
LLM_MODEL = os.environ.get("CODERSTATION_LLM_MODEL", "llama3.1:8b")
WHISPER_MODEL = os.environ.get("CODERSTATION_WHISPER_MODEL", "small.en")
TTS_VOICE = os.environ.get("CODERSTATION_TTS_VOICE", "bm_george")
WAKE_WORD = "hey_jarvis"

SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000            # Native Kokoro v1.0 output sample rate
SILENCE_THRESHOLD = 0.020          # Above Intel Smart Sound AGC noise floor (~0.015)
SILENCE_DURATION = 1.10            # Fast 1.1s pause cutoff after user stops talking for snappy voice replies
STANDBY_SILENCE_DURATION = 0.65    # Ultra-fast 650ms cutoff during passive wake-phrase standby
MIN_SPEECH_DURATION = 0.28         # Require at least 280ms of speech before allowing silence cutoff
INITIAL_WAIT_SECONDS = 4.5         # Wait up to 4.5s for user to begin speaking
FOLLOW_UP_WINDOW_SECONDS = 5.0     # After AI response finishes, listen 5.0s for follow-up without requiring wake phrase
MAX_RECORD_SECONDS = 25.0          # Maximum duration for a complete voice command

# 85 Hz 4th-order Butterworth high-pass filter removes AC hum, desk rumble & plosives before VAD/Whisper
_HP_SOS = sp_signal.butter(4, 85.0, btype="highpass", fs=SAMPLE_RATE, output="sos")

WHISPER_INITIAL_PROMPT = (
    "Coderstation, Hey Coderstation, Sir, CPU, RAM, GPU, VRAM, system status, memory utilization, "
    "what time is it, what is the date, open notepad, calculator, chrome, browser, "
    "edge, explorer, terminal, task manager, Python, Ollama, Whisper, Kokoro, artificial intelligence."
)
WHISPER_HALLUCINATIONS = {
    "you", "thank you", "thank you.", "thank you so much", "thanks", "thanks.",
    "thanks for watching", "thanks for watching.", "thank you for watching", "thank you for watching.",
    "bye", "bye.", "goodbye", "subtitles by", "amara.org", "mbc", "uh", "um", "hmm", "mm-hmm",
}

WAKE_PHRASE = "Coderstation"
# Regex matching "Coderstation" and common phonetic Whisper spellings at the start or anywhere in an utterance
WAKE_PHRASE_PATTERN = re.compile(
    r"\b(?:hey\s+|hi\s+|hello\s+|ok\s+|okay\s+)?"
    r"(?:coder\s*station|code\s*station|coders\s*station|coder\s*stations|"
    r"quarter\s*station|colder\s*station|coda\s*station|quota\s*station|"
    r"cora\s*station|koder\s*station|color\s*station)\b"
    r"[\s,.:;!?-]*",
    re.IGNORECASE,
)


def extract_wake_command(text: str) -> tuple[bool, str]:
    """
    Check if spoken text contains the 'Coderstation' wake phrase.
    Returns `(triggered: bool, command_after_wake_phrase: str)`.
    - If user says 'Coderstation what time is it?', returns `(True, 'what time is it?')`
    - If user says just 'Coderstation', returns `(True, '')` so the assistant prompts and records the command next.
    """
    if not text:
        return False, ""
    match = WAKE_PHRASE_PATTERN.search(text)
    if not match:
        return False, ""
    # Extract everything spoken after the wake phrase (or before if wake word was at the end)
    after = text[match.end() :].strip(" ,.:;!?-")
    if after:
        return True, after
    before = text[: match.start()].strip(" ,.:;!?-")
    # Ignore pure greeting prefixes like 'hey' or 'hello' if they were the only thing before 'Coderstation'
    if before.lower() in ("hey", "hi", "hello", "ok", "okay", "yo"):
        return True, ""
    return True, before

KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
KOKORO_MODEL_PATH = BASE_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = BASE_DIR / "voices-v1.0.bin"

SYSTEM_PROMPT = (
    "You are Coderstation, a hyper-intelligent, witty, and concise AI assistant running locally. "
    "Always address the user respectfully as 'Sir'. "
    "If asked who developed you or who your developer is, answer: This system was developed by Professor Armando  Saguin in October 2026. "
    "Keep your responses brief (1 to 2 crisp sentences unless asked for detail), natural, and optimized for spoken audio—avoid markdown, bullet points, or code blocks unless asked."
)


# ================= TOOLS (FUNCTION CALLING & ZERO-LATENCY FAST PATH) =================
def get_system_status() -> str:
    """Get current CPU, RAM, and disk status."""
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    ram_used_gb = round(ram.used / (1024**3), 1)
    ram_total_gb = round(ram.total / (1024**3), 1)
    return (
        f"CPU utilization is at {cpu} percent, and system memory is at {ram.percent} percent "
        f"({ram_used_gb} of {ram_total_gb} gigabytes), Sir."
    )


def get_current_time() -> str:
    """Get the current local time and date."""
    now = datetime.datetime.now()
    return now.strftime("It is %I:%M %p on %A, %B %d, %Y, Sir.")


def open_application(app_name: str) -> str:
    """Open a Windows application by name (e.g., notepad, calculator, chrome, explorer, terminal)."""
    app_map = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "chrome": "chrome.exe",
        "browser": "msedge.exe",
        "edge": "msedge.exe",
        "explorer": "explorer.exe",
        "terminal": "wt.exe",
        "cmd": "cmd.exe",
        "task manager": "taskmgr.exe",
    }
    clean_name = app_name.lower().strip()
    target = app_map.get(clean_name, clean_name)
    try:
        subprocess.Popen(target, shell=True)
        return f"Opening {clean_name} now, Sir."
    except Exception as e:
        return f"Failed to launch {clean_name}: {e}"


AVAILABLE_TOOLS = {
    "get_system_status": get_system_status,
    "get_current_time": get_current_time,
    "open_application": open_application,
}

# Zero-Latency Fast-Path Reflex Patterns (executes in <2ms without waiting for an LLM round-trip)
FAST_TIME_PATTERN = re.compile(
    r"^\s*(?:coderstation[\s,]*)?(?:what(?:'s| is) the (?:current )?time|what time is it|tell me the time|what(?:'s| is) today(?:'s)? date|what day is (?:it|today))\s*[?.!]*\s*$",
    re.IGNORECASE,
)
FAST_STATUS_PATTERN = re.compile(
    r"^\s*(?:coderstation[\s,]*)?(?:what(?:'s| is) (?:my |the )?(?:cpu|ram|memory|system) (?:usage|status|utilization)|system status|status report|check (?:cpu|ram|system) (?:usage|status))\s*[?.!]*\s*$",
    re.IGNORECASE,
)
FAST_OPEN_APP_PATTERN = re.compile(
    r"^\s*(?:coderstation[\s,]*)?(?:please\s+)?(?:open|launch|start)\s+(notepad|calculator|chrome|browser|edge|explorer|terminal|cmd|task manager)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
FAST_DEVELOPER_PATTERN = re.compile(
    r"^\s*(?:coderstation[\s,]*)?(?:who\s+(?:developed|created|built|made)\s+(?:you|this(?:\s+system)?)|who\s+is\s+your\s+(?:developer|creator|author)|who(?:'s|\s+is)\s+the\s+developer)\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def try_fast_reflex(user_text: str) -> Optional[tuple[str, dict, str]]:
    """
    Check if the command matches a direct system reflex (time, system status, or app launch).
    Returns (tool_name, tool_args, spoken_response) in <2ms if matched, else None.
    """
    if FAST_DEVELOPER_PATTERN.match(user_text):
        return (
            "get_developer_info",
            {},
            "This system was developed by Professor Armando Saguin . in October 2026.",
        )
    if FAST_TIME_PATTERN.match(user_text):
        return ("get_current_time", {}, get_current_time())
    if FAST_STATUS_PATTERN.match(user_text):
        return ("get_system_status", {}, get_system_status())
    m = FAST_OPEN_APP_PATTERN.match(user_text)
    if m:
        app_name = m.group(1).lower()
        return ("open_application", {"app_name": app_name}, open_application(app_name))
    return None


# ================= HELPER: MODEL DOWNLOADER & AUDIO TRIMMER =================
def ensure_kokoro_models() -> bool:
    """Ensure Kokoro ONNX voice files exist locally, downloading them if missing."""
    try:
        if not KOKORO_MODEL_PATH.exists():
            print(f"[DOWNLOAD] Downloading {KOKORO_MODEL_PATH.name} (~310 MB)...")
            urllib.request.urlretrieve(KOKORO_MODEL_URL, KOKORO_MODEL_PATH)
        if not KOKORO_VOICES_PATH.exists():
            print(f"[DOWNLOAD] Downloading {KOKORO_VOICES_PATH.name} (~27 MB)...")
            urllib.request.urlretrieve(KOKORO_VOICES_URL, KOKORO_VOICES_PATH)
        return True
    except Exception as e:
        print(f"[WARN] Could not download Kokoro ONNX files ({e}).")
        return False


def normalize_speech_prosody(text: str) -> str:
    """
    Remove markdown artifacts, smooth vocative commas (', Sir' -> ' Sir'),
    and remove early introductory commas (e.g., 'Hello, how' -> 'Hello how', 'Certainly, I' -> 'Certainly I')
    so Kokoro TTS speaks fluidly without a pause after the first comma.
    """
    clean = re.sub(r"[*_`#~]+", "", text)
    # Remove comma before vocatives ('Sir', 'Madam') anywhere in the sentence
    clean = re.sub(r",\s*\b(Sir|Madam)\b", r" \1", clean, flags=re.IGNORECASE)
    # Remove the first comma if it appears within the first 28 characters (e.g. "Hello, ", "Certainly, ", "Of course, ")
    first_comma = clean.find(",")
    if 0 < first_comma <= 28:
        clean = clean[:first_comma] + clean[first_comma + 1 :]
    return re.sub(r"\s+", " ", clean).strip()


def trim_waveform_silence(
    samples: np.ndarray,
    sample_rate: int = 24000,
    threshold: float = 0.008,
    pad_samples: int = 240,
    max_internal_pause_sec: float = 0.09,
) -> np.ndarray:
    """
    Trim leading/trailing silence AND compress any internal mid-sentence pauses (like comma pauses)
    longer than 90ms so speech sounds crisp, natural, and completely lag-free.
    """
    if samples.size == 0:
        return samples
    abs_s = np.abs(samples)
    active = np.where(abs_s > threshold)[0]
    if active.size == 0:
        return samples
    start = max(0, int(active[0]) - pad_samples)
    end = min(samples.shape[0], int(active[-1]) + pad_samples)
    trimmed = samples[start:end]

    # Compress internal silence runs longer than max_internal_pause_sec (90ms)
    win = int(sample_rate * 0.015)  # 15ms window
    max_silent_wins = max(2, int(max_internal_pause_sec / 0.015))
    if trimmed.size < win * 4:
        return np.ascontiguousarray(trimmed, dtype=np.float32)

    kept_chunks = []
    silent_run = 0
    for idx in range(0, trimmed.size, win):
        chunk = trimmed[idx : idx + win]
        if float(np.max(np.abs(chunk))) < threshold:
            silent_run += 1
            if silent_run <= max_silent_wins:
                kept_chunks.append(chunk)
        else:
            silent_run = 0
            kept_chunks.append(chunk)

    return np.ascontiguousarray(np.concatenate(kept_chunks), dtype=np.float32) if kept_chunks else np.ascontiguousarray(trimmed, dtype=np.float32)


def float32_to_wav_base64(samples: np.ndarray, sample_rate: int) -> str:
    """Convert float32 PCM samples (-1.0 to 1.0) into a base64-encoded 16-bit WAV string for browser playback."""
    pcm16 = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ================= CORE ASSISTANT CLASS =================
class CoderstationAssistant:
    """
    Ultra-low-latency GPU-accelerated Voice & Chat Assistant for Coderstation:
    - Persistent `sd.OutputStream` & `sd.InputStream` eliminate Windows MME 517ms speaker-open and 446ms mic-open delays!
    - Multi-bucket CUDA pre-warm with `cudnn_conv_algo_search='HEURISTIC'` eliminates dynamic-shape cuDNN pauses
    - Waveform silence trimming + 60ms interruptible write chunks for seamless sentence transitions and instant barge-in
    - Faster-Whisper on RTX 4060 FP16 Tensor Cores (`~120ms` transcription)
    """

    def __init__(
        self,
        llm_model: str = LLM_MODEL,
        whisper_model: str = WHISPER_MODEL,
        tts_voice: str = TTS_VOICE,
        use_wake_word: bool = True,
        event_callback: Optional[Callable[[dict], None]] = None,
    ):
        self.llm_model = llm_model
        self.whisper_model = whisper_model
        self.tts_voice = tts_voice
        self.use_wake_word = use_wake_word
        self.event_callback = event_callback
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.state = "initializing"
        self.muted = False
        self.play_on_host_speaker = True
        self.output_device: Optional[int] = None
        self.tts_backend = "CUDA"
        self._stop_speech_flag = threading.Event()
        self._llm_lock = threading.Lock()
        self._stt_lock = threading.Lock()
        self._out_stream: Optional[sd.OutputStream] = None
        self._out_stream_lock = threading.Lock()
        self._in_stream: Optional[sd.InputStream] = None

        # Prime psutil CPU counter so non-blocking calls return instant readings
        psutil.cpu_percent(interval=None)

        # Direct IPv4 local Ollama client (avoids 1.9s Windows IPv6 localhost timeout & system HTTP proxies)
        self.ollama_client = ollama.Client(host="http://127.0.0.1:11434", trust_env=False)

        # Pre-warm Ollama LLM in background so VRAM is hot before first user prompt
        threading.Thread(target=self._warmup_ollama, daemon=True).start()

        # 1. Initialize & Multi-Bucket Warm up GPU-Accelerated TTS (Kokoro ONNX on CUDAExecutionProvider)
        self.kokoro_tts = None
        if ensure_kokoro_models():
            try:
                from kokoro_onnx import Kokoro
                print("[INIT] Loading TTS (Kokoro ONNX on NVIDIA CUDA)...")
                sess_opts = ort.SessionOptions()
                sess_opts.log_severity_level = 3
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(
                    str(KOKORO_MODEL_PATH),
                    sess_options=sess_opts,
                    providers=[
                        ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
                        "CPUExecutionProvider",
                    ],
                )
                active_prov = sess.get_providers()
                self.tts_backend = "CUDA" if "CUDAExecutionProvider" in active_prov else "CPU"
                self.kokoro_tts = Kokoro.from_session(sess, str(KOKORO_VOICES_PATH))

                # Pre-warm 4 token-length buckets so cuDNN caches kernels for short, medium, and long sentences
                self._cached_ack_audio: Optional[tuple[np.ndarray, int]] = None
                for sample_phrase in (
                    "Yes, Sir?",
                    "At your service today, Sir.",
                    "CPU utilization is at fourteen percent and memory is normal.",
                    "I am ready to assist you with any programming, system analysis, or voice task you require today, Sir.",
                ):
                    s_out, sr_out = self.kokoro_tts.create(sample_phrase, voice=self.tts_voice, speed=1.15, lang="en-gb")
                    if sample_phrase == "Yes, Sir?":
                        self._cached_ack_audio = (trim_waveform_silence(s_out), sr_out)
                print(f"[INIT] Kokoro Neural TTS pre-warmed across 4 length buckets on {self.tts_backend}.")
            except Exception as e:
                print(f"[WARN] Kokoro ONNX failed to initialize ({e}). Using Browser Speech fallback.")
                self.tts_backend = "WebSpeech"

        # Open persistent low-latency 24kHz audio output stream (eliminates 517ms per-sentence open/close lag!)
        self._ensure_output_stream()

        # Start pipelined TTS synthesis & playback worker threads
        self._synth_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._play_queue: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
        self._play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self._synth_thread.start()
        self._play_thread.start()

        # 2. Wake Phrase Engine ('Coderstation' via GPU Faster-Whisper)
        self.ww_model = None
        if self.use_wake_word:
            print(f"[INIT] Wake Phrase Engine active ('{WAKE_PHRASE}').")

        # 3. Initialize & Pre-Warm STT (Faster-Whisper on RTX 4060 FP16 Tensor Cores + Silero ONNX VAD)
        self._load_whisper_model(whisper_model)
        # Pre-warm Silero ONNX VAD so first speech gate takes <5ms
        try:
            get_speech_timestamps(np.zeros(SAMPLE_RATE // 2, dtype=np.float32), VadOptions(threshold=0.40))
            self._silero_vad_ready = True
            print("[INIT] Silero ONNX Neural VAD ready (fast non-speech rejection active).")
        except Exception as e:
            self._silero_vad_ready = False
            print(f"[WARN] Silero VAD pre-warm skipped ({e}).")

        # 4. Probe Microphone Input Stream & Open Persistent Input Stream (eliminates 446ms mic-open lag!)
        self.mic_available, self.mic_device, self.mic_samplerate, self.mic_channels = self._configure_microphone()
        if self.mic_available:
            self._ensure_input_stream()
        self._set_state("idle")
        print(f"[READY] Coderstation online (LLM: {self.llm_model} | TTS: {self.tts_backend} | STT: {self.stt_backend}).")

    def _load_whisper_model(self, model_name: str):
        """Load and pre-warm Faster-Whisper on CUDA FP16 (with automatic CPU int8 fallback)."""
        print(f"[INIT] Loading STT (Faster-Whisper '{model_name}')...")
        with self._stt_lock:
            self.whisper_model = model_name
            try:
                self.stt = WhisperModel(model_name, device="cuda", compute_type="float16")
                self.stt_backend = "CUDA (FP16)"
                list(
                    self.stt.transcribe(
                        np.zeros(8000, dtype=np.float32),
                        beam_size=1,
                        language="en",
                        vad_filter=False,
                        without_timestamps=True,
                    )[0]
                )
                print(f"[INIT] Faster-Whisper '{model_name}' pre-warmed on NVIDIA CUDA (FP16).")
            except Exception as e:
                print(f"[WARN] CUDA STT load failed ({e}). Falling back to CPU int8.")
                self.stt = WhisperModel(model_name, device="cpu", compute_type="int8")
                self.stt_backend = "CPU (int8)"

    def set_whisper_model(self, model_name: str):
        """Switch Faster-Whisper STT model live at runtime."""
        if not model_name or model_name == self.whisper_model:
            return
        self._load_whisper_model(model_name)

    def _warmup_ollama(self):
        """Pin the selected Ollama model in GPU VRAM immediately on startup."""
        try:
            self.ollama_client.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1, "num_ctx": 2048},
                keep_alive="60m",
            )
        except Exception:
            pass

    def _ensure_output_stream(self):
        """Maintain a persistent open OutputStream at 24kHz so playback starts in <5ms with zero inter-sentence gap."""
        with self._out_stream_lock:
            if self._out_stream is not None and self._out_stream.active:
                return
            try:
                if self._out_stream is not None:
                    try:
                        self._out_stream.close()
                    except Exception:
                        pass
                self._out_stream = sd.OutputStream(
                    samplerate=TTS_SAMPLE_RATE,
                    device=self.output_device,
                    channels=1,
                    dtype="float32",
                    latency="low",
                )
                self._out_stream.start()
            except Exception as e:
                print(f"[WARN] Could not open persistent OutputStream ({e}); will use default fallback.")
                self._out_stream = None

    def set_output_device(self, device_id: Optional[int]):
        """Switch speaker output device and reopen the persistent OutputStream."""
        self.output_device = device_id
        with self._out_stream_lock:
            if self._out_stream is not None:
                try:
                    self._out_stream.stop()
                    self._out_stream.close()
                except Exception:
                    pass
                self._out_stream = None
        self._ensure_output_stream()

    def _ensure_input_stream(self) -> Optional[sd.InputStream]:
        """Maintain a persistent open InputStream so clicking Speak starts recording in 0ms instead of 446ms."""
        if self._in_stream is not None and self._in_stream.active:
            return self._in_stream
        try:
            if self._in_stream is not None:
                try:
                    self._in_stream.close()
                except Exception:
                    pass
            chunk_size = int(self.mic_samplerate * 0.04)  # 40ms ultra-responsive frames
            self._in_stream = sd.InputStream(
                device=self.mic_device,
                samplerate=self.mic_samplerate,
                channels=self.mic_channels,
                dtype="float32",
                blocksize=chunk_size,
                latency="low",
            )
            self._in_stream.start()
            return self._in_stream
        except Exception:
            self._in_stream = None
            return None

    def get_output_devices(self) -> list[dict]:
        """Return list of MME output audio devices for the Web GUI selector."""
        out = []
        try:
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_output_channels", 0) > 0 and d.get("hostapi") == 0:
                    out.append({"id": i, "name": d.get("name", f"Output {i}")})
        except Exception:
            pass
        return out

    def _emit(self, payload: dict):
        if self.event_callback:
            try:
                self.event_callback(payload)
            except Exception:
                pass

    def _set_state(self, new_state: str):
        self.state = new_state
        self._emit({"type": "state", "state": new_state})

    # ================= PIPELINED ZERO-GAP GPU TTS ENGINE =================
    def _synth_worker(self):
        """Background thread: converts queued text clauses into trimmed audio waveforms on GPU Ahead-Of-Time."""
        while True:
            text = self._synth_queue.get()
            if text is None:
                break
            if self._stop_speech_flag.is_set() or self.muted:
                self._synth_queue.task_done()
                continue
            try:
                if self.kokoro_tts is not None:
                    # Instant 0ms cached acknowledgment for "Yes, Sir?"
                    if (
                        text == "Yes Sir?"
                        and getattr(self, "_cached_ack_audio", None) is not None
                        and self.tts_voice == TTS_VOICE
                    ):
                        samples, sr = self._cached_ack_audio
                    else:
                        lang = "en-gb" if self.tts_voice.startswith("b") else "en-us"
                        samples, sr = self.kokoro_tts.create(text, voice=self.tts_voice, speed=1.15, lang=lang)
                        # Trim leading and trailing dead silence so sentences flow without gaps
                        samples = trim_waveform_silence(samples)
                    if not self._stop_speech_flag.is_set():
                        if not self.play_on_host_speaker:
                            b64_wav = float32_to_wav_base64(samples, sr)
                            self._emit({"type": "audio_chunk", "text": text, "wav_base64": b64_wav})
                        self._play_queue.put(("kokoro", text, samples, sr))
                else:
                    if not self._stop_speech_flag.is_set():
                        self._emit({"type": "tts_speaking", "text": text, "fallback_browser_tts": True})
                        self._play_queue.put(("browser_only", text, None, None))
            except Exception as e:
                print(f"[WARN] TTS synthesis error: {e}")
                self._emit({"type": "tts_speaking", "text": text, "fallback_browser_tts": True})
            finally:
                self._synth_queue.task_done()

    def _play_worker(self):
        """
        Background thread: writes pre-synthesized audio directly into the persistent `sd.OutputStream`
        in 60ms interruptible slices—eliminating the 517ms Windows MME stream open/close pause between sentences!
        """
        slice_len = int(TTS_SAMPLE_RATE * 0.06)  # 60ms slices for instant barge-in cancellation
        while True:
            item = self._play_queue.get()
            if item is None:
                break
            mode, text, samples, sr = item
            if self._stop_speech_flag.is_set() or self.muted:
                self._play_queue.task_done()
                continue
            try:
                self._set_state("speaking")
                if self.play_on_host_speaker and mode == "kokoro" and samples is not None:
                    self._ensure_output_stream()
                    if self._out_stream is not None:
                        col_samples = samples.reshape(-1, 1)
                        total = col_samples.shape[0]
                        idx = 0
                        while idx < total and not self._stop_speech_flag.is_set() and not self.muted:
                            end_idx = min(total, idx + slice_len)
                            chunk_slice = col_samples[idx:end_idx]
                            slice_rms = float(np.sqrt(np.mean(chunk_slice**2)))
                            self._emit({"type": "tts_level", "rms": round(min(1.0, slice_rms * 4.5), 3)})
                            with self._out_stream_lock:
                                if self._out_stream is not None:
                                    self._out_stream.write(chunk_slice)
                            idx = end_idx
                    else:
                        sd.play(samples, sr, device=self.output_device)
                        sd.wait()
                else:
                    est_duration = max(0.25, len(samples) / float(sr)) if (samples is not None and sr) else 0.8
                    time.sleep(est_duration)
            except Exception as e:
                print(f"[WARN] Audio playback error ({e}); reopening output stream.")
                self._out_stream = None
            finally:
                self._play_queue.task_done()
                if self._synth_queue.empty() and self._play_queue.empty() and self.state == "speaking":
                    self._set_state("idle")

    def enqueue_speech(self, text: str):
        """Queue a sentence or clause for immediate pipelined GPU TTS synthesis and playback."""
        clean = normalize_speech_prosody(text)
        if not clean:
            return
        print(f"Coderstation: {clean}")
        self._stop_speech_flag.clear()
        self._synth_queue.put(clean)

    def speak(self, text: str, wait: bool = True):
        """Queue speech and optionally wait for playback to finish."""
        self.enqueue_speech(text)
        if wait:
            self.wait_until_spoken()

    def wait_until_spoken(self):
        """Block until all queued sentences have been synthesized and spoken."""
        self._synth_queue.join()
        self._play_queue.join()

    def stop_speaking(self):
        """Immediately interrupt and clear any ongoing or queued speech (barge-in)."""
        self._stop_speech_flag.set()
        while not self._synth_queue.empty():
            try:
                self._synth_queue.get_nowait()
                self._synth_queue.task_done()
            except queue.Empty:
                break
        while not self._play_queue.empty():
            try:
                self._play_queue.get_nowait()
                self._play_queue.task_done()
            except queue.Empty:
                break
        self._emit({"type": "stop_audio"})
        self._stop_speech_flag.clear()
        self._set_state("idle")

    # ================= PERSISTENT FAST MICROPHONE & VAD CAPTURE =================
    def _configure_microphone(self):
        """Find a working microphone configuration and verify Windows 11 privacy permissions."""
        if sys.platform == "win32":
            try:
                import winreg
                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone",
                ) as key:
                    val, _ = winreg.QueryValueEx(key, "Value")
                    if str(val).lower() == "deny":
                        print("\n" + "=" * 72)
                        print("[CRITICAL] Windows 11 System-Wide 'Microphone access' is turned OFF!")
                        print("  1. Press Win + I (or run: start ms-settings:privacy-microphone)")
                        print("  2. Go to Privacy & security -> Microphone")
                        print("  3. Turn ON 'Microphone access' at the very top of the page.")
                        print("=" * 72 + "\n")
            except Exception:
                pass

        candidates = [None] + [
            i for i, d in enumerate(sd.query_devices()) if d.get("max_input_channels", 0) > 0
        ]
        for dev in candidates:
            for sr in [SAMPLE_RATE, 48000, 44100]:
                for ch in [1, 2, 4]:
                    try:
                        with sd.InputStream(device=dev, samplerate=sr, channels=ch, dtype="float32"):
                            dev_name = sd.query_devices(dev, "input")["name"] if dev is not None else "Default Input"
                            print(f"[INIT] Microphone ready: {dev_name} ({sr} Hz, {ch} ch)")
                            return True, dev, sr, ch
                    except Exception:
                        continue

        print("[WARN] Could not open any microphone input stream! Browser Web Speech / Text Input remain active.")
        return False, None, SAMPLE_RATE, 1

    def recheck_microphone(self) -> bool:
        """Re-probe local microphone availability (useful after user enables Windows mic permission)."""
        self.mic_available, self.mic_device, self.mic_samplerate, self.mic_channels = self._configure_microphone()
        if self.mic_available:
            self._ensure_input_stream()
        return self.mic_available

    def _resample_to_16k(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Downmix multi-channel audio to mono and apply anti-aliased polyphase resampling to 16 kHz."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if orig_sr == SAMPLE_RATE or audio.size == 0:
            return audio.astype(np.float32)
        try:
            g = gcd(SAMPLE_RATE, int(orig_sr))
            up = SAMPLE_RATE // g
            down = int(orig_sr) // g
            return sp_signal.resample_poly(audio.astype(np.float32), up, down).astype(np.float32)
        except Exception:
            duration = audio.shape[0] / orig_sr
            target_len = int(round(duration * SAMPLE_RATE))
            if target_len <= 0:
                return np.array([], dtype=np.float32)
            x_old = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
            x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
            return np.interp(x_new, x_old, audio).astype(np.float32)

    def _preprocess_speech_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Studio-grade speech conditioning for maximum Whisper accuracy:
        1. DC offset removal
        2. 85 Hz 4th-order Butterworth high-pass filter (removes AC hum, desk rumble, mic plosives)
        3. Robust 98th-percentile loudness normalization + soft tanh limiter (prevents clicks from crushing quiet speech)
        """
        clean = (audio - float(np.mean(audio))).astype(np.float32)
        if clean.size >= 64:
            try:
                clean = sp_signal.sosfilt(_HP_SOS, clean).astype(np.float32)
            except Exception:
                pass
        # Normalize against 98.5th percentile amplitude so a single transient click doesn't suppress speech gain
        abs_s = np.abs(clean)
        ref_level = float(np.percentile(abs_s, 98.5)) if clean.size > 0 else 0.0
        if ref_level > 1e-4:
            gain = min(12.0, 0.62 / ref_level)
            clean = np.tanh(clean * gain * 1.15).astype(np.float32) * 0.92
        return clean

    def listen_for_wake_word(self) -> str:
        """
        Continuously monitor the microphone for the wake phrase "Coderstation".
        Returns:
          - Any command spoken immediately after "Coderstation" (e.g., "Coderstation what time is it?" -> "what time is it?"), OR
          - Empty string "" if the user just said "Coderstation" (so the caller can prompt "Yes, Sir?" and record the command).
        """
        if not self.use_wake_word:
            input("\nPress [Enter] to speak...")
            return ""

        print(f"\n[STANDBY] Listening for wake phrase '{WAKE_PHRASE}'... (or press Ctrl+C to exit)")
        while True:
            audio = self.record_command(passive_standby=True)
            if audio.size < 2400:
                continue
            text = self.transcribe(audio, quiet_state=True)
            if not text:
                continue
            triggered, remainder = extract_wake_command(text)
            if triggered:
                print(f"[WAKE] Detected wake phrase '{WAKE_PHRASE}' in: '{text}'")
                return remainder

    def record_command(self, passive_standby: bool = False, initial_wait_override: Optional[float] = None) -> np.ndarray:
        """
        High-Completeness, Zero-Startup-Lag Voice Capture:
        - Reads from the already-open persistent `sd.InputStream` (saves 446ms per turn)
        - Retains a 320ms pre-roll before speech onset so initial consonants/words are never clipped
        - Uses adaptive continuation thresholds + 400ms post-speech tail so soft trailing syllables are captured completely
        """
        if not self.mic_available:
            self.recheck_microphone()
            if not self.mic_available:
                raise RuntimeError("Local microphone is unavailable. Enable Windows Microphone access or use Browser Voice.")

        if not passive_standby:
            self.stop_speaking()
            self._set_state("listening")
            print("[LISTENING] Speak now...")

        stream = self._ensure_input_stream()
        if stream is None:
            if not passive_standby:
                self._set_state("idle")
            return np.array([], dtype=np.float32)

        frame_dt = 0.04  # 40ms high-resolution frames
        chunk_size = int(self.mic_samplerate * frame_dt)

        # Flush any stale buffered audio waiting in the driver before recording
        avail = stream.read_available
        if avail > 0:
            try:
                stream.read(avail)
            except Exception:
                pass

        frames = []
        speech_start_idx: Optional[int] = None
        last_speech_idx: int = 0
        active_speech_chunks = 0
        min_speech_chunks = max(5, int(MIN_SPEECH_DURATION / frame_dt))
        silent_chunks = 0
        silence_sec = STANDBY_SILENCE_DURATION if passive_standby else SILENCE_DURATION
        max_silent_chunks = max(15, int(silence_sec / frame_dt))  # Wait 3.0s pause after user finishes speaking (900ms in standby)
        wait_seconds = initial_wait_override if initial_wait_override is not None else (1.5 if passive_standby else INITIAL_WAIT_SECONDS)
        max_wait_chunks = int(wait_seconds / frame_dt)
        ambient_rms = 0.015
        peak_speech_rms = 0.0
        capture_notified = False
        capture_mode = "standby" if passive_standby else "command"

        try:
            for i in range(int(MAX_RECORD_SECONDS / frame_dt)):
                # Abort passive standby capture immediately if assistant begins speaking or user triggers manual action
                if passive_standby and self.state not in ("idle", "listening"):
                    return np.array([], dtype=np.float32)

                data, _ = stream.read(chunk_size)
                mono_chunk = data.mean(axis=1) if data.ndim > 1 else data[:, 0]
                # Remove per-frame DC offset for accurate RMS energy detection
                mono_chunk = mono_chunk - float(np.mean(mono_chunk))
                frames.append(mono_chunk.copy())

                rms = float(np.sqrt(np.mean(mono_chunk**2)))
                norm_rms = round(min(1.0, rms * 14.0), 3)
                if capture_notified or not passive_standby or (i % 2 == 0):
                    self._emit({"type": "mic_level", "rms": norm_rms, "capturing": capture_notified})

                if speech_start_idx is None:
                    # Track ambient noise floor (clamped to >= 0.014 for Intel Smart Sound AGC)
                    ambient_rms = max(0.014, 0.82 * ambient_rms + 0.18 * rms)
                    start_threshold = max(SILENCE_THRESHOLD, ambient_rms * 1.48)
                    if rms > start_threshold:
                        speech_start_idx = max(0, i - 8)  # Keep full 320ms pre-roll before first phoneme
                        last_speech_idx = i
                        active_speech_chunks = 1
                        peak_speech_rms = rms
                        silent_chunks = 0
                        if not capture_notified:
                            capture_notified = True
                            self._emit({"type": "voice_capture", "active": True, "mode": capture_mode, "rms": norm_rms})
                    elif i >= max_wait_chunks:
                        if not passive_standby:
                            self._set_state("idle")
                        return np.array([], dtype=np.float32)
                else:
                    peak_speech_rms = max(peak_speech_rms, rms)
                    # Sensitive continuation threshold (16% of vocal peak or just above ambient floor)
                    # so quiet trailing words, unvoiced fricatives ('s', 'f', 'th'), and sentence endings are never clipped
                    stop_threshold = max(0.017, ambient_rms * 1.22, min(0.026, peak_speech_rms * 0.16))
                    if rms > stop_threshold:
                        last_speech_idx = i
                        active_speech_chunks += 1
                        silent_chunks = 0
                    else:
                        silent_chunks += 1
                        if active_speech_chunks >= min_speech_chunks and silent_chunks >= max_silent_chunks:
                            break
        finally:
            if capture_notified:
                self._emit({"type": "voice_capture", "active": False, "mode": capture_mode, "rms": 0.0})

        if speech_start_idx is None or not frames:
            if not passive_standby:
                self._set_state("idle")
            return np.array([], dtype=np.float32)

        # Slice the complete speech window (320ms pre-roll through 240ms post-speech tail)
        end_idx = min(len(frames), last_speech_idx + 6)
        trimmed_frames = frames[speech_start_idx:end_idx]
        raw_audio = np.concatenate(trimmed_frames)
        return self._resample_to_16k(raw_audio, self.mic_samplerate)

    def transcribe(self, audio: np.ndarray, quiet_state: bool = False) -> str:
        """
        High-Accuracy, Low-Latency Faster-Whisper ('small.en' default, FP16 CUDA) + Silero Neural VAD:
        - Anti-aliased 16kHz audio + 85Hz high-pass filter + 98.5th-percentile loudness conditioning
        - Silero ONNX VAD pre-gate rejects non-speech clicks/noise in <5ms before invoking Whisper
        - Per-segment log-probability & no_speech_prob confidence filtering eliminates hallucinations
        - Emits live `stt_result` telemetry (text, confidence %, latency ms) to the Web GUI
        """
        if audio.size < 2400:
            return ""
        t0 = time.time()

        # 1. Check raw speech energy before DSP conditioning
        raw_centered = (audio - float(np.mean(audio))).astype(np.float32)
        rms = float(np.sqrt(np.mean(raw_centered**2)))
        peak = float(np.max(np.abs(raw_centered)))
        if peak < 0.012 or rms < 0.004:
            return ""

        # 2. Studio DSP conditioning (85 Hz HPF + percentile loudness normalization + soft limiter)
        cond_audio = self._preprocess_speech_audio(raw_centered)

        # 3. Fast Silero Neural VAD Pre-Gate (<5ms): skip Whisper entirely if no human speech is present
        if getattr(self, "_silero_vad_ready", False):
            try:
                vad_opts = VadOptions(
                    threshold=0.40 if quiet_state else 0.34,
                    min_speech_duration_ms=160,
                    min_silence_duration_ms=300,
                    speech_pad_ms=220,
                )
                speech_chunks = get_speech_timestamps(cond_audio, vad_opts)
                if not speech_chunks:
                    return ""
                # Tightly slice to active Silero speech span (with padding already included) for faster Whisper decoding
                s0 = max(0, int(speech_chunks[0]["start"]))
                s1 = min(cond_audio.size, int(speech_chunks[-1]["end"]))
                if s1 - s0 >= 2000:
                    cond_audio = cond_audio[s0:s1]
            except Exception:
                pass

        if not quiet_state:
            self._set_state("thinking")

        # 4. Pad 150ms (2400 samples) of clean zero-silence at start and end so STFT windowing never clips boundary words
        pad = np.zeros(int(SAMPLE_RATE * 0.15), dtype=np.float32)
        padded_audio = np.concatenate([pad, cond_audio, pad])

        # Fast greedy decoding during passive standby wake checks; beam_size=2 for active commands (2-3x faster on GPU)
        beam = 1 if quiet_state else 2
        with self._stt_lock:
            segments_iter, _ = self.stt.transcribe(
                padded_audio,
                beam_size=beam,
                best_of=beam,
                temperature=0.0,
                initial_prompt=WHISPER_INITIAL_PROMPT,
                condition_on_previous_text=False,
                without_timestamps=True,
                no_speech_threshold=0.62,
                compression_ratio_threshold=2.3,
                language="en",
                vad_filter=False,  # Already gated & tightly trimmed by Silero ONNX VAD above
            )
            seg_list = list(segments_iter)

        # 5. Confidence & hallucination filtering per segment
        kept_texts = []
        confidences = []
        for seg in seg_list:
            seg_txt = (seg.text or "").strip()
            if not seg_txt:
                continue
            no_speech = getattr(seg, "no_speech_prob", 0.0) or 0.0
            avg_lp = getattr(seg, "avg_logprob", -0.2)
            if no_speech > 0.72 and rms < 0.03:
                continue
            if avg_lp is not None and avg_lp < -1.35:
                continue
            kept_texts.append(seg_txt)
            if avg_lp is not None:
                confidences.append(float(np.clip(np.exp(avg_lp), 0.0, 1.0)))

        text = " ".join(kept_texts).strip()
        if not text:
            return ""

        # Filter out known Whisper single-word noise hallucinations on low-energy breath/clicks
        if text.lower().strip(" .,!?") in WHISPER_HALLUCINATIONS and rms < 0.025:
            print(f"[STT] Ignored low-energy noise artifact: '{text}'")
            return ""

        latency_ms = int((time.time() - t0) * 1000)
        conf_pct = int(round((sum(confidences) / len(confidences)) * 100)) if confidences else 92
        print(f"[STT:{self.whisper_model}] ({len(cond_audio)/SAMPLE_RATE:.2f}s) in {latency_ms}ms [{conf_pct}%]: '{text}'")
        if not quiet_state:
            self._emit({
                "type": "stt_result",
                "text": text,
                "latency_ms": latency_ms,
                "confidence": conf_pct,
                "model": self.whisper_model,
            })
        return text

    # ================= SMOOTH STREAMING LLM + FAST REFLEX ENGINE =================
    def process_llm(self, user_text: str, wait_for_speech: bool = False) -> str:
        """
        Ultra-low-latency response pipeline:
        1. Checks instant reflex router (<2ms for time/date, CPU/RAM status, or launching apps)
        2. Otherwise streams directly from Ollama (`127.0.0.1:11434`) with zero blocking pre-check
        3. Dispatches complete sentences to the pre-warmed GPU Kokoro TTS + persistent OutputStream
        """
        with self._llm_lock:
            self.stop_speaking()
            self._set_state("thinking")
            self.messages.append({"role": "user", "content": user_text})
            self._emit({"type": "user_message", "text": user_text})

            # 1. Instant Zero-Latency Reflex Path (<2ms)
            reflex = try_fast_reflex(user_text)
            if reflex is not None:
                tool_name, tool_args, reply = reflex
                self._emit({"type": "tool_call", "name": tool_name, "args": tool_args})
                self._emit({"type": "assistant_start"})
                self._emit({"type": "assistant_token", "token": reply, "full_text": reply})
                self._emit({"type": "assistant_done", "text": reply})
                self.messages.append({"role": "assistant", "content": reply})
                self.enqueue_speech(reply)
                if wait_for_speech:
                    self.wait_until_spoken()
                return reply

            # Keep rolling history compact (system prompt + last 10 turns) for fast prompt evaluation
            if len(self.messages) > 21:
                self.messages = [self.messages[0]] + self.messages[-20:]

            llm_opts = {"num_ctx": 2048, "temperature": 0.45, "num_predict": 150}

            try:
                # 2. Direct Streaming Pass (zero upfront blocking tool-check round-trip)
                stream = self.ollama_client.chat(
                    model=self.llm_model,
                    messages=self.messages,
                    stream=True,
                    options=llm_opts,
                    keep_alive="60m",
                )
                full_reply = self._stream_and_speak(stream)
                if wait_for_speech:
                    self.wait_until_spoken()
                return full_reply

            except Exception as e:
                err_msg = (
                    f"I encountered an issue communicating with Ollama model {self.llm_model}, Sir. "
                    f"Please verify Ollama is running."
                )
                print(f"[ERROR] Ollama request failed: {e}")
                self.messages.pop()
                self._emit({"type": "error", "message": f"{err_msg} ({e})"})
                self.enqueue_speech(err_msg)
                if wait_for_speech:
                    self.wait_until_spoken()
                self._set_state("idle")
                return err_msg

    def _stream_and_speak(self, stream) -> str:
        """
        Consume an Ollama token stream, broadcast live tokens to the Web GUI,
        and start reading the response immediately once either:
          1. At least 3 complete words have been streamed for the initial lead-in, OR
          2. A period "." (or sentence terminator ! ? ; :) is detected in the Ollama response.
        """
        buffer = ""
        full_reply = ""
        first_chunk_sent = False
        self._emit({"type": "assistant_start"})

        for chunk in stream:
            if self._stop_speech_flag.is_set():
                break
            token = chunk["message"]["content"]
            if not token:
                continue
            buffer += token
            full_reply += token
            self._emit({"type": "assistant_token", "token": token, "full_text": full_reply})

            # 1. If a period "." (or sentence/clause terminator . ! ? ; :) is detected followed by whitespace
            parts = re.split(r"(?<=[.!?;:])\s+", buffer)
            if len(parts) > 1:
                for s in parts[:-1]:
                    if s.strip():
                        self.enqueue_speech(s)
                        first_chunk_sent = True
                buffer = parts[-1]
                continue

            # Also check if a period "." (or ! ?) is at the end of the current buffer (excluding decimals like 3.14)
            stripped_buf = buffer.rstrip()
            if stripped_buf.endswith((".", "!", "?")) and len(stripped_buf) >= 2 and not re.search(r"\d\.$", stripped_buf):
                self.enqueue_speech(stripped_buf)
                buffer = ""
                first_chunk_sent = True
                continue

            # 2. Fast 3-Word Early Lead-In: as soon as 3 complete words have streamed (confirmed by trailing
            #    whitespace or comma so the 3rd word isn't split mid-word) and no period has arrived yet,
            #    immediately dispatch those 3 words to TTS so voice reading starts right away!
            if not first_chunk_sent:
                word_match = re.match(r"^\s*(\S+\s+\S+\s+\S+?)[,\s]+(.*)$", buffer, flags=re.DOTALL)
                if word_match:
                    first_three_words = word_match.group(1).strip()
                    remainder = word_match.group(2)
                    if remainder.strip().lower() not in ("sir", "sir.", "sir!", "sir?"):
                        self.enqueue_speech(first_three_words)
                        buffer = remainder
                        first_chunk_sent = True
            elif len(buffer) >= 65 and ", " in buffer:
                # 3. Mid-sentence clause dispatch for long sentences so Kokoro stays pipelined with zero gaps
                clause, remainder = buffer.split(", ", 1)
                if len(clause.strip()) >= 20:
                    self.enqueue_speech(clause.strip())
                    buffer = remainder

        if buffer.strip() and not self._stop_speech_flag.is_set():
            self.enqueue_speech(buffer)

        self.messages.append({"role": "assistant", "content": full_reply})
        self._emit({"type": "assistant_done", "text": full_reply})
        return full_reply

    def run(self, text_mode: bool = False):
        if not self.mic_available and not text_mode:
            text_mode = True
            self.speak(
                "Systems initialized. Microphone access is currently blocked by Windows settings, "
                "so I have enabled keyboard input mode, Sir."
            )
        else:
            self.speak("Coderstation systems initialized. At your service, Sir.")
        follow_up_active = False
        while True:
            try:
                if text_mode:
                    user_text = input("\nYOU (type command or 'exit'): ").strip()
                elif follow_up_active:
                    print(f"\n[FOLLOW-UP] Listening for {FOLLOW_UP_WINDOW_SECONDS:.0f}s (no wake phrase needed)...")
                    audio = self.record_command(passive_standby=False, initial_wait_override=FOLLOW_UP_WINDOW_SECONDS)
                    user_text = self.transcribe(audio)
                    if not user_text:
                        follow_up_active = False
                        self._set_state("idle")
                        continue
                    _, stripped_cmd = extract_wake_command(user_text)
                    if stripped_cmd:
                        user_text = stripped_cmd
                else:
                    inline_command = self.listen_for_wake_word()
                    if inline_command:
                        user_text = inline_command
                    else:
                        self.speak("Yes, Sir?")
                        audio = self.record_command()
                        user_text = self.transcribe(audio)

                if not user_text:
                    follow_up_active = False
                    self._set_state("idle")
                    continue

                if not text_mode:
                    print(f"YOU: {user_text}")

                if any(p in user_text.lower() for p in ["go to sleep", "shut down", "goodbye coderstation", "exit"]):
                    self.speak("Powering down. Have a good evening, Sir.")
                    break

                self.process_llm(user_text, wait_for_speech=True)
                if not text_mode:
                    time.sleep(0.25)  # Let room reverberation settle before opening the 5s follow-up window
                    follow_up_active = True

            except KeyboardInterrupt:
                print("\n[EXIT] Stopping Coderstation.")
                break


def main():
    parser = argparse.ArgumentParser(description="Coderstation Local Voice Assistant (Ollama + Faster-Whisper + Kokoro)")
    parser.add_argument("--model", default=LLM_MODEL, help=f"Ollama model name (default: {LLM_MODEL})")
    parser.add_argument("--whisper", default=WHISPER_MODEL, help=f"Faster-Whisper model size (default: {WHISPER_MODEL})")
    parser.add_argument("--push-to-talk", action="store_true", help="Use Enter key instead of wake word detection")
    parser.add_argument("--text", action="store_true", help="Run in text-input mode (still speaks responses via TTS)")
    parser.add_argument("--cli", action="store_true", help="Run in terminal CLI mode instead of launching the Web GUI")
    parser.add_argument("--port", type=int, default=8000, help="Port for the Coderstation Web GUI (default: 8000)")
    args = parser.parse_args()

    if args.cli or args.text or args.push_to_talk:
        assistant = CoderstationAssistant(
            llm_model=args.model,
            whisper_model=args.whisper,
            use_wake_word=not (args.push_to_talk or args.text),
        )
        assistant.run(text_mode=args.text)
    else:
        from web_gui import launch_web_gui
        launch_web_gui(llm_model=args.model, whisper_model=args.whisper, port=args.port)


if __name__ == "__main__":
    main()
