 
<h1 align="center">CODERSTATION-Local AI Voice Chat</h1>

<p align="center">
  A GPU-accelerated, 100% local AI voice assistant featuring a 60FPS 3D holographic pure-dot avatar, live system telemetry HUD, Faster-Whisper + Silero VAD speech recognition, Ollama LLM streaming, and Kokoro ONNX neural TTS.
</p>
 
---
<img width="824" height="431" alt="image" src="https://github.com/user-attachments/assets/1191e52d-bb08-400f-88c1-83cf0b8ac163" />

## Features

- **3D Holographic Pure-Dot Avatar**: Real-time 60FPS HTML5 Canvas 3D human-form dot constellation with glowing `#B9F2FD` Heart-to-Brain & Heart-to-Shoulder energy pulses, rotating Arc Reactor core, and live voice-capture wave indicator.
- **High-Accuracy Local Voice Recognition**: GPU-accelerated **Faster-Whisper** (`small.en` default, live-switchable) paired with **Silero ONNX VAD**, 85 Hz Butterworth high-pass filtering, and polyphase 16 kHz resampling.
- **Hands-Free & One-Click Voice Control**: Always-on `"Coderstation"` wake phrase with automatic 5-second follow-up conversation window, plus instant `Speak Now` capture and optional text input.
- **Low-Latency Neural Speech**: **Kokoro v1.0 ONNX** on CUDA with all **28 English (US) & British (UK) voices**, clause-level streaming TTS, and zero-gap persistent audio output.
- **Local LLM & Instant Reflexes**: Streams responses directly from local **Ollama** models (`llama3.1:8b`, `qwen3:8b`, etc.) with `<2ms` fast-path reflexes for time/date, CPU/RAM status, and launching Windows apps.
- **Transparent Telemetry HUD**: Live CPU, GPU, VRAM, RAM sparklines, active Ollama model monitor (`ollama ps`), and system clock integrated seamlessly into the avatar stage.

---

## Requirements

- **OS**: Windows 10/11
- **Python**: 3.10+
- **Dependencies**:kokoro-v1.0.onnx and voices-v1.0.bin placed in the project root
- **GPU**: NVIDIA CUDA GPU recommended (CPU fallback supported)
- **LLM Backend**: [Ollama](https://ollama.com/) running locally (`http://127.0.0.1:11434`) with a model pulled (e.g., `ollama pull llama3.1:8b`)

---

## Quick Start

1. **Clone & enter the project**:
   ```bash
   git clone <your-repo-url>
   cd "Coderstation -Assistant"
   ```
2. **Launch with the one-click Windows launcher** (automatically creates `.venv`, installs dependencies, and opens `http://127.0.0.1:8000`):
   ```bat
   run_coderstation.bat
   ```
   *(Or double-click `Coderstation.lnk`.)*
3. **Optional CLI / custom model launch**:
   ```bash
   .venv\Scripts\python.exe coderstation.py --model llama3.1:8b --whisper small.en --port 8000
   ```

---

## Project Structure

```text
Coderstation -Assistant/
├── coderstation.py        # Core STT (Faster-Whisper + Silero VAD), Ollama LLM & Kokoro TTS engine
├── web_gui.py             # FastAPI WebSocket server, telemetry API & static file host
├── static/
│   ├── index.html         # HUD layout & controls
│   ├── styles.css         # Borderless transparent sci-fi dashboard styling
│   └── app.js             # 60FPS 3D holographic avatar, Web Audio sync & live charts
├── run_coderstation.bat   # Windows auto-setup & launcher script
├── Coderstation.lnk       # Windows application shortcut with custom icon
├── coderstation.png       # Brand & web favicon image
├── coderstation.ico       # Windows application icon
└── requirements.txt       # Python dependencies
```

---

 
