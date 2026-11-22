import os
import json
import time
import asyncio
import subprocess
import threading
import webbrowser
from typing import Set

import psutil
import uvicorn
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
ICON_PNG_PATH = BASE_DIR / "coderstation.png"
ICON_ICO_PATH = BASE_DIR / "coderstation.ico"
STATIC_DIR = BASE_DIR / "static"

from coderstation import (
    CoderstationAssistant,
    LLM_MODEL,
    WHISPER_MODEL,
    SYSTEM_PROMPT,
    WAKE_PHRASE,
    FOLLOW_UP_WINDOW_SECONDS,
    extract_wake_command,
)

app = FastAPI(title="Coderstation AI Assistant Web GUI")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

assistant: CoderstationAssistant | None = None
connected_clients: Set[WebSocket] = set()
main_loop: asyncio.AbstractEventLoop | None = None
always_listen_enabled = True  # Enabled by default so saying "Coderstation" works without clicking any button
manual_listen_active = threading.Event()
follow_up_until: float = 0.0  # Timestamp until which follow-up voice input does NOT require the wake phrase


def broadcast_event(payload: dict):
    """Thread-safe broadcast from Coderstation worker threads to all connected WebSocket clients."""
    if main_loop is None or not connected_clients:
        return
    msg = json.dumps(payload)
    for ws in list(connected_clients):
        asyncio.run_coroutine_threadsafe(_safe_send(ws, msg), main_loop)


async def _safe_send(ws: WebSocket, msg: str):
    try:
        await ws.send_text(msg)
    except Exception:
        connected_clients.discard(ws)


def _background_wake_word_loop():
    """
    Always-on background listener for the wake phrase "Coderstation", plus automatic
    5-second follow-up conversation window after every AI response (no wake phrase needed).
    """
    global always_listen_enabled, follow_up_until
    while True:
        now = time.time()
        in_follow_up = now < follow_up_until

        if (not always_listen_enabled and not in_follow_up) or assistant is None or not assistant.mic_available or manual_listen_active.is_set():
            time.sleep(0.10)
            continue
        if assistant.state != "idle":
            time.sleep(0.05)
            continue

        try:
            # 1. Active 5-Second Follow-Up Window (no wake phrase required)
            if in_follow_up:
                remaining = max(0.5, follow_up_until - time.time())
                broadcast_event({"type": "status_note", "message": ""})
                audio = assistant.record_command(passive_standby=False, initial_wait_override=remaining)
                if manual_listen_active.is_set():
                    continue
                if audio.size >= 2400:
                    text = assistant.transcribe(audio)
                    if text and not manual_listen_active.is_set():
                        _, stripped_cmd = extract_wake_command(text)
                        final_text = stripped_cmd if stripped_cmd else text
                        assistant.process_llm(final_text, wait_for_speech=True)
                        time.sleep(0.10)  # Brief room-reverb settle before re-opening follow-up capture
                        follow_up_until = time.time() + FOLLOW_UP_WINDOW_SECONDS
                        continue

                # 5-second window expired with no additional voice input
                follow_up_until = 0.0
                assistant._set_state("idle")
                broadcast_event({"type": "status_note", "message": ""})
                continue

            # 2. Passive Standby Mode (requires wake phrase "Coderstation")
            audio = assistant.record_command(passive_standby=True)
            if manual_listen_active.is_set() or not always_listen_enabled:
                continue
            if audio.size < 2400:
                continue

            text = assistant.transcribe(audio, quiet_state=True)
            if not text or manual_listen_active.is_set():
                continue

            triggered, command_text = extract_wake_command(text)
            if not triggered:
                # Ignore background speech that does not include the wake phrase "Coderstation"
                continue

            broadcast_event({"type": "status_note", "message": f"Wake phrase '{WAKE_PHRASE}' detected!"})

            # If user only said "Coderstation", prompt and capture their command next
            if not command_text:
                assistant.speak("Yes, Sir?", wait=True)
                time.sleep(0.08)
                cmd_audio = assistant.record_command(passive_standby=False)
                command_text = assistant.transcribe(cmd_audio)

            if command_text:
                # Wait until Ollama provides and reads the response, then open the 5-second follow-up window
                assistant.process_llm(command_text, wait_for_speech=True)
                time.sleep(0.10)
                follow_up_until = time.time() + FOLLOW_UP_WINDOW_SECONDS
            else:
                assistant._set_state("idle")
                broadcast_event({"type": "status_note", "message": ""})
        except Exception as e:
            broadcast_event({"type": "error", "message": str(e)})
            time.sleep(1.0)


@app.on_event("startup")
async def on_startup():
    global main_loop
    main_loop = asyncio.get_running_loop()
    threading.Thread(target=_background_wake_word_loop, daemon=True).start()


# Cache Ollama model list so /api/telemetry is near-zero overhead (<1ms)
_cached_models: list[str] = []
_cached_models_ts: float = 0.0


def _get_gpu_status() -> dict:
    """Query NVIDIA GPU utilization, VRAM, and temperature via nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True,
        ).strip()
        if not out:
            return {"available": False}
        first_line = out.splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) >= 5:
            mem_used_mb = float(parts[2])
            mem_total_mb = float(parts[3])
            return {
                "available": True,
                "name": parts[0],
                "util_percent": int(float(parts[1])),
                "mem_used_gb": round(mem_used_mb / 1024.0, 1),
                "mem_total_gb": round(mem_total_mb / 1024.0, 1),
                "mem_percent": round((mem_used_mb / mem_total_mb) * 100, 1) if mem_total_mb > 0 else 0.0,
                "temp_c": int(float(parts[4])),
            }
    except Exception:
        pass
    return {"available": False}


def _get_ollama_ps() -> list[dict]:
    """Fetch currently loaded Ollama models (equivalent to `ollama ps`)."""
    if assistant is not None:
        try:
            res = assistant.ollama_client.ps()
            raw_models = res.get("models", []) if isinstance(res, dict) else getattr(res, "models", [])
            items = []
            for m in raw_models:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("model") or "unknown"
                    size = m.get("size", 0)
                    size_vram = m.get("size_vram", 0)
                    ctx = m.get("context_length") or "-"
                    expires = m.get("expires_at") or ""
                else:
                    name = getattr(m, "name", None) or getattr(m, "model", "unknown")
                    size = getattr(m, "size", 0) or 0
                    size_vram = getattr(m, "size_vram", 0) or 0
                    ctx = getattr(m, "context_length", None) or "-"
                    expires = str(getattr(m, "expires_at", "") or "")

                size_gb = round(size / (1024**3), 1) if size else 0.0
                if size and size_vram is not None:
                    gpu_pct = round((size_vram / size) * 100)
                    processor = f"{gpu_pct}% GPU" if gpu_pct >= 100 else (f"{gpu_pct}% GPU / {100 - gpu_pct}% CPU" if gpu_pct > 0 else "100% CPU")
                else:
                    processor = "GPU"

                items.append(
                    {
                        "name": name,
                        "size_gb": size_gb,
                        "processor": processor,
                        "context": ctx,
                        "until": expires[:19].replace("T", " ") if expires else "Active",
                    }
                )
            return items
        except Exception:
            pass
    return []


@app.get("/api/telemetry")
async def get_telemetry():
    global _cached_models, _cached_models_ts
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    now = time.time()
    if assistant is not None and (not _cached_models or now - _cached_models_ts > 30.0):
        try:
            res = assistant.ollama_client.list()
            raw_models = res.get("models", []) if isinstance(res, dict) else getattr(res, "models", [])
            parsed = []
            for m in raw_models:
                name = m.get("model") or m.get("name") if isinstance(m, dict) else getattr(m, "model", None)
                if name and "embed" not in name.lower():
                    parsed.append(name)
            if parsed:
                _cached_models = parsed
                _cached_models_ts = now
        except Exception:
            pass

    models = _cached_models or [assistant.llm_model if assistant else LLM_MODEL]

    return JSONResponse(
        {
            "cpu_percent": cpu,
            "cpu_threads": psutil.cpu_count(logical=True) or 0,
            "ram_percent": ram.percent,
            "ram_used_gb": round(ram.used / (1024**3), 1),
            "ram_total_gb": round(ram.total / (1024**3), 1),
            "mic_available": assistant.mic_available if assistant else False,
            "stt_backend": getattr(assistant, "stt_backend", "CUDA"),
            "current_whisper_model": getattr(assistant, "whisper_model", WHISPER_MODEL),
            "tts_backend": getattr(assistant, "tts_backend", "CUDA"),
            "current_model": assistant.llm_model if assistant else LLM_MODEL,
            "current_voice": assistant.tts_voice if assistant else "bm_george",
            "muted": assistant.muted if assistant else False,
            "state": assistant.state if assistant else "idle",
            "available_models": models,
            "output_devices": assistant.get_output_devices() if assistant else [],
            "selected_output_device": assistant.output_device if assistant else None,
            "gpu": _get_gpu_status(),
            "ollama_ps": _get_ollama_ps(),
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global always_listen_enabled, follow_up_until
    await websocket.accept()
    connected_clients.add(websocket)
    # When a browser client is connected, let the browser's Web Audio queue handle playback
    # so there is zero double-echo and audio plays through whichever speaker the browser uses,
    # unless host speaker mode is also desired.
    if assistant:
        assistant.play_on_host_speaker = True
        await websocket.send_text(
            json.dumps(
                {
                    "type": "init",
                    "state": assistant.state,
                    "model": assistant.llm_model,
                    "voice": assistant.tts_voice,
                    "mic_available": assistant.mic_available,
                    "muted": assistant.muted,
                }
            )
        )

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            action = data.get("action")

            if assistant is None:
                continue

            if action == "chat":
                user_text = (data.get("text") or "").strip()
                if user_text:
                    threading.Thread(
                        target=assistant.process_llm,
                        args=(user_text, False),
                        daemon=True,
                    ).start()

            elif action == "test_voice":
                threading.Thread(
                    target=assistant.speak,
                    args=("Coderstation audio systems are online and operating at peak efficiency, Sir.", False),
                    daemon=True,
                ).start()

            elif action == "toggle_continuous":
                always_listen_enabled = bool(data.get("enabled", False))
                if not always_listen_enabled:
                    follow_up_until = 0.0
                if always_listen_enabled and not assistant.mic_available:
                    assistant.recheck_microphone()
                broadcast_event({"type": "continuous_mode", "enabled": always_listen_enabled})

            elif action == "stop_speaking":
                follow_up_until = 0.0
                assistant.stop_speaking()

            elif action == "set_mute":
                assistant.muted = bool(data.get("muted", False))
                if assistant.muted:
                    assistant.stop_speaking()
                broadcast_event({"type": "mute_changed", "muted": assistant.muted})

            elif action == "set_model":
                new_model = (data.get("model") or "").strip()
                if new_model:
                    assistant.llm_model = new_model
                    broadcast_event({"type": "status_note", "message": f"Switched LLM to {new_model}"})

            elif action == "set_voice":
                new_voice = (data.get("voice") or "").strip()
                if new_voice:
                    assistant.tts_voice = new_voice
                    broadcast_event({"type": "status_note", "message": f"Switched TTS voice to {new_voice}"})

            elif action == "set_output_device":
                dev_id = data.get("device_id")
                if dev_id == "browser":
                    assistant.play_on_host_speaker = False
                    broadcast_event({"type": "status_note", "message": "Switched to Browser Audio Output"})
                else:
                    assistant.play_on_host_speaker = True
                    new_dev = int(dev_id) if dev_id not in (None, "") else None
                    assistant.set_output_device(new_dev)
                    broadcast_event({"type": "status_note", "message": f"Switched to PC Speaker (#{assistant.output_device or 'Default'})"})

            elif action == "set_host_speaker":
                assistant.play_on_host_speaker = bool(data.get("enabled", True))

            elif action == "clear_history":
                assistant.stop_speaking()
                assistant.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                broadcast_event({"type": "history_cleared"})

            elif action == "recheck_mic":
                ok = assistant.recheck_microphone()
                broadcast_event({"type": "mic_status", "mic_available": ok})

            elif action == "listen_now":
                if not manual_listen_active.is_set():
                    def _manual_voice_turn():
                        global follow_up_until
                        manual_listen_active.set()
                        try:
                            broadcast_event({"type": "status_note", "message": "Listening — speak your command now..."})
                            audio = assistant.record_command(passive_standby=False)
                            if audio.size >= 2400:
                                text = assistant.transcribe(audio)
                                if text:
                                    _, stripped = extract_wake_command(text)
                                    final_text = stripped if stripped else text
                                    assistant.process_llm(final_text, wait_for_speech=True)
                                    time.sleep(0.10)
                                    follow_up_until = time.time() + FOLLOW_UP_WINDOW_SECONDS
                                else:
                                    assistant._set_state("idle")
                                    broadcast_event({"type": "status_note", "message": "No clear speech detected."})
                            else:
                                assistant._set_state("idle")
                                broadcast_event({"type": "status_note", "message": ""})
                        except Exception as e:
                            assistant._set_state("idle")
                            broadcast_event({"type": "error", "message": str(e)})
                        finally:
                            manual_listen_active.clear()
                    threading.Thread(target=_manual_voice_turn, daemon=True).start()

            elif action == "set_whisper_model":
                new_stt = (data.get("model") or "").strip()
                if new_stt:
                    def _switch_stt():
                        broadcast_event({"type": "status_note", "message": f"Loading Whisper STT model '{new_stt}'..."})
                        try:
                            assistant.set_whisper_model(new_stt)
                            broadcast_event({"type": "status_note", "message": f"Switched STT engine to Whisper '{new_stt}' ({assistant.stt_backend})"})
                        except Exception as e:
                            broadcast_event({"type": "error", "message": f"Failed to load Whisper '{new_stt}': {e}"})
                    threading.Thread(target=_switch_stt, daemon=True).start()

    except WebSocketDisconnect:
        connected_clients.discard(websocket)
    except Exception:
        connected_clients.discard(websocket)




@app.get("/coderstation.png")
async def get_brand_png():
    return FileResponse(ICON_PNG_PATH, media_type="image/png")


@app.get("/favicon.ico")
async def get_favicon():
    if ICON_ICO_PATH.exists():
        return FileResponse(ICON_ICO_PATH, media_type="image/x-icon")
    return FileResponse(ICON_PNG_PATH, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


def _apply_windows_console_icon():
    """Set the Windows console/taskbar window icon to coderstation.ico when running on Windows."""
    if os.name != "nt" or not ICON_ICO_PATH.exists():
        return
    try:
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if not hwnd:
            return
        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x00000010
        LR_DEFAULTSIZE = 0x00000040
        WM_SETICON = 0x0080
        hicon_small = ctypes.windll.user32.LoadImageW(None, str(ICON_ICO_PATH), IMAGE_ICON, 16, 16, LR_LOADFROMFILE)
        hicon_big = ctypes.windll.user32.LoadImageW(None, str(ICON_ICO_PATH), IMAGE_ICON, 32, 32, LR_LOADFROMFILE | LR_DEFAULTSIZE)
        if hicon_small:
            ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, 0, hicon_small)
        if hicon_big:
            ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, 1, hicon_big)
    except Exception:
        pass


def launch_web_gui(llm_model: str = LLM_MODEL, whisper_model: str = WHISPER_MODEL, port: int = 8000):
    global assistant
    _apply_windows_console_icon()
    print("[WEB GUI] Initializing Coderstation Core Engine...")
    assistant = CoderstationAssistant(
        llm_model=llm_model,
        whisper_model=whisper_model,
        use_wake_word=False,
        event_callback=broadcast_event,
    )
    url = f"http://127.0.0.1:{port}"
    print(f"[WEB GUI] Launching Coderstation Command Center at {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    launch_web_gui()
