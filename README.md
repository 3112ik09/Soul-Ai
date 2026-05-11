# Siri — On-Device Voice Assistant

Fully local macOS voice assistant. No cloud APIs. Built on Apple Silicon with MLX.

**Wake word → STT → Intent routing → LLM / Tools → TTS → Live2D dashboard**

---

## Requirements

- macOS Apple Silicon (M1/M2/M3)
- Python 3.12
- [Ollama](https://ollama.ai) — for code generation

---

## Setup

```bash
# Runtime
python3.12 -m venv venv && source venv/bin/activate
pip install faster-whisper mlx-lm websockets sounddevice torch requests

# Code model
ollama pull qwen2.5-coder:3b

# Training env (for fine-tuning only)
python3.12 -m venv .venv2 && source .venv2/bin/activate
pip install mlx-lm datasets huggingface_hub
```

Set your mic device in `voice_chat.py` → `device=1` in `mic_listener()`.  
List devices: `python -c "import sounddevice as sd; print(sd.query_devices())"`

---

## Fine-tune & Run

```bash
# Build the model (downloads data → trains → fuses into ./fused_model)
source .venv2/bin/activate
python train_human_chat.py

# Start assistant
source venv/bin/activate
python voice_chat.py

# Dashboard (separate terminal)
python -m http.server 8080
# Open http://localhost:8080/dashboard.html
```

---

## What it does

| You say | Action |
|---|---|
| `Hey Siri` | Wake up |
| `Siri stop` | Sleep |
| `Search for X` | DuckDuckGo → results on dashboard |
| `Write a Python script to X` | Code generation → code panel |
| `Play X` / `Next song` / `Mute` | Spotify + volume control |
| `Open X` / `Brightness up` | App launch / display control |
| `Screenshot` / `My schedule` | System tools |
| Anything else | Chat with fine-tuned Mistral-7B |

---

## Stack

`faster-whisper` · `Silero VAD` · `MLX-LM + Mistral-7B` · `Ollama qwen2.5-coder` · `macOS say` · `pixi-live2d-display` · `SQLite`

## Credits

Live2D model used in the dashboard:
- [BOOTH.pm Model Asset](https://booth.pm/en/items/7483530?registration=1&utm_source=chatgpt.com)

Used for personal/non-commercial purposes only.
