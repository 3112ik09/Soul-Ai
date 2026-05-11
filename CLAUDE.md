# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A macOS voice assistant named "Siri" — wake word activated, real-time speech-to-text, LLM response, TTS output, and a browser-based dashboard with a Live2D avatar. There is no Discord bot here despite the repo name; the project evolved into a local voice assistant.

## Running

```bash
# Start the voice assistant (loads models, blocks until Ctrl+C)
python voice_chat.py

# Serve the dashboard (separate terminal, then open http://localhost:8080/dashboard.html)
python -m http.server 8080

# Test intent routing and search/code dispatch without audio
python skills.py "search for latest AI news"
python skills.py "write a python script to list files"
```

`voice_chat.py` exits on startup if `./fused_model` is missing — the MLX fine-tuned model must be present.

## Fine-tuning Pipeline

Two training scripts exist. Use `train_human_chat.py` for natural conversation quality; use `train_siri.py` for personality-only retraining.

```bash
source .venv2/bin/activate   # Python 3.12 — has mlx-lm + training deps

# Recommended: train on HuggingFace human conversation datasets → fuse to ./fused_model
python train_human_chat.py               # full pipeline (data → train → fuse)
python train_human_chat.py --data-only   # just build train.jsonl / valid.jsonl
python train_human_chat.py --fuse-only   # fuse existing adapters (skip re-training)

# Original personality-only training → GGUF for Ollama
python train_siri.py         # outputs adapters to ./siri-mistral
python export_model.py       # fuses + exports to ./siri-mistral-fp16 / gguf
```

`train_human_chat.py` downloads three datasets automatically:
- `empathetic_dialogues` — emotional, warm responses
- `daily_dialog` — everyday casual conversation
- `Anthropic/hh-rlhf` — natural helpful dialogue (filtered to short responses)

Custom Siri personality examples are weighted ×30 so the persona survives the large conversation dataset. All responses are filtered to 15–160 chars (voice-appropriate). The `--mask-prompt` flag trains only on response tokens.

The `.venv2/` env (Python 3.12) has mlx-lm and training dependencies. The `venv/` env (Python 3.14) is for the runtime.

## Architecture

The system has three concurrent threads:

1. **`mic_listener()`** — captures audio with Silero VAD (threshold 0.3), fires a transcription thread per utterance after 1500ms of silence. Microphone is hardcoded to `device=1` — change if wrong device.

2. **`transcribe_and_queue()`** — runs faster-whisper, detects wake/stop words, puts commands on `input_queue`. Only passes input to the queue when `is_awake=True` (toggled by wake word "Hey Siri").

3. **`ai_loop()`** — the main consumer of `input_queue`. Routes each input through `skills.route()` first; if intent is `search` or `code`, calls `skills.dispatch()`; otherwise streams the fused MLX model for plain chat.

A fourth thread runs the **WebSocket server** (port 8765) that pushes JSON messages to `dashboard.html` for live updates (avatar state, chat log, code panel, web results).

### Intent Routing (`skills.py`)

`route(text)` returns `(intent, query)` where intent is `search`, `code`, or `chat`. Uses keyword scoring, not regex anchoring — handles garbled/polite phrasing. Hard-override phrases take priority over scoring. The threshold to trigger a skill is a combined score ≥ 3.

- **search** → DuckDuckGo HTML scrape (no API key), results formatted as HTML injected into dashboard
- **code** → Ollama at `localhost:11434` with `qwen2.5-coder:3b`, returns JSON `{language, code, explanation}`
- **chat** → fused MLX model via `mlx_lm.stream_generate`, max 60 tokens, last 6 history turns

### LLM Streaming in Chat

`stream_llm()` uses `CLAUSE_RE` for the first chunk (speaks immediately on comma/semicolon boundaries) then `SENTENCE_RE` for subsequent chunks. This minimizes time-to-first-speech. Each chunk is sent to TTS (`say` command) and the WebSocket simultaneously.

### Emotion System

`EmotionalState` tracks a float `mood` in `[-1.0, 1.0]`, updated by keyword sentiment in each exchange and decayed 5% each idle tick. The current mood label is injected into the system prompt at inference time.

### WebSocket Protocol

All messages are JSON. Types:
- `state` — avatar emotion + talking flag + subtitle text
- `chat` — role (`user` | `siri`) + content for chat log
- `code` — code string + language + explanation for code panel
- `output` — arbitrary HTML/text for web results panel

### History Handling

Only the last 6 turns are sent to the LLM. Skill actions are stored as placeholders (`[User requested web search]` / `[Action completed. Details are on the user's screen.]`) to prevent the LLM from hallucinating answers to questions it couldn't have seen.

### `mac_agent.py`

Standalone module (not yet integrated into `voice_chat.py`) — takes a natural language command, asks Mistral via Ollama to generate AppleScript or shell, and executes it. Call `execute(command)` to use it.

## Key Config Constants

All in top of each file:

| File | Constant | Default | Notes |
|---|---|---|---|
| `voice_chat.py` | `MODEL` | `"mistral"` | Unused — actual model is `./fused_model` via MLX |
| `voice_chat.py` | `WHISPER_MODEL` | `"base"` | faster-whisper model size |
| `voice_chat.py` | `SILENCE_MS` | `1500` | ms of silence before transcribing |
| `voice_chat.py` | `WS_PORT` | `8765` | WebSocket port |
| `skills.py` | `CODE_MODEL` | `"qwen2.5-coder:3b"` | Ollama model for code gen |
| `skills.py` | `SEARCH_N` | `6` | Number of DDG results |
