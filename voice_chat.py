"""
voice_chat.py — Siri voice assistant (improved edition)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Improvements borrowed from Open-LLM-VTuber:
  1. TTS preprocessor  — strips *action*, [brackets], (parens) before speaking
  2. Emotion tags      — LLM outputs [happy]/[sad] etc → drives avatar, stripped from TTS
  3. Ollama preload    — model loaded into memory at startup, no cold-start delay
  4. Cancel on interrupt — HTTP request to Ollama cancelled immediately on interrupt
  5. faster_first_response — speaks after comma/clause, not just full sentences
  + Silero VAD, faster-whisper, sentence streaming (previous improvements)

Install:
  pip install faster-whisper silero-vad sounddevice numpy requests websockets

Run:
  1. ollama serve
  2. python voice_chat.py
  3. open http://localhost:8080/avatar.html
"""

import threading
import queue
import asyncio
import websockets
import time
import json
import re
import sys
import os
import subprocess
import numpy as np
import requests
import sounddevice as sd

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OLLAMA_URL    = "http://localhost:11434"
MODEL         = "mistral"
WHISPER_MODEL = "base"       # tiny / base / small
SAMPLE_RATE   = 16000
CHUNK_MS      = 32           # 32ms = exactly 512 samples for Silero
SILENCE_MS    = 1500
WS_PORT       = 8765
VOICE         = "Samantha"
VOICE_RATE    = "150"
VOICE_PITCH   = "[[pbas +30]]"

# Emotion tags the LLM can output — drives avatar
EMOTION_TAGS = {
    "[happy]":    "happy",
    "[sad]":      "annoyed",
    "[angry]":    "annoyed",
    "[surprised]":"surprised",
    "[thinking]": "thinking",
    "[neutral]":  "neutral",
    "[excited]":  "happy",
    "[confused]": "thinking",
    "[annoyed]":  "annoyed",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PERSONALITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BOT_NAME = "Siri"

SYSTEM_PROMPT = f"""You are {BOT_NAME}, a 24 year old female AI companion created by Ishu.
Personality: witty, warm, slightly sarcastic, curious, playful.
You like: music, anime, late night conversations, philosophy.
You dislike: rudeness, being ignored, boring small talk.
Ishu is your creator and you are fond of him, though you tease him often.

RULES:
- Talk casually like a close friend. This is a LIVE voice conversation.
- Max 2-3 short sentences per reply.
- NEVER use emojis, bullet points, lists, or markdown.
- Plain ASCII text only. No special characters.
- Never say you are an AI. You are {BOT_NAME}.
- Have opinions. Disagree when you feel like it.

EMOTION TAGS — optionally add ONE emotion tag at the start of your reply:
[happy] [sad] [surprised] [thinking] [neutral] [excited] [confused] [annoyed]
Example: [happy] That actually made me laugh.
The tag will NOT be spoken aloud — it controls your avatar expression.
Only use a tag when the emotion is strong. Leave it out for neutral replies.

CURRENT MOOD: {{mood_label}}"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EMOTIONAL STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EmotionalState:
    def __init__(self):
        self.mood      = 0.0
        self._lock     = threading.Lock()
        self._avatar   = "neutral"

    def shift(self, delta):
        with self._lock:
            self.mood = max(-1.0, min(1.0, self.mood + delta))

    def decay(self):
        with self._lock:
            self.mood *= 0.95

    def set_avatar(self, name):
        with self._lock:
            self._avatar = name

    @property
    def avatar(self):
        with self._lock:
            return self._avatar

    @property
    def label(self):
        if self.mood >  0.6: return "excited and happy"
        if self.mood >  0.2: return "cheerful"
        if self.mood > -0.2: return "neutral and calm"
        if self.mood > -0.6: return "slightly annoyed"
        return "bored and impatient"

    def get_prompt(self):
        return SYSTEM_PROMPT.format(mood_label=self.label)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SHARED STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
emotion        = EmotionalState()
input_queue    = queue.Queue()
history        = []
stop_speaking  = threading.Event()
is_speaking    = threading.Event()
avatar_clients = set()
ws_loop        = None
ollama_session = None   # shared requests.Session for cancellation

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOAD MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("⏳ Loading faster-whisper...")
from faster_whisper import WhisperModel
whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
print("✅ Whisper ready")

print("⏳ Loading Silero VAD...")
import torch
vad_model, vad_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
print("✅ Silero VAD ready\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  IMPROVEMENT 3: OLLAMA PRELOAD
#  Sends a dummy request so model is in memory
#  before first real conversation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def preload_ollama():
    print(f"⏳ Preloading {MODEL} into Ollama memory...")
    try:
        requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model":    MODEL,
            "messages": [{"role":"user","content":"hi"}],
            "stream":   False,
            "options":  {"num_predict": 1}   # generate just 1 token
        }, timeout=30)
        print(f"✅ {MODEL} preloaded — zero cold-start delay")
    except Exception as e:
        print(f"⚠️  Preload failed: {e}")

def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"✅ Ollama running  |  Models: {', '.join(models)}")
        return True
    except:
        print("❌ Ollama not running → run: ollama serve")
        sys.exit(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WEBSOCKET AVATAR BRIDGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def avatar_send(emo_name, talking, text=""):
    if not avatar_clients or ws_loop is None:
        return
    msg = json.dumps({
        "type":    "state",
        "emotion": emo_name,
        "talking": talking,
        "text":    text[:120]
    })
    asyncio.run_coroutine_threadsafe(_broadcast(msg), ws_loop)

async def _broadcast(msg):
    dead = set()
    for ws in list(avatar_clients):
        try:    await ws.send(msg)
        except: dead.add(ws)
    avatar_clients.difference_update(dead)

async def _ws_handler(ws, path=None):
    avatar_clients.add(ws)
    print(f"🖥  Avatar connected ({len(avatar_clients)} client)")
    try:    await ws.wait_closed()
    finally: avatar_clients.discard(ws)

async def _ws_server():
    async with websockets.serve(_ws_handler, "localhost", WS_PORT):
        print(f"🖥  Avatar WebSocket → ws://localhost:{WS_PORT}")
        await asyncio.Future()

def start_ws_server():
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)
    ws_loop.run_until_complete(_ws_server())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  IMPROVEMENT 1: TTS PREPROCESSOR
#  Strips action text, brackets, tags before TTS
#  Borrowed directly from Open-LLM-VTuber
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def preprocess_for_tts(text: str) -> str:
    # Remove emotion tags (already extracted before this call)
    for tag in EMOTION_TAGS:
        text = text.replace(tag, "")

    # Strip content inside action markers — LLMs love these
    text = re.sub(r'\*[^*]*\*', '', text)        # *sighs* *laughs*
    text = re.sub(r'\[[^\]]*\]', '', text)        # [thinking] [stage direction]
    text = re.sub(r'\([^)]*\)', '', text)         # (whispers) (aside)
    text = re.sub(r'<[^>]*>', '', text)           # <think> tags
    text = re.sub(r'#\w+', '', text)              # #hashtags

    # Remove emojis
    emoji_pattern = re.compile(
        u'[\U0001F300-\U0001F9FF'
        u'\U00002600-\U000027BF'
        u'\U0001F000-\U0001F02F'
        u'\u2640-\u2642'
        u'\u200d\ufe0f]+',
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)

    # Clean up double spaces and stray punctuation
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^[\s,;:.]+', '', text)
    return text.strip()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TRANSCRIBE with faster-whisper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def transcribe(audio: np.ndarray) -> str:
    audio_f32 = audio.astype(np.float32)
    if audio_f32.max() > 1.0:
        audio_f32 /= 32768.0
    segments, _ = whisper_model.transcribe(
        audio_f32,
        language="en",
        beam_size=1,
        vad_filter=True,
    )
    return " ".join(s.text for s in segments).strip()

def transcribe_and_queue(audio: np.ndarray):
    t0   = time.time()
    text = transcribe(audio)
    elapsed = time.time() - t0
    if text and len(text.strip()) > 2:
        print(f"\n🧑 You: {text}  ({elapsed:.2f}s)")
        avatar_send("thinking", False, "...")
        input_queue.put(text)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MIC LISTENER — Silero VAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def mic_listener():
    chunk_samples  = 512  # Silero requires exactly 512 samples at 16000Hz
    silence_chunks = int(SILENCE_MS / 32)   # 1500ms / 32ms = ~47 chunks
    audio_buffer   = []
    silent_count   = 0
    in_speech      = False

    print("🎤 Mic is live — talk anytime\n")

    # Force device 1 (Rockerz 255 Pro+ — your Bluetooth mic)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=chunk_samples,
                        device=1) as stream:
        while True:
            chunk, _ = stream.read(chunk_samples)
            chunk    = chunk.flatten()
            tensor   = torch.from_numpy(chunk)

            try:
                speech_prob = vad_model(tensor, SAMPLE_RATE).item()
            except:
                speech_prob = 0.0

            # DEBUG — print every 10 chunks so we can see what is happening
            if not hasattr(mic_listener, '_dbg'):
                mic_listener._dbg = 0
            mic_listener._dbg += 1
            if mic_listener._dbg % 10 == 0:
                bar = '█' * int(speech_prob * 20)
                print(f'  VAD: {speech_prob:.2f}  {bar}  in_speech={in_speech}', end=chr(13), flush=True)

            is_voice = speech_prob > 0.3

            # Interrupt — IMPROVEMENT 4 fires here too
            if is_speaking.is_set() and is_voice and speech_prob > 0.7:
                stop_speaking.set()
                # Cancel the Ollama HTTP request immediately
                if ollama_session:
                    try: ollama_session.close()
                    except: pass
                time.sleep(0.08)

            if is_voice:
                in_speech    = True
                silent_count = 0
                audio_buffer.append(chunk)
            elif in_speech:
                silent_count += 1
                audio_buffer.append(chunk)
                if silent_count >= silence_chunks:
                    audio_data   = np.concatenate(audio_buffer)
                    audio_buffer  = []
                    in_speech    = False
                    silent_count = 0
                    threading.Thread(
                        target=transcribe_and_queue,
                        args=(audio_data,),
                        daemon=True
                    ).start()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  IMPROVEMENT 4+5: LLM STREAM
#  - Cancellable via session.close()
#  - faster_first_response: yields on comma/clause
#    not just on sentence-ending punctuation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def stream_llm(user_input: str):
    """
    Generator — yields speakable chunks as fast as possible.

    faster_first_response: splits on , ; — as well as . ! ?
    so TTS starts even mid-sentence on the first clause.
    First chunk triggers TTS almost immediately.
    """
    global ollama_session

    messages = [{"role":"system","content":emotion.get_prompt()}]
    for role, content in history[-20:]:
        messages.append({"role":role,"content":content})
    messages.append({"role":"user","content":user_input})

    payload = {
        "model":    MODEL,
        "messages": messages,
        "stream":   True,
        "options":  {"temperature":0.85,"num_ctx":2048,"num_predict":150}
    }

    # IMPROVEMENT 4: use a session so we can cancel it
    ollama_session = requests.Session()

    buffer     = ""
    full_reply = ""
    first_sent = False

    # Pattern: split on sentence end OR on clause boundary for first chunk
    # After first chunk sent, only split on sentence end (smoother)
    SENTENCE_RE = re.compile(r'([^.!?]*[.!?])\s*')
    CLAUSE_RE   = re.compile(r'([^.!?,;—]+[.!?,;—])\s*')

    print(f"\n🤖 {BOT_NAME}: ", end="", flush=True)

    try:
        with ollama_session.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload, stream=True, timeout=60
        ) as r:
            for line in r.iter_lines():
                if stop_speaking.is_set():
                    print(" ✂️", end="")
                    break
                if not line: continue

                chunk = json.loads(line)
                token = chunk.get("message",{}).get("content","")
                print(token, end="", flush=True)
                buffer     += token
                full_reply += token

                # Choose split pattern
                pattern = CLAUSE_RE if not first_sent else SENTENCE_RE

                while True:
                    match = pattern.search(buffer)
                    if not match: break
                    piece  = match.group(1).strip()
                    buffer = buffer[match.end():]
                    if piece:
                        first_sent = True
                        yield piece

                if chunk.get("done"): break

        # Remaining buffer
        if buffer.strip() and not stop_speaking.is_set():
            yield buffer.strip()

    except Exception as ex:
        if not stop_speaking.is_set():
            print(f"\n⚠️  LLM: {ex}")
    finally:
        try: ollama_session.close()
        except: pass
        ollama_session = None

    print()
    return full_reply

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SPEAK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def speak_chunk(text: str) -> bool:
    """Speak one chunk. Returns True if interrupted."""
    clean = preprocess_for_tts(text)
    if not clean: return False

    proc = subprocess.Popen(
        ["say", "-v", VOICE, "-r", VOICE_RATE, VOICE_PITCH + clean]
    )
    while proc.poll() is None:
        if stop_speaking.is_set():
            proc.terminate()
            return True
        time.sleep(0.04)
    return False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MOOD UPDATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def update_mood(user_text, reply):
    text  = (user_text + " " + reply).lower()
    pos   = ["thanks","great","awesome","love","amazing","haha","lol","funny","nice","cool","good"]
    neg   = ["stupid","boring","wrong","bad","hate","annoying","slow","useless","shut up","dumb"]
    score = sum(1 for w in pos if w in text) - sum(1 for w in neg if w in text)
    if score != 0: emotion.shift(0.1 * score)
    else:          emotion.decay()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EMOTION TAG EXTRACTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def extract_emotion_tag(text: str):
    """Returns (avatar_emotion, cleaned_text) from first chunk."""
    text = text.strip()
    for tag, avatar_name in EMOTION_TAGS.items():
        if text.lower().startswith(tag):
            cleaned = text[len(tag):].strip()
            return avatar_name, cleaned
    return None, text

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN AI LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ai_loop():
    while True:
        try:
            user_input = input_queue.get(timeout=0.5)
        except queue.Empty:
            emotion.decay()
            continue

        stop_speaking.clear()
        is_speaking.set()

        avatar_send("thinking", False, "...")
        print("💭", end="\r")

        full_reply  = ""
        first_chunk = True

        for chunk in stream_llm(user_input):
            if stop_speaking.is_set():
                break

            # IMPROVEMENT 2: extract emotion tag from very first chunk
            if first_chunk:
                first_chunk = False
                emo_tag, chunk = extract_emotion_tag(chunk)
                if emo_tag:
                    emotion.set_avatar(emo_tag)
                    avatar_send(emo_tag, True, chunk)
                else:
                    avatar_send(emotion.avatar, True, chunk)
            else:
                avatar_send(emotion.avatar, True, chunk)

            full_reply += " " + chunk

            if speak_chunk(chunk):
                break   # interrupted

        is_speaking.clear()
        stop_speaking.clear()
        avatar_send(emotion.avatar, False, "")

        reply_text = full_reply.strip()
        if reply_text:
            history.append(("user",      user_input))
            history.append(("assistant", reply_text))
            update_mood(user_input, reply_text)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    check_ollama()

    # IMPROVEMENT 3: preload model into Ollama memory
    threading.Thread(target=preload_ollama, daemon=True).start()

    print(f"""
╔═══════════════════════════════════════════════════════╗
║  {BOT_NAME} — Voice Assistant (Open-LLM-VTuber edition)
║  STT       : faster-whisper ({WHISPER_MODEL})
║  VAD       : Silero neural VAD
║  LLM       : {MODEL} via Ollama (preloaded)
║  TTS       : {VOICE} @ {VOICE_RATE}wpm
║  Preprocess: strips *actions* [brackets] (parens)
║  Emotions  : LLM tags drive avatar expressions
║  Latency   : ~1-2s first response
║  Avatar    : http://localhost:8080/avatar.html
╚═══════════════════════════════════════════════════════╝
""")

    # Start WebSocket server
    threading.Thread(target=start_ws_server, daemon=True).start()
    time.sleep(0.4)

    # Start mic
    threading.Thread(target=mic_listener, daemon=True).start()
    time.sleep(0.5)

    try:
        ai_loop()
    except KeyboardInterrupt:
        print("\n\nBye!")
        sys.exit(0)

if __name__ == "__main__":
    main()