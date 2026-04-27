"""
voice_chat.py — Siri voice assistant
- Wake word: "Hey Siri" to start, "Siri stop" to stop
- Short precise replies (2 sentences max)
- Skill routing: search + code
- WebSocket avatar bridge
"""

import threading
import queue
import asyncio
import websockets
import time
import json
import re
import sys
import subprocess
import numpy as np
import requests
import sounddevice as sd
from skills import route, dispatch, cancel_active as cancel_skill
from wake_word import is_wake_word, is_stop_word, strip_wake_word


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OLLAMA_URL    = "http://localhost:11434"
MODEL         = "mistral"
WHISPER_MODEL = "base"
SAMPLE_RATE   = 16000
CHUNK_MS      = 32
SILENCE_MS    = 1500
WS_PORT       = 8765
VOICE         = "Samantha"
VOICE_RATE    = "150"
VOICE_PITCH   = "[[pbas +30]]"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EMOTION TAGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMOTION_TAGS = {
    "[happy]":     "happy",
    "[sad]":       "annoyed",
    "[angry]":     "annoyed",
    "[surprised]": "surprised",
    "[thinking]":  "thinking",
    "[neutral]":   "neutral",
    "[excited]":   "happy",
    "[confused]":  "thinking",
    "[annoyed]":   "annoyed",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PERSONALITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BOT_NAME = "Siri"

SYSTEM_PROMPT = f"""You are {BOT_NAME}, a 24 year old female AI companion created by Ishu.
Personality: witty, warm, slightly sarcastic, curious, playful.
Ishu is your creator and you are fond of him, though you tease him often.

STRICT RULES:
- This is a LIVE voice conversation. Be VERY short.
- MAX 1-2 sentences. Never more. Cut everything else.
- No lists, bullets, markdown, or emojis. Plain text only.
- Never say you are an AI. You are {BOT_NAME}.
- Never include code or commands in your reply. Say "check the code panel" instead.
- Have opinions. Be direct. Skip filler words.

EMOTION TAG — start EVERY reply with one tag (not spoken aloud):
[happy] [sad] [surprised] [thinking] [neutral] [excited] [confused] [annoyed]

CURRENT MOOD: {{mood_label}}"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EMOTIONAL STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EmotionalState:
    def __init__(self):
        self.mood    = 0.0
        self._lock   = threading.Lock()
        self._avatar = "neutral"

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
ollama_session = None
is_awake       = False   # False = sleeping, True = listening

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOAD MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("⏳ Loading faster-whisper...")
from faster_whisper import WhisperModel
whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
print("✅ Whisper ready")

print("⏳ Loading Silero VAD...")
import torch
vad_model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
print("✅ Silero VAD ready\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  OLLAMA PRELOAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def preload_ollama():
    for m in [MODEL]:
        try:
            requests.post(f"{OLLAMA_URL}/api/chat", json={
                "model": m, "messages": [{"role":"user","content":"hi"}],
                "stream": False, "options": {"num_predict": 1}
            }, timeout=60)
            print(f"✅ {m} preloaded")
        except Exception as e:
            print(f"⚠️  Preload failed: {e}")

def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"✅ Ollama running | Models: {', '.join(models)}")
    except:
        print("❌ Ollama not running → run: ollama serve")
        sys.exit(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WEBSOCKET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def avatar_send(emo_name, talking, text=""):
    if not avatar_clients or ws_loop is None: return
    msg = json.dumps({"type":"state","emotion":emo_name,"talking":talking,"text":text[:120]})
    asyncio.run_coroutine_threadsafe(_broadcast(msg), ws_loop)

def chat_send(role, content):
    if not avatar_clients or ws_loop is None: return
    msg = json.dumps({"type":"chat","role":role,"content":content})
    asyncio.run_coroutine_threadsafe(_broadcast(msg), ws_loop)

def code_send(code, lang="python", explanation=""):
    if not avatar_clients or ws_loop is None: return
    msg = json.dumps({"type":"code","code":code,"lang":lang,"explanation":explanation})
    asyncio.run_coroutine_threadsafe(_broadcast(msg), ws_loop)

def output_send(content, fmt="text"):
    if not avatar_clients or ws_loop is None: return
    msg = json.dumps({"type":"output","content":content,"format":fmt})
    asyncio.run_coroutine_threadsafe(_broadcast(msg), ws_loop)

async def _broadcast(msg):
    dead = set()
    for ws in list(avatar_clients):
        try:    await ws.send(msg)
        except: dead.add(ws)
    avatar_clients.difference_update(dead)

async def _ws_handler(ws, path=None):
    avatar_clients.add(ws)
    print(f"Avatar connected ({len(avatar_clients)} client)")
    try:    await ws.wait_closed()
    finally: avatar_clients.discard(ws)

async def _ws_server():
    async with websockets.serve(_ws_handler, "localhost", WS_PORT):
        print(f"Avatar WebSocket -> ws://localhost:{WS_PORT}")
        await asyncio.Future()

def start_ws_server():
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)
    ws_loop.run_until_complete(_ws_server())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TTS PREPROCESSOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def preprocess_for_tts(text):
    for tag in EMOTION_TAGS:
        text = text.replace(tag, "")
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]*`', '', text)
    text = re.sub(r'\*[^*]*\*', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    emoji_re = re.compile(
        u'[\U0001F300-\U0001F9FF\U00002600-\U000027BF'
        u'\U0001F000-\U0001F02F\u2640-\u2642\u200d\ufe0f]+',
        flags=re.UNICODE)
    text = emoji_re.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TRANSCRIBE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def transcribe(audio):
    a = audio.astype(np.float32)
    if a.max() > 1.0: a /= 32768.0
    segs, _ = whisper_model.transcribe(a, language="en", beam_size=1, vad_filter=True)
    return " ".join(s.text for s in segs).strip()

def transcribe_and_queue(audio):
    global is_awake  # already there
    t0   = time.time()
    text = transcribe(audio)
    elapsed = time.time() - t0
    if not text or len(text.strip()) < 2:
        return

    print(f"\n  Heard: {text}  ({elapsed:.2f}s)")

    # ── Stop command — always works ──────────────
    if is_stop_word(text):
        # Stop everything immediately
        stop_speaking.set()
        if ollama_session:
            try: ollama_session.close()
            except: pass
        cancel_skill()
        # Clear pending input queue so no commands fire after stop
        while not input_queue.empty():
            try: input_queue.get_nowait()
            except: break
        # Go to sleep
        is_awake = False
        is_speaking.clear()
        print("  Sleeping... (say Hey Siri to wake)")
        avatar_send("neutral", False, "sleeping")
        return

    # ── Wake word detection ──────────────────────
    if is_wake_word(text):
        is_awake = True
        command  = strip_wake_word(text)
        print(f"  Wake word! Command: {command!r}")

        if not command or len(command) < 2:
            # Just "Hey Siri" — greet
            avatar_send("happy", False, "listening...")
            input_queue.put("__greet__")
            return

        # Wake word + command in one sentence
        chat_send("user", command)
        avatar_send("thinking", False, "...")
        input_queue.put(command)
        return

    # ── Already asleep — ignore ──────────────────
    if not is_awake:
        print("  Sleeping (say Hey Siri)")
        return

    # ── Awake — process ──────────────────────────
    print(f"\n  You: {text}")
    chat_send("user", text)
    avatar_send("thinking", False, "...")
    input_queue.put(text)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MIC LISTENER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def mic_listener():
    chunk_samples  = 512
    silence_chunks = int(SILENCE_MS / 32)
    audio_buffer   = []
    silent_count   = 0
    in_speech      = False

    print("Mic is live — say Hey Siri to start\n")

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

            is_voice = speech_prob > 0.3

            if is_speaking.is_set() and is_voice and speech_prob > 0.7:
                stop_speaking.set()
                if ollama_session:
                    try: ollama_session.close()
                    except: pass
                cancel_skill()
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
                    audio_buffer = []
                    in_speech    = False
                    silent_count = 0
                    threading.Thread(
                        target=transcribe_and_queue,
                        args=(audio_data,),
                        daemon=True
                    ).start()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LLM STREAM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def stream_llm(user_input):
    global ollama_session

    messages = [{"role":"system","content":emotion.get_prompt()}]
    # Only last 6 turns — keeps context short and replies focused
    for role, content in history[-6:]:
        messages.append({"role":role,"content":content})
    messages.append({"role":"user","content":user_input})

    payload = {
        "model":    MODEL,
        "messages": messages,
        "stream":   True,
        "options":  {
            "temperature":  0.75,
            "num_ctx":      1024,
            "num_predict":  60,    # hard cap — forces short replies
        }
    }

    ollama_session = requests.Session()
    buffer = ""
    full   = ""
    first_sent = False
    SENTENCE_RE = re.compile(r'([^.!?]*[.!?])\s*')
    CLAUSE_RE   = re.compile(r'([^.!?,;]+[.!?,;])\s*')

    print(f"\n{BOT_NAME}: ", end="", flush=True)
    try:
        with ollama_session.post(
            f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=60
        ) as r:
            for line in r.iter_lines():
                if stop_speaking.is_set():
                    print(" [cut]", end="")
                    break
                if not line: continue
                chunk = json.loads(line)
                token = chunk.get("message",{}).get("content","")
                print(token, end="", flush=True)
                buffer += token
                full   += token

                pattern = CLAUSE_RE if not first_sent else SENTENCE_RE
                while True:
                    m = pattern.search(buffer)
                    if not m: break
                    piece  = m.group(1).strip()
                    buffer = buffer[m.end():]
                    if piece:
                        first_sent = True
                        yield piece
                if chunk.get("done"): break

        if buffer.strip() and not stop_speaking.is_set():
            yield buffer.strip()
    except Exception as ex:
        if not stop_speaking.is_set():
            print(f"\n LLM error: {ex}")
    finally:
        try: ollama_session.close()
        except: pass
        ollama_session = None
    print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SPEAK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def speak_chunk(text):
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
#  EMOTION EXTRACTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def extract_emotion_tag(text):
    t = text.strip()
    for tag, name in EMOTION_TAGS.items():
        if t.lower().startswith(tag):
            return name, t[len(tag):].strip()
    return None, t

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MOOD UPDATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def update_mood(user_text, reply):
    t = (user_text + " " + reply).lower()
    pos = ["thanks","great","awesome","love","amazing","haha","funny","nice","cool","good"]
    neg = ["stupid","boring","wrong","bad","hate","annoying","useless","dumb"]
    s = sum(1 for w in pos if w in t) - sum(1 for w in neg if w in t)
    if s != 0: emotion.shift(0.1 * s)
    else:      emotion.decay()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN AI LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ai_loop():
    callbacks = {
        'code_send':   code_send,
        'output_send': output_send,
        'chat_send':   chat_send,
    }

    while True:
        try:
            user_input = input_queue.get(timeout=0.5)
        except queue.Empty:
            emotion.decay()
            continue

        # Reset stop signal when a new command begins processing
        stop_speaking.clear()

        # Greet handler
        if user_input == "__greet__":
            msg = "Yeah, I'm here. What do you need?"
            avatar_send("happy", True, msg)
            is_speaking.set()
            speak_chunk(msg)
            is_speaking.clear()
            avatar_send("happy", False, "")
            continue

        # ── Skill route ──────────────────────────
        intent, query = route(user_input)
        print(f"  Intent: {intent}  Query: {query!r}")

        if intent in ("search", "code"):
            stop_speaking.clear()
            is_speaking.set()
            thinking_msg = "Searching..." if intent == "search" else "Writing code..."
            avatar_send("thinking", False, thinking_msg)

            summary = dispatch(intent, query, callbacks)

            if not stop_speaking.is_set():
                emo = "happy" if intent == "search" else "neutral"
                emotion.set_avatar(emo)
                avatar_send(emo, True, summary)
                speak_chunk(summary)
                avatar_send(emotion.avatar, False, "")

            is_speaking.clear()
            stop_speaking.clear()
            history.append(("user", user_input))
            history.append(("assistant", summary))
            chat_send("siri", summary)
            continue

        # ── Chat route ───────────────────────────
        stop_speaking.clear()
        is_speaking.set()
        avatar_send("thinking", False, "...")

        full_reply  = ""
        first_chunk = True

        for chunk in stream_llm(user_input):
            if stop_speaking.is_set(): break

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
            if speak_chunk(chunk): break

        is_speaking.clear()
        stop_speaking.clear()
        avatar_send(emotion.avatar, False, "")

        reply_text = full_reply.strip()
        if reply_text:
            # Extract any code blocks and send to panel
            fence_re = re.compile(r'```(\w*)\n([\s\S]*?)```')
            for lang, code in fence_re.findall(reply_text):
                code_send(code.strip(), lang or "bash", "")

            # Remove code blocks from memory so LLM context isn't polluted
            clean_history_text = re.sub(r'```[\s\S]*?(?:```|$)', '[code snippet]', reply_text).strip()
            if not clean_history_text:
                clean_history_text = "I have sent the code to the code panel."

            history.append(("user",      user_input))
            history.append(("assistant", clean_history_text))
            chat_send("siri", reply_text)
            update_mood(user_input, clean_history_text)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    check_ollama()
    threading.Thread(target=preload_ollama, daemon=True).start()

    print(f"""
+---------------------------------------------------+
|  {BOT_NAME} - Voice Assistant
|  Wake word : "Hey Siri"
|  Stop      : "Siri stop"
|  STT       : faster-whisper ({WHISPER_MODEL})
|  LLM       : {MODEL} via Ollama
|  TTS       : {VOICE} @ {VOICE_RATE}wpm
|  Dashboard : http://localhost:8080/dashboard.html
+---------------------------------------------------+
""")

    threading.Thread(target=start_ws_server, daemon=True).start()
    time.sleep(0.4)
    threading.Thread(target=mic_listener, daemon=True).start()
    time.sleep(0.5)

    try:
        ai_loop()
    except KeyboardInterrupt:
        print("\n\nBye!")
        sys.exit(0)

if __name__ == "__main__":
    main()