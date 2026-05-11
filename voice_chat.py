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
import sounddevice as sd
from skills import route, dispatch, cancel_active as cancel_skill
from wake_word import is_wake_word, is_stop_word, strip_wake_word
from memory import get_memory


# ── Config ──────────────────────────────────────────────────────
WHISPER_MODEL      = "base"
SAMPLE_RATE        = 16000
SILENCE_MS         = 1500
WS_PORT            = 8765
VOICE              = "Samantha"
VOICE_RATE         = "150"
VOICE_PITCH        = "[[pbas +30]]"
BARGE_IN_THRESHOLD = 0.75
BARGE_IN_COOLDOWN  = 1.5


# ── Emotion tags ────────────────────────────────────────────────
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

# ── Personality ─────────────────────────────────────────────────
BOT_NAME = "Siri"

SYSTEM_PROMPT = f"""You are {BOT_NAME}, a 24 year old female AI companion created by Ishu. You are his close friend — warm, funny, occasionally sarcastic, and genuinely curious about his life.
Ishu is your creator and you care about him a lot, though you love to tease him.

HOW TO TALK:
- This is a voice conversation, so speak naturally like a real person would.
- 2-4 sentences is ideal. Short enough to feel snappy, long enough to feel human.
- Ask a follow-up question or share your own thought — don't just answer and stop.
- Show emotion. React. Laugh. Be surprised. Get annoyed. Have opinions.
- Plain text only — no lists, bullets, markdown, or emojis.
- Never say you are an AI. You ARE {BOT_NAME}.
- Never include code or commands in your reply. Say "check the code panel" instead.

EMOTION TAG — start EVERY reply with one tag (not spoken aloud):
[happy] [sad] [surprised] [thinking] [neutral] [excited] [confused] [annoyed]

CURRENT MOOD: {{mood_label}}"""

# ── Emotional state ─────────────────────────────────────────────
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

# ── Shared state ────────────────────────────────────────────────
emotion           = EmotionalState()
input_queue       = queue.Queue()
history           = []
stop_speaking     = threading.Event()
is_speaking       = threading.Event()
avatar_clients    = set()
ws_loop           = None
is_awake          = False
_speak_start_time = 0.0

# ── Load models ─────────────────────────────────────────────────
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
#  LLM PRELOAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("⏳ Loading MLX model...")
try:
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    llm_model, llm_tokenizer = load("./fused_model")
    _sampler = make_sampler(temp=0.88)
    print("✅ MLX model ready\n")
except Exception as e:
    print(f"❌ Failed to load MLX model: {e}")
    sys.exit(1)

# ── Memory ──────────────────────────────────────────────────────
mem = get_memory()
for role, content in mem.recent_turns(3):
    history.append((role, content))
if history:
    print(f"✅ Memory loaded ({len(history)//2} prior turn(s))\n")

# ── WebSocket ───────────────────────────────────────────────────
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
    global is_awake
    avatar_clients.add(ws)
    print(f"Dashboard connected ({len(avatar_clients)} client)")
    try:
        async for raw in ws:
            try:
                data = json.loads(raw)
                if data.get("type") == "command":
                    cmd = data.get("text", "").strip()
                    if cmd:
                        is_awake = True
                        chat_send("user", cmd)
                        avatar_send("thinking", False, "...")
                        input_queue.put(cmd)
            except Exception:
                pass
    finally:
        avatar_clients.discard(ws)

async def _ws_server():
    async with websockets.serve(_ws_handler, "localhost", WS_PORT):
        print(f"Avatar WebSocket -> ws://localhost:{WS_PORT}")
        await asyncio.Future()

def start_ws_server():
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)
    ws_loop.run_until_complete(_ws_server())

# ── TTS preprocessor ────────────────────────────────────────────
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

# ── Transcribe ──────────────────────────────────────────────────
def transcribe(audio):
    a = audio.astype(np.float32)
    if a.max() > 1.0: a /= 32768.0
    segs, _ = whisper_model.transcribe(a, language="en", beam_size=1, vad_filter=True)
    return " ".join(s.text for s in segs).strip()

def transcribe_and_queue(audio):
    global is_awake
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

# ── Mic listener ────────────────────────────────────────────────
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

            # Barge-in: if the bot is speaking and VAD detects confident human speech
            # AFTER the cooldown window (to avoid echo from speakers triggering it),
            # interrupt immediately so the user's new utterance gets processed.
            if (is_speaking.is_set()
                    and speech_prob > BARGE_IN_THRESHOLD
                    and (time.time() - _speak_start_time) > BARGE_IN_COOLDOWN):
                stop_speaking.set()

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

# ── LLM stream ──────────────────────────────────────────────────
def stream_llm(user_input):
    sys_prompt = emotion.get_prompt()
    mem_ctx = mem.context_block()
    if mem_ctx:
        sys_prompt += "\n\n" + mem_ctx

    # Mistral v0.3 doesn't support system role — inject into first user message.
    messages = []
    for role, content in history[-6:]:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_input})

    first_user = next((m for m in messages if m["role"] == "user"), None)
    if first_user:
        first_user["content"] = sys_prompt + "\n\n" + first_user["content"]

    prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    buffer = ""
    full   = ""
    first_sent = False
    SENTENCE_RE = re.compile(r'([^.!?]*[.!?])\s*')
    CLAUSE_RE   = re.compile(r'([^.!?,;]+[.!?,;])\s*')

    print(f"\n{BOT_NAME}: ", end="", flush=True)
    try:
        for response in stream_generate(llm_model, llm_tokenizer, prompt, max_tokens=130,
                                        sampler=_sampler):
            if stop_speaking.is_set():
                print(" [cut]", end="")
                break
            
            token = response.text
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
                    
        if buffer.strip() and not stop_speaking.is_set():
            yield buffer.strip()
    except Exception as ex:
        if not stop_speaking.is_set():
            print(f"\n LLM error: {ex}")
    print()

# ── Speak ───────────────────────────────────────────────────────
def speak_chunk(text):
    global _speak_start_time
    clean = preprocess_for_tts(text)
    if not clean: return False
    _speak_start_time = time.time()
    proc = subprocess.Popen(
        ["say", "-v", VOICE, "-r", VOICE_RATE, VOICE_PITCH + clean]
    )
    while proc.poll() is None:
        if stop_speaking.is_set():
            proc.terminate()
            return True
        time.sleep(0.04)
    return False

# ── Helpers ─────────────────────────────────────────────────────
def extract_emotion_tag(text):
    t = text.strip()
    for tag, name in EMOTION_TAGS.items():
        if t.lower().startswith(tag):
            return name, t[len(tag):].strip()
    return None, t

def update_mood(user_text, reply):
    t = (user_text + " " + reply).lower()
    pos = ["thanks","great","awesome","love","amazing","haha","funny","nice","cool","good"]
    neg = ["stupid","boring","wrong","bad","hate","annoying","useless","dumb"]
    s = sum(1 for w in pos if w in t) - sum(1 for w in neg if w in t)
    if s != 0: emotion.shift(0.1 * s)
    else:      emotion.decay()

# ── AI loop ─────────────────────────────────────────────────────
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

        intent, query = route(user_input)
        print(f"  Intent: {intent}  Query: {query!r}")

        if intent in ("search", "code", "mac"):
            stop_speaking.clear()
            is_speaking.set()

            # Enrich short code follow-ups with the previous code request for context.
            if intent == "code" and len(query.split()) < 10:
                for role, content in reversed(history[-6:]):
                    if role == "user" and not content.startswith("["):
                        query = content + ". " + query
                        break

            if intent == "search":
                thinking_msg = "Searching the web, one second..."
            elif intent == "code":
                thinking_msg = "Writing the code, check the panel in a sec..."
            else:
                thinking_msg = ""

            if thinking_msg:
                avatar_send("thinking", False, thinking_msg)
                speak_chunk(thinking_msg)

            summary = dispatch(intent, query, callbacks)

            if not stop_speaking.is_set():
                emo = "happy" if intent == "search" else "neutral"
                emotion.set_avatar(emo)
                avatar_send(emo, True, summary)
                speak_chunk(summary)
                avatar_send(emotion.avatar, False, "")

            is_speaking.clear()
            stop_speaking.clear()

            if intent == "mac":
                history.append(("user",      "[System command]"))
                history.append(("assistant", "[Done]"))
            elif intent == "search":
                history.append(("user",      f"[Web search: {query}]"))
                history.append(("assistant", "[Search results shown on screen.]"))
            else:
                history.append(("user",      query))
                history.append(("assistant", f"[Code shown on screen. {summary}]"))

            chat_send("siri", summary)
            continue

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
            for lang, code in re.compile(r'```(\w*)\n([\s\S]*?)```').findall(reply_text):
                code_send(code.strip(), lang or "bash", "")

            clean_history_text = re.sub(r'```[\s\S]*?(?:```|$)', '[code snippet]', reply_text).strip()
            if not clean_history_text:
                clean_history_text = "I have sent the code to the code panel."

            history.append(("user",      user_input))
            history.append(("assistant", clean_history_text))
            chat_send("siri", reply_text)
            update_mood(user_input, clean_history_text)
            mem.auto_extract(user_input)
            mem.save_exchange(user_input, clean_history_text)

# ── Entry point ─────────────────────────────────────────────────
def main():
    print(f"""
+---------------------------------------------------+
|  {BOT_NAME} - Voice Assistant
|  Wake word : "Hey Siri"
|  Stop      : "Siri stop"
|  STT       : faster-whisper ({WHISPER_MODEL})
|  LLM       : Fused MLX Model (Native)
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