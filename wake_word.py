"""
wake_word.py — Wake word detection for Siri voice assistant
Listens for "hey siri" to activate and "siri stop" to stop

Install:
    pip install openwakeword pyaudio

Usage: imported by voice_chat.py
"""

import threading
import time
import numpy as np
import sounddevice as sd

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Wake word config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WAKE_WORD       = "hey siri"   # closest built-in to "hey siri"
STOP_WORD       = "siri stop"
SAMPLE_RATE     = 16000
CHUNK_MS        = 80             # openwakeword needs ~80ms chunks
MIC_DEVICE      = 1              # your Rockerz device


class WakeWordDetector:
    """
    Listens always-on for wake word.
    Calls on_wake() when detected.
    Calls on_stop() when stop word detected.
    """

    def __init__(self, on_wake=None, on_stop=None):
        self.on_wake    = on_wake
        self.on_stop    = on_stop
        self.active     = False     # True = currently in conversation
        self.running    = False
        self._thread    = None
        self._oww       = None

        # Load openWakeWord
        try:
            from openwakeword.model import Model
            self._oww = Model(
                wakeword_models=["hey_jarvis"],   # built-in model
                inference_framework="onnx"
            )
            print("✅ Wake word model loaded (hey_jarvis)")
        except Exception as e:
            print(f"⚠️  openWakeWord not available: {e}")
            print("   Falling back to volume-based activation")
            self._oww = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def set_active(self, val: bool):
        """Called by voice_chat when conversation starts/ends."""
        self.active = val

    def _loop(self):
        chunk_size = int(SAMPLE_RATE * CHUNK_MS / 1000)

        print("👂 Listening for 'Hey Siri'...")

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
            device=MIC_DEVICE
        ) as stream:
            while self.running:
                chunk, _ = stream.read(chunk_size)
                chunk    = chunk.flatten()

                if self._oww:
                    self._detect_oww(chunk)
                else:
                    self._detect_volume(chunk)

    def _detect_oww(self, chunk: np.ndarray):
        """Use openWakeWord neural model."""
        import torch
        # Convert to int16 for OWW
        audio_int16 = (chunk * 32768).astype(np.int16)

        try:
            prediction = self._oww.predict(audio_int16)

            for model_name, score in prediction.items():
                if score > 0.5 and not self.active:
                    print(f"\n🔔 Wake word detected! (score: {score:.2f})")
                    self.active = True
                    if self.on_wake:
                        self.on_wake()

        except Exception as e:
            pass

    def _detect_volume(self, chunk: np.ndarray):
        """
        Fallback: use Whisper to detect wake/stop phrases
        This is called from the existing VAD pipeline instead
        """
        pass

    def check_for_stop(self, transcript: str) -> bool:
        """
        Call this with every transcribed phrase.
        Returns True if stop phrase detected.
        """
        text = transcript.lower().strip()
        stop_phrases = [
            "siri stop", "stop siri", "stop talking",
            "be quiet", "shut up", "that's enough",
            "ok stop", "okay stop"
        ]
        for phrase in stop_phrases:
            if phrase in text:
                print(f"\n🛑 Stop phrase detected: '{phrase}'")
                self.active = False
                if self.on_stop:
                    self.on_stop()
                return True
        return False

    def check_for_wake(self, transcript: str) -> bool:
        """
        Whisper-based wake word detection as fallback.
        Call this with every transcribed phrase.
        Returns True if wake phrase detected.
        """
        text = transcript.lower().strip()
        wake_phrases = [
            "hey siri", "ok siri", "hey series",   # common mishears
            "hi siri", "hello siri", "siri"
        ]
        for phrase in wake_phrases:
            if text.startswith(phrase) or text == phrase:
                print(f"\n🔔 Wake phrase detected: '{transcript}'")
                self.active = True
                if self.on_wake:
                    self.on_wake()
                return True
        return False