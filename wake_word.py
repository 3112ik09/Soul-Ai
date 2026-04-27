"""
wake_word.py — Hey Siri wake word + Siri stop
"""

WAKE_WORDS = [
    "hey siri",
    "hi siri",
    "ok siri",
    "okay siri",
    "siri",
]

STOP_WORDS = [
    "siri stop",
    "stop siri",
    "stop talking",
    "be quiet",
    "shut up",
    "that's enough",
    "stop",
]

def is_wake_word(text: str) -> bool:
    t = text.lower().strip()
    return any(t.startswith(w) for w in WAKE_WORDS)

import re

def is_stop_word(text: str) -> bool:
    t = text.lower().strip()
    for w in STOP_WORDS:
        if re.search(r'\b' + re.escape(w) + r'\b', t):
            return True
    return False

def strip_wake_word(text: str) -> str:
    """Remove wake word from start — returns just the command."""
    t = text.lower().strip()
    for w in sorted(WAKE_WORDS, key=len, reverse=True):
        if t.startswith(w):
            return text[len(w):].lstrip(" ,!?").strip()
    return text