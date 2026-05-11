"""
wake_word.py — Hey Siri wake word + Siri stop
"""

WAKE_WORDS = [
    "hey siri",
    "hi siri",
    "high siri",
    "ok siri",
    "okay siri",
    # Common Whisper mishearings of "Hey Siri"
    "high city",
    "hey series",
    "hi series",
    "a siri",
]

# Bare "siri" only triggers when it's the first word spoken
WAKE_WORD_START = "siri"

STOP_WORDS = [
    "siri stop",
    "stop siri",
    "stop talking",
    "be quiet",
    "shut up",
    "that's enough",
    "okay stop",
    "alright stop",
]

import re

def _normalize(text: str) -> str:
    """Lowercase and strip punctuation so 'Hi, Siri.' matches 'hi siri'."""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def is_wake_word(text: str) -> bool:
    t = _normalize(text)
    for w in WAKE_WORDS:
        if re.search(r'\b' + re.escape(w) + r'\b', t):
            return True
    # Bare "siri" only wakes if it's the first word
    if re.match(r'^' + re.escape(WAKE_WORD_START) + r'\b', t):
        return True
    return False

def is_stop_word(text: str) -> bool:
    t = _normalize(text)
    for w in STOP_WORDS:
        if re.search(r'\b' + re.escape(w) + r'\b', t):
            return True
    return False

def strip_wake_word(text: str) -> str:
    """Remove wake word and anything before it — returns just the command."""
    t = _normalize(text)
    all_wake = sorted(WAKE_WORDS, key=len, reverse=True) + [WAKE_WORD_START]
    for w in all_wake:
        match = re.search(r'\b' + re.escape(w) + r'\b', t)
        if match:
            return text[match.end():].lstrip(" ,!?").strip()
    return text