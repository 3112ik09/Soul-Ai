import re
import json
import html
import requests
from urllib.parse import unquote
from html.parser import HTMLParser

OLLAMA_URL      = "http://localhost:11434"
CODE_MODEL      = "qwen2.5-coder:3b"
SEARCH_N        = 6
_active_session = None

def cancel_active():
    """Called from voice_chat when user interrupts mid-action."""
    global _active_session
    if _active_session is not None:
        try: _active_session.close()
        except: pass
        _active_session = None


# Strategy: keyword scoring, not strict regex.
# We score each intent based on trigger words present anywhere in the
# sentence. Highest score wins. This handles polite/garbled phrasing
# like "Siri could you please search the NBA score on web" or
# "I want python code such that I can copy files".

# Strong triggers — single word presence is usually enough
_SEARCH_KEYWORDS = {
    # verb triggers
    "search": 1, "google": 2, "lookup": 2, "browse": 1,
    # context triggers (need pairing with a verb-ish word, see logic)
    "web": 1, "internet": 1, "online": 1,
    # common search intents
    "latest": 1, "news": 1, "score": 1, "weather": 1,
}
_SEARCH_PHRASES = [
    "look up", "find out", "find me", "search for", "search up",
    "on the web", "on the internet", "on google",
    "what is the latest", "what's the latest",
    "any news on", "any news about",
]

_CODE_KEYWORDS = {
    "code": 1, "script": 1, "program": 1, "function": 1, "snippet": 1,
    "python": 1, "javascript": 1, "bash": 1, "shell": 1,
    "typescript": 1, "rust": 1, "golang": 1,
    "implement": 1, "algorithm": 1,
}
_CODE_PHRASES = [
    "write code", "write a code", "write me code", "write a script",
    "write me a script", "write a function", "write a program",
    "give me code", "give me a script", "show me code",
    "generate code", "create a script", "create a function",
    "how do i code", "how do i write", "how do i implement",
    "python code", "python script", "javascript code", "js code",
    "bash script", "shell script", "write a python code",
    "write some code", "can you write a python", "can you code",
    "could you write a python code", "write a python program",
    # natural phrasings
    "i need code", "i need a code", "i need a script", "i need a function",
    "i need a program", "get me code", "make me code", "make a script",
    "make a function", "code for me", "code to", "script to", "script for",
    "need help coding", "help me code", "help me write",
]

# Phrases that should HARD OVERRIDE — if present, intent is locked
_HARD_SEARCH = ["search the web", "search on web", "search on the web",
                "google it", "google that", "search online"]
_HARD_CODE   = ["write code", "write a code", "write a script",
                "write me code", "write me a script", "write the code"]

# Negative signals — words that suggest pure chat even if a keyword is present
_CHAT_OVERRIDES = ["how are you", "tell me about yourself",
                   "what do you think", "i feel", "i love", "i hate"]

# Mac system operation scoring — weighted phrases/keywords
# Using scores instead of binary match so compound commands
# (e.g. "open safari and search for news") route to the dominant intent.
_MAC_SCORE_TABLE: list[tuple[str, int]] = [
    # Spotify — high confidence
    ("pause music", 5), ("stop music", 5), ("next song", 5), ("next track", 5),
    ("previous song", 5), ("previous track", 5), ("what's playing", 5), ("what is playing", 5),
    ("current song", 5), ("play music", 5), ("play spotify", 5),
    ("shuffle", 4), ("skip song", 4), ("skip track", 4),
    # Volume
    ("volume up", 5), ("volume down", 5), ("set volume", 5),
    ("increase volume", 5), ("decrease volume", 5),
    ("louder", 4), ("quieter", 4), ("mute", 4), ("unmute", 4),
    # Brightness
    ("brightness up", 5), ("brightness down", 5),
    ("brighter", 4), ("dimmer", 4),
    # Screenshot / calendar
    ("screenshot", 5), ("take a screenshot", 5),
    ("my schedule", 5), ("events today", 5),
    # Apps (lower score — "open" appears in many non-mac sentences)
    ("open ", 3), ("launch ", 3), ("quit ", 3), ("close ", 2),
    # Spotify keyword alone (medium — "spotify news" should still go to search)
    ("spotify", 3),
]


def _score_mac(low: str) -> int:
    score = 0
    for phrase, weight in _MAC_SCORE_TABLE:
        if phrase in low:
            score += weight
    # "play X" where X is not generic music → likely Spotify search
    if re.search(r"\bplay\b", low) and not re.search(
            r"\b(music|it|resume|spotify)\b", low):
        score += 4
    return score


def _score_intent(low: str, keywords: dict, phrases: list) -> int:
    score = 0
    # Word-boundary matches for keywords
    for kw, weight in keywords.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", low):
            score += weight
    # Phrase matches
    for ph in phrases:
        if ph in low:
            score += 4
    return score


def route(text: str) -> tuple[str, str]:
    """
    Returns (intent, query). intent in {'search', 'code', 'mac', 'chat'}.
    Uses keyword scoring across the whole sentence, not start-of-string regex.
    Mac is scored like other intents — highest score wins — so compound
    sentences like "open safari and search for news" route to search.
    """
    t = text.strip()
    low = t.lower()

    # Hard overrides — unambiguous explicit phrases
    for ph in _HARD_SEARCH:
        if ph in low:
            q = _extract_search_query(low, t)
            return "search", q
    for ph in _HARD_CODE:
        if ph in low:
            return "code", t

    # Chat overrides — pure conversation, never route to skill
    for ph in _CHAT_OVERRIDES:
        if ph in low:
            return "chat", t

    # Score all intents — highest unambiguous winner takes it
    THRESHOLD   = 3
    mac_score    = _score_mac(low)
    search_score = _score_intent(low, _SEARCH_KEYWORDS, _SEARCH_PHRASES)
    code_score   = _score_intent(low, _CODE_KEYWORDS,   _CODE_PHRASES)

    best = max(mac_score, search_score, code_score)

    if best < THRESHOLD:
        return "chat", t

    if mac_score == best and mac_score > search_score and mac_score > code_score:
        return "mac", t

    if search_score == best and search_score >= code_score:
        q = _extract_search_query(low, t)
        return "search", q

    if code_score >= THRESHOLD:
        return "code", t

    return "chat", t


def _extract_search_query(low: str, original: str) -> str:
    """
    Try to pull out just the topic from a search request.
    Falls back to the full sentence if extraction is unclear.
    """
    # Common patterns to strip
    cleanups = [
        # remove polite preambles
        r"^(?:hey |hi |yo |siri,? )?(?:could |can |will |would )?you (?:please )?",
        r"^(?:please |kindly )",
        r"^(?:i want (?:you )?to |i need (?:you )?to |i'd like (?:you )?to )",
        # remove search verb phrases
        r"\b(?:search (?:for |up )?|google |look up |find (?:out |me )?|browse (?:for )?)\b",
        # remove trailing context
        r"\s+on (?:the )?(?:web|internet|google|online)\s*\??\.?\s*$",
        r"\s+for me\s*\??\.?\s*$",
    ]
    q = low
    for pat in cleanups:
        q = re.sub(pat, " ", q)
    q = re.sub(r"\s+", " ", q).strip(" ?.!,")
    # If we stripped too much, fall back to original
    if len(q) < 3:
        return original.strip(" ?.!,")
    return q


def dispatch(intent: str, query: str, callbacks: dict) -> str:
    """
    Run the action. Returns a SHORT (1 sentence) spoken summary
    that voice_chat.py will pass to TTS.

    callbacks = {
        'code_send':   fn(code, lang, explanation),
        'output_send': fn(html_string, format='html'),
    }
    """
    try:
        if intent == "search":
            results = web_search(query, n=SEARCH_N)
            if not results:
                callbacks['output_send'](
                    f"<div style='padding:14px;color:#888'>No results for "
                    f"<b>{html.escape(query)}</b></div>", "html"
                )
                return f"Hmm, nothing came back for that one."
            callbacks['output_send'](format_search_html(query, results), "html")
            return f"Got {len(results)} results for {query}, up on your screen."

        if intent == "code":
            code, lang, explain = generate_code(query)
            callbacks['code_send'](code, lang, explain)
            return explain or "Done. Code is on your screen."

        if intent == "mac":
            from mac_tools import execute as mac_execute
            return mac_execute(query)

    except Exception as e:
        return f"That didn't work. {str(e)[:80]}"

    return "Hmm, not sure what to do with that."


# DDG's html.duckduckgo.com endpoint returns real organic results
# without an API key. Format: <a class="result__a" href="...">title</a>
# followed by <a class="result__snippet">snippet</a>.

class _DDGParser(HTMLParser):
    """Tiny HTML parser that pulls out (title, url, snippet) triples."""
    def __init__(self):
        super().__init__()
        self.results = []
        self._cur = {}
        self._capture = None  # 'title' | 'snippet' | None

    def handle_starttag(self, tag, attrs):
        ad = dict(attrs)
        cls = ad.get("class", "")
        if tag == "a" and "result__a" in cls:
            self._cur = {"url": _clean_ddg_url(ad.get("href", "")), "title": "", "snippet": ""}
            self._capture = "title"
        elif tag == "a" and "result__snippet" in cls:
            self._capture = "snippet"

    def handle_endtag(self, tag):
        if tag == "a" and self._capture == "snippet" and self._cur:
            self.results.append(self._cur)
            self._cur = {}
            self._capture = None
        elif tag == "a" and self._capture == "title":
            self._capture = None  # title done, wait for snippet

    def handle_data(self, data):
        if self._capture == "title" and self._cur:
            self._cur["title"] += data
        elif self._capture == "snippet" and self._cur:
            self._cur["snippet"] += data


def _clean_ddg_url(href: str) -> str:
    """DDG wraps real URLs in /l/?uddg=<encoded>. Unwrap it."""
    if not href: return ""
    m = re.search(r"uddg=([^&]+)", href)
    if m:
        return unquote(m.group(1))
    if href.startswith("//"):
        return "https:" + href
    return href


def web_search(query: str, n: int = 6) -> list[dict]:
    """Returns list of {title, url, snippet} dicts."""
    global _active_session
    _active_session = requests.Session()
    try:
        r = _active_session.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=10,
        )
        r.raise_for_status()
        parser = _DDGParser()
        parser.feed(r.text)
        out = []
        for res in parser.results[:n]:
            title = re.sub(r"\s+", " ", res["title"]).strip()
            snip  = re.sub(r"\s+", " ", res["snippet"]).strip()
            url   = res["url"].strip()
            if title and url:
                out.append({"title": title, "url": url, "snippet": snip})
        return out
    finally:
        try: _active_session.close()
        except: pass
        _active_session = None


def format_search_html(query: str, results: list[dict]) -> str:
    """HTML to inject into the dashboard's #web-results div."""
    parts = [
        f'<div style="padding:6px 0 12px;font-size:11px;'
        f'color:#888;letter-spacing:.08em;text-transform:uppercase">'
        f'Results for "{html.escape(query)}"</div>'
    ]
    for r in results:
        parts.append(
            f'<div class="web-result">'
            f'  <a class="web-result-title" href="{html.escape(r["url"])}" '
            f'     target="_blank" rel="noopener">{html.escape(r["title"])}</a>'
            f'  <div class="web-result-url">{html.escape(r["url"][:80])}</div>'
            f'  <div class="web-result-snippet">{html.escape(r["snippet"])}</div>'
            f'</div>'
        )
    return "".join(parts)


_CODE_SYSTEM = """You are a code generator. Given a user request, respond with ONLY a JSON object:
{"language": "<lang>", "code": "<code>", "explanation": "<one short sentence>"}

Rules:
- Default to Python unless the user specifies another language.
- language: lowercase, one of: python, javascript, bash, typescript, go, rust, html, sql, java, cpp
- code: complete, runnable, well-commented. No markdown fences inside the JSON string.
- explanation: ONE sentence, max 15 words, casual tone. This will be spoken aloud.
- Output ONLY the JSON object. No preamble. No markdown. No code fences around the JSON."""


def generate_code(query: str) -> tuple[str, str, str]:
    """Returns (code, language, spoken_explanation)."""
    global _active_session
    _active_session = requests.Session()

    payload = {
        "model": CODE_MODEL,
        "messages": [
            {"role": "system", "content": _CODE_SYSTEM},
            {"role": "user",   "content": query},
        ],
        "stream": False,
        "format": "json",         # forces valid JSON output
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096,
            "num_predict": 1500,
        },
    }

    try:
        r = _active_session.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload, timeout=90,
        )
        r.raise_for_status()
        raw = r.json().get("message", {}).get("content", "")
    finally:
        try: _active_session.close()
        except: pass
        _active_session = None

    # Parse the JSON response
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to pull JSON out of mixed output
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError("Coder model returned non-JSON")
        parsed = json.loads(m.group(0))

    code  = parsed.get("code", "").strip()
    lang  = parsed.get("language", "python").strip().lower()
    expl  = parsed.get("explanation", "Code is ready.").strip()

    # Strip accidental markdown fences inside the code field
    code = re.sub(r"^```\w*\n?", "", code)
    code = re.sub(r"\n?```$", "", code)

    return code, lang, expl


if __name__ == "__main__":
    import sys
    test = " ".join(sys.argv[1:]) or "search for fastapi tutorial"
    intent, q = route(test)
    print(f"Input    : {test}")
    print(f"Intent   : {intent}")
    print(f"Query    : {q}")
    if intent == "search":
        results = web_search(q, n=3)
        for i, res in enumerate(results, 1):
            print(f"\n[{i}] {res['title']}\n    {res['url']}\n    {res['snippet'][:100]}...")
    elif intent == "code":
        code, lang, expl = generate_code(q)
        print(f"\nLang     : {lang}\nExplain  : {expl}\n--- CODE ---\n{code}")