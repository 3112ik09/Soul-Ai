"""
mac_tools.py — MCP-style tool registry for Mac system control
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fast path : direct keyword match → AppleScript/shell (no Ollama, <100ms)
Slow path : unrecognised commands → mac_agent.execute() (LLM-generated script)

Public API
  is_mac_command(text)  →  bool
  execute(text)         →  spoken_response (str)

Tools
  Spotify  : play / pause / next / previous / shuffle / search song
  Volume   : up / down / set N / mute / unmute
  Brightness: up / down
  Apps     : open / quit any app
  Screenshot: save to Desktop
  Calendar  : list today's events
"""

import re
import subprocess


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KEYWORD TRIGGERS  (fast pre-check before routing)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_TRIGGERS = [
    # Spotify
    "play ", "pause music", "stop music", "next song", "next track", "skip",
    "previous song", "previous track", "go back", "what's playing", "current song",
    "what song", "shuffle", "spotify",
    # Volume
    "volume up", "volume down", "louder", "quieter", "mute", "unmute",
    "set volume", "increase volume", "decrease volume", "raise volume", "lower volume",
    "turn it up", "turn it down",
    # Brightness
    "brightness up", "brightness down", "brighter", "dimmer", "dim the screen",
    "increase brightness", "decrease brightness",
    # Apps
    "open ", "launch ", "close ", "quit ",
    # Screenshot
    "screenshot", "take a screenshot",
    # Calendar
    "calendar", "my schedule", "what's today", "events today",
]

def is_mac_command(text: str) -> bool:
    t = text.lower()
    return any(trigger in t for trigger in _TRIGGERS)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOW-LEVEL RUNNERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _applescript(script: str) -> str:
    try:
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""

def _shell(cmd: str) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TOOLS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Spotify ─────────────────────────────────────
def _spotify(cmd: str) -> str:
    return _applescript(f'tell application "Spotify" to {cmd}')

def _spotify_current() -> str:
    track  = _spotify("get name of current track")
    artist = _spotify("get artist of current track")
    if track:
        return f"[happy] Playing {track} by {artist}."
    return "[neutral] Nothing's playing right now."

def _spotify_search_play(query: str) -> str:
    # Spotify URI search opens the app and plays
    uri = "spotify:search:" + query.replace(" ", "%20")
    _shell(f"open '{uri}'")
    return f"[happy] Searching for {query} on Spotify."


# ── Volume ──────────────────────────────────────
def _vol_up() -> str:
    _applescript("set volume output volume ((output volume of (get volume settings)) + 15)")
    return "[happy] Turned it up."

def _vol_down() -> str:
    _applescript("set volume output volume ((output volume of (get volume settings)) - 15)")
    return "[neutral] Turned it down."

def _vol_set(level: int) -> str:
    level = max(0, min(100, level))
    _applescript(f"set volume output volume {level}")
    return f"[neutral] Volume at {level}."

def _vol_mute() -> str:
    _applescript("set volume with output muted")
    return "[neutral] Muted."

def _vol_unmute() -> str:
    _applescript("set volume without output muted")
    return "[neutral] Unmuted."


# ── Brightness ──────────────────────────────────
# key code 113 = F2 (brightness up), 107 = F1 (brightness down) on most Macs
def _bright_up() -> str:
    _applescript('tell application "System Events" to key code 113')
    _applescript('tell application "System Events" to key code 113')
    return "[neutral] Brighter."

def _bright_down() -> str:
    _applescript('tell application "System Events" to key code 107')
    _applescript('tell application "System Events" to key code 107')
    return "[neutral] Dimmed."


# ── Apps ────────────────────────────────────────
def _open_app(name: str) -> str:
    out = _shell(f'open -a "{name.title()}" 2>&1')
    if "Unable" in out or "not found" in out.lower():
        return f"[confused] Can't find {name}, check the spelling."
    return f"[happy] Opened {name.title()}."

def _quit_app(name: str) -> str:
    _applescript(f'tell application "{name.title()}" to quit')
    return f"[neutral] Closed {name.title()}."


# ── Screenshot ──────────────────────────────────
def _screenshot() -> str:
    _shell("screencapture ~/Desktop/screenshot.png")
    return "[happy] Screenshot saved to Desktop."


# ── Calendar ────────────────────────────────────
def _calendar_today() -> str:
    script = """
    tell application "Calendar"
        set today_start to current date
        set hours of today_start to 0
        set minutes of today_start to 0
        set seconds of today_start to 0
        set today_end to today_start + (24 * 60 * 60)
        set evts to (every event of every calendar whose start date >= today_start and start date < today_end)
        set names_list to {}
        repeat with evt in evts
            if (count of evt) > 0 then
                set end of names_list to summary of item 1 of evt
            end if
        end repeat
        return names_list
    end tell
    """
    out = _applescript(script).strip()
    if not out:
        return "[neutral] Nothing on the calendar today."
    # AppleScript returns comma-separated items
    items = [x.strip() for x in out.split(",") if x.strip()][:4]
    return "[thinking] Today: " + ", ".join(items) + "."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ROUTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def execute(text: str) -> str:
    """Route a mac command. Returns a short spoken response with emotion tag."""
    t = text.lower().strip()

    # ── Spotify ─────────────────────────────────
    if any(w in t for w in ["what's playing", "what is playing", "current song", "what song"]):
        return _spotify_current()

    if any(w in t for w in ["pause music", "pause the music", "stop music", "pause spotify"]):
        _spotify("pause"); return "[neutral] Paused."

    if any(w in t for w in ["next song", "next track", "skip", "next one"]):
        _spotify("next track"); return "[happy] Next track."

    if any(w in t for w in ["previous song", "previous track", "go back", "prev song"]):
        _spotify("previous track"); return "[neutral] Going back."

    if any(w in t for w in ["shuffle", "shuffle mode", "shuffle on"]):
        _spotify("set shuffling to true"); return "[happy] Shuffle on."

    # "play X" with a song/artist qualifier → search
    if re.search(r'\bplay\b', t):
        # "play music" / "play spotify" / "resume" → just play
        if re.search(r'\b(music|spotify|resume|it)\b', t) and not re.search(
                r'\b(song|by|track|artist|playlist|band)\b', t):
            _spotify("play"); return "[happy] Playing."
        # Strip verb + qualifiers to get the search query
        query = re.sub(
            r"\b(play|song|by|track|artist|playlist|on spotify|for me)\b", "", t
        ).strip(" ,.?!")
        if len(query) > 1:
            return _spotify_search_play(query)
        _spotify("play"); return "[happy] Playing."

    # ── Volume ──────────────────────────────────
    if any(w in t for w in ["volume up", "louder", "turn it up", "increase volume",
                             "raise volume", "turn up"]):
        return _vol_up()

    if any(w in t for w in ["volume down", "quieter", "turn it down", "decrease volume",
                             "lower volume", "turn down"]):
        return _vol_down()

    if re.search(r'\bunmute\b', t):
        return _vol_unmute()

    if re.search(r'\bmute\b', t):
        return _vol_mute()

    m = re.search(r'(?:set volume|volume)\s+(?:to\s+)?(\d+)', t)
    if m:
        return _vol_set(int(m.group(1)))

    # ── Brightness ──────────────────────────────
    if any(w in t for w in ["brightness up", "brighter", "increase brightness",
                             "screen brighter"]):
        return _bright_up()

    if any(w in t for w in ["brightness down", "dimmer", "dim", "decrease brightness",
                             "screen dimmer"]):
        return _bright_down()

    # ── Screenshot ──────────────────────────────
    if "screenshot" in t:
        return _screenshot()

    # ── Calendar ────────────────────────────────
    if any(w in t for w in ["calendar", "my schedule", "what's today", "events today",
                             "what do i have today"]):
        return _calendar_today()

    # ── Open / quit app ─────────────────────────
    m = re.search(r'\b(?:open|launch)\b\s+(.+?)(?:\s+(?:for me|please|app))?$', t)
    if m:
        return _open_app(m.group(1).strip())

    m = re.search(r'\b(?:close|quit)\b\s+(.+?)(?:\s+(?:for me|please))?$', t)
    if m:
        return _quit_app(m.group(1).strip())

    # ── Fallback → LLM-based mac agent ──────────
    try:
        from mac_agent import execute as agent_execute
        return agent_execute(text)
    except Exception as e:
        return f"[confused] Couldn't figure that one out. {str(e)[:60]}"


# ── Quick self-test:  python mac_tools.py "volume up" ─────────────
if __name__ == "__main__":
    import sys
    cmd = " ".join(sys.argv[1:]) or "what's playing"
    print(f"Input    : {cmd}")
    print(f"Is mac?  : {is_mac_command(cmd)}")
    print(f"Response : {execute(cmd)}")
