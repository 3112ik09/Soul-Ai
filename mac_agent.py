"""
mac_agent.py — Smart Mac control agent
Instead of keyword matching, sends the command to the LLM
which generates the exact AppleScript to run.
Handles: Spotify, Calendar, Screenshots, Volume, Apps, Files, Reminders
"""

import subprocess
import requests
import json
import re
import os

OLLAMA_URL = "http://localhost:11434"
MODEL      = "mistral"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AGENT SYSTEM PROMPT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGENT_PROMPT = """You are a macOS automation agent. 
The user will give you a command and you must respond with ONLY a JSON object.

JSON format:
{
  "action": "applescript" | "shell" | "none",
  "script": "the applescript or shell command to run",
  "spoken_response": "what Siri should say after doing it (short, 1 sentence)"
}

CAPABILITIES:
- Spotify: play/pause/next/previous/shuffle/search songs/playlists/artists
- Calendar: read today's events, add new events
- Reminders: add reminders, read reminders
- Screenshots: take screenshot, save to desktop
- Volume: set/increase/decrease Mac system volume
- Apps: open/close/switch any Mac application
- Files: open folders, find files, move files
- Brightness: increase/decrease screen brightness
- Web: open URLs in browser

APPLESCRIPT EXAMPLES:
- Spotify next: tell application "Spotify" to next track
- Spotify play song: tell application "Spotify" to play track "spotify:search:songname"  
- Calendar today: tell application "Calendar" to get summary of events of today
- Volume up: set volume output volume ((output volume of (get volume settings)) + 15)
- Open app: tell application "Safari" to activate
- Screenshot: do shell script "screencapture ~/Desktop/screenshot.png"
- Reminder: tell application "Reminders" to make new reminder with properties {name:"task", due date:current date}

Return ONLY the JSON. No explanation. No markdown. Just raw JSON.
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KEYWORD DETECTION — is this a Mac command?
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAC_KEYWORDS = [
    # Spotify
    "play", "pause", "next song", "skip", "previous", "shuffle",
    "stop music", "what's playing", "current song", "spotify",
    # Calendar
    "calendar", "schedule", "what's today", "my meetings",
    "add event", "add meeting", "what do i have",
    # System
    "screenshot", "take a screenshot", "volume up", "volume down",
    "mute", "unmute", "set volume", "louder", "quieter",
    "open ", "close ", "quit ", "lock screen", "brightness",
    # Reminders
    "remind me", "set a reminder", "add a reminder",
    # Files
    "open folder", "open downloads", "open documents",
    "find file", "show me",
]

def is_mac_command(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in MAC_KEYWORDS)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RUN APPLESCRIPT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_applescript(script: str) -> str:
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        return f"Error: {result.stderr.strip()}"
    return result.stdout.strip()

def run_shell(cmd: str) -> str:
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=10
    )
    return result.stdout.strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LLM AGENT — generates the right script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask_agent(user_command: str) -> dict:
    """Ask LLM what AppleScript to run for this command."""
    payload = {
        "model":    MODEL,
        "messages": [
            {"role": "system", "content": AGENT_PROMPT},
            {"role": "user",   "content": user_command}
        ],
        "stream":  False,
        "options": {"temperature": 0.1, "num_predict": 200}
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=30)
        raw = r.json()["message"]["content"].strip()

        # Strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()

        return json.loads(raw)
    except Exception as e:
        return {
            "action": "none",
            "script": "",
            "spoken_response": f"Sorry, I couldn't figure out how to do that. {e}"
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN EXECUTE FUNCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def execute(user_command: str) -> str:
    """
    Main entry point.
    1. Ask LLM what to do
    2. Run the script
    3. Return spoken response
    """
    print(f"  🤖 Mac agent: {user_command}")

    result = ask_agent(user_command)

    action   = result.get("action", "none")
    script   = result.get("script", "")
    response = result.get("spoken_response", "Done.")

    print(f"  ⚡ Action: {action}")
    print(f"  📜 Script: {script[:80]}...")

    if action == "applescript" and script:
        output = run_applescript(script)
        # If the script returns useful data, include it in response
        if output and "Error" not in output and len(output) < 200:
            # Replace placeholder in response
            response = response.replace("{output}", output)
            if "{output}" not in result.get("spoken_response",""):
                # Append output if meaningful
                if output and output != "":
                    response = f"{response} {output}"
        elif "Error" in output:
            response = f"Something went wrong. {output}"

    elif action == "shell" and script:
        output = run_shell(script)
        if output and "Error" not in output:
            response = response.replace("{output}", output)

    print(f"  💬 Response: {response}")
    return response.strip()