"""
memory.py — Persistent memory for Siri across sessions.

Facts    : key/value store (user name, preferences, etc.) — auto-extracted + manual
History  : last N conversation turns saved to SQLite, loaded on startup
"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock

DB_PATH = Path("./siri_memory.db")


class Memory:
    def __init__(self, db_path: Path = DB_PATH):
        self._lock = Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                ts    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS history (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                role    TEXT NOT NULL,
                content TEXT NOT NULL,
                ts      TEXT NOT NULL
            );
        """)
        self._conn.commit()

    # ── Facts ───────────────────────────────────────────────────
    def set_fact(self, key: str, value: str):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO facts (key, value, ts) VALUES (?, ?, ?)",
                (key.lower().strip(), value.strip(), datetime.now().isoformat()),
            )
            self._conn.commit()

    def get_fact(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM facts WHERE key = ?", (key.lower().strip(),)
        ).fetchone()
        return row["value"] if row else None

    def all_facts(self) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT key, value FROM facts ORDER BY ts DESC"
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def delete_fact(self, key: str):
        with self._lock:
            self._conn.execute("DELETE FROM facts WHERE key = ?", (key.lower().strip(),))
            self._conn.commit()

    # ── Conversation history ─────────────────────────────────────
    def save_exchange(self, user: str, assistant: str):
        now = datetime.now().isoformat()
        with self._lock:
            self._conn.executemany(
                "INSERT INTO history (role, content, ts) VALUES (?, ?, ?)",
                [("user", user, now), ("assistant", assistant, now)],
            )
            self._conn.commit()
        self._trim_history()

    def _trim_history(self, keep: int = 200):
        with self._lock:
            self._conn.execute("""
                DELETE FROM history WHERE id NOT IN (
                    SELECT id FROM history ORDER BY id DESC LIMIT ?
                )
            """, (keep,))
            self._conn.commit()

    def recent_turns(self, n: int = 3) -> list[tuple[str, str]]:
        """Returns last n (role, content) pairs, oldest first."""
        rows = self._conn.execute(
            "SELECT role, content FROM history ORDER BY id DESC LIMIT ?",
            (n * 2,),
        ).fetchall()
        return [(r["role"], r["content"]) for r in reversed(rows)]

    # ── System-prompt context block ──────────────────────────────
    def context_block(self) -> str:
        facts = self.all_facts()
        if not facts:
            return ""
        lines = [f"  {k}: {v}" for k, v in list(facts.items())[:10]]
        return "THINGS YOU KNOW ABOUT THE USER:\n" + "\n".join(lines)

    # ── Auto-extract facts from user messages ───────────────────
    _PATTERNS: list[tuple[re.Pattern, str]] = [
        (re.compile(r"\bmy name is ([A-Za-z]+)\b", re.I),                         "user_name"),
        (re.compile(r"\bpeople call me ([A-Za-z]+)\b", re.I),                     "user_name"),
        (re.compile(r"\bi(?:'m| am) from ([A-Za-z ,]+?)(?:\.|,|$)", re.I),        "user_location"),
        (re.compile(r"\bi work (?:as|at) ([^\.\,]{3,40}?)(?:\.|,|$)", re.I),     "user_work"),
        (re.compile(r"\bi(?:'m| am) a[n]? ([\w ]{3,30}?)(?:\.|,|$)", re.I),      "user_role"),
        (re.compile(r"\bmy (?:fav(?:ou?rite)?) (?:song|artist|band) is ([^\.,]{2,40})", re.I), "fav_music"),
        (re.compile(r"\bmy (?:fav(?:ou?rite)?) (?:movie|show) is ([^\.,]{2,40})", re.I),       "fav_media"),
        (re.compile(r"\bi (?:like|love|enjoy) ([^\.,]{3,40}?)(?:\.|,|and|$)", re.I),           "user_likes"),
    ]

    def auto_extract(self, user_text: str):
        for pattern, key in self._PATTERNS:
            m = pattern.search(user_text)
            if m:
                val = m.group(1).strip(" .,?!")
                if 2 < len(val) < 50:
                    # Don't overwrite a concrete name with something vague
                    if key == "user_name" and len(val.split()) > 2:
                        continue
                    self.set_fact(key, val)


# ── Singleton ────────────────────────────────────────────────────
_instance: Memory | None = None

def get_memory() -> Memory:
    global _instance
    if _instance is None:
        _instance = Memory()
    return _instance
