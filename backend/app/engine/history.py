"""Conversation history — privacy by design.

* OFF by default. Nothing is written unless the user turns "Save history" on.
* Text only. Audio is never stored anywhere (it lives in memory for a few seconds).
* Local only: a SQLite file in the user's data folder; "Clear history" deletes it all.
Schema mirrors the future server tables (sessions/messages) so syncing can be added later.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from typing import Any

from app.core.config import data_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_started REAL NOT NULL,
    speaker TEXT NOT NULL CHECK (speaker IN ('me', 'other')),
    source_language TEXT NOT NULL,
    source_text TEXT NOT NULL,
    target_language TEXT NOT NULL,
    translated_text TEXT NOT NULL,
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages (timestamp);
"""


class HistoryStore:
    def __init__(self, enabled: bool = False, path=None):
        self.enabled = enabled
        self.path = path or (data_dir() / "history.db")
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._session_started = time.time()

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.executescript(SCHEMA)
        return self._conn

    def add(self, message: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            db = self._db()
            db.execute(
                "INSERT INTO messages (session_started, speaker, source_language, source_text, "
                "target_language, translated_text, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self._session_started, message["speaker"], message["source_language"], message["source_text"],
                 message["target_language"], message["translated_text"], message.get("timestamp", time.time())))
            db.commit()

    def recent(self, limit: int = 200) -> list[dict]:
        if not self.path.exists():
            return []
        with self._lock:
            rows = self._db().execute(
                "SELECT id, speaker, source_language, source_text, target_language, translated_text, timestamp "
                "FROM messages ORDER BY id DESC LIMIT ?", (max(1, min(limit, 5000)),)).fetchall()
        keys = ("id", "speaker", "source_language", "source_text", "target_language", "translated_text", "timestamp")
        return [dict(zip(keys, r)) for r in reversed(rows)]

    def clear(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            self.path.unlink(missing_ok=True)
