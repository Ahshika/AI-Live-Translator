"""Catch the meeting / our own translation leaking into the microphone (speakers, no headset).

Without this, the loop is: they speak -> you hear the Arabic translation on your speakers ->
your mic hears it -> it is "your" sentence -> translated back and sent to the meeting, on top
of everybody. The audio can't tell us it's an echo, but the words can: an echo repeats, almost
word for word, something that was played a few seconds ago.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from difflib import SequenceMatcher

_DIACRITICS = re.compile(r"[ً-ٰٟـ]")  # Arabic tashkeel + tatweel
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    text = _DIACRITICS.sub("", text.lower())
    text = text.translate(str.maketrans("أإآىة", "اااىه"))
    return " ".join(_NON_WORD.sub(" ", text).split())


class EchoFilter:
    WINDOW_S = 20.0
    MIN_WORDS = 3  # "ok", "ja genau" are said by real people all the time
    THRESHOLD = 0.6

    def __init__(self):
        self._recent: deque[tuple[float, str]] = deque(maxlen=40)
        self._lock = threading.Lock()
        self.hits = 0

    def heard(self, *texts: str) -> None:
        """Something that was played to the user's room (original or translation)."""
        now = time.monotonic()
        with self._lock:
            for t in texts:
                n = normalize(t or "")
                if len(n.split()) >= self.MIN_WORDS:
                    self._recent.append((now, n))

    def is_echo(self, text: str) -> bool:
        n = normalize(text)
        words = set(n.split())
        if len(words) < self.MIN_WORDS:
            return False
        cutoff = time.monotonic() - self.WINDOW_S
        with self._lock:
            recent = [r for t, r in self._recent if t >= cutoff]
        for r in recent:
            overlap = len(words & set(r.split())) / len(words)
            if overlap >= self.THRESHOLD or SequenceMatcher(None, n, r).ratio() >= self.THRESHOLD:
                self.hits += 1
                return True
        return False
