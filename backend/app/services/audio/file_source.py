"""An AudioSource that plays a file (or array) as if it were a live microphone.

Used by tests and demos: the live pipeline can't tell it from a real device.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from app.services.audio.format import FRAME_SAMPLES, SAMPLE_RATE, to_engine_format


class FileAudioSource:
    def __init__(self, audio: np.ndarray | str | Path, *, realtime: bool = False,
                 lead_silence_s: float = 0.5, tail_silence_s: float = 2.0):
        if not isinstance(audio, np.ndarray):
            import soundfile as sf

            data, rate = sf.read(audio, dtype="float32")
            audio = to_engine_format(data, rate)
        pad = lambda s: np.zeros(int(s * SAMPLE_RATE), np.float32)  # noqa: E731
        self.audio = np.concatenate([pad(lead_silence_s), audio, pad(tail_silence_s)])
        self.realtime = realtime
        self._pos = 0
        self._t0 = 0.0
        self.finished = threading.Event()

    def start(self) -> None:
        self._pos, self._t0 = 0, time.perf_counter()

    def stop(self) -> None:
        pass

    def read_frame(self, timeout: float | None = 1.0) -> np.ndarray | None:
        if self._pos >= len(self.audio):
            self.finished.set()
            time.sleep(min(timeout or 0, 0.05))
            return None
        if self.realtime:
            due = self._t0 + self._pos / SAMPLE_RATE
            delay = due - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        frame = self.audio[self._pos:self._pos + FRAME_SAMPLES]
        self._pos += FRAME_SAMPLES
        if len(frame) < FRAME_SAMPLES:
            frame = np.pad(frame, (0, FRAME_SAMPLES - len(frame)))
        return frame
