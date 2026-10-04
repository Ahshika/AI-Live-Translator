"""Microphone capture: device callback -> bounded queue of canonical 20 ms frames.

The PortAudio callback runs on a real-time audio thread, so it only copies the raw
block into a queue. Conversion/resampling happens on the consumer side. If the
consumer falls behind, the oldest audio is dropped and counted (never unbounded RAM).
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd

from app.services.audio.devices import AudioDevice, default_input_device
from app.services.audio.format import FRAME_MS, FRAME_SAMPLES, StreamResampler


class MicrophoneUnavailableError(RuntimeError):
    pass


@dataclass
class CaptureStats:
    blocks_received: int = 0
    blocks_dropped: int = 0
    overflows: int = 0


@dataclass
class MicrophoneCapture:
    device: AudioDevice | None = None
    max_queue_seconds: float = 5.0
    stats: CaptureStats = field(default_factory=CaptureStats)

    def __post_init__(self) -> None:
        self.device = self.device or default_input_device()
        if self.device is None:
            raise MicrophoneUnavailableError("No input device found")
        self._rate = int(self.device.default_sample_rate)
        self._channels = min(self.device.max_input_channels, 2)
        self._block = self._rate * FRAME_MS // 1000
        self._queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=max(1, int(self.max_queue_seconds * 1000 / FRAME_MS))
        )
        self._stream: sd.InputStream | None = None
        self._pending = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._resampler = StreamResampler(self._rate)

    # -- device thread -------------------------------------------------------
    def _callback(self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags) -> None:  # noqa: ARG002
        if status.input_overflow:
            self.stats.overflows += 1
        self.stats.blocks_received += 1
        block = indata.copy()
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self.stats.blocks_dropped += 1
            except queue.Empty:
                pass
            self._queue.put_nowait(block)

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        try:
            self._stream = sd.InputStream(
                device=self.device.index,
                samplerate=self._rate,
                channels=self._channels,
                dtype="float32",
                blocksize=self._block,
                latency="low",
                callback=self._callback,
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            raise MicrophoneUnavailableError(f"Cannot open {self.device.name!r}: {exc}") from exc

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "MicrophoneCapture":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def clear(self) -> None:
        """Discard anything captured so far (e.g. right before push-to-talk starts)."""
        with self._lock:
            while not self._queue.empty():
                self._queue.get_nowait()
            self._pending = np.zeros(0, dtype=np.float32)

    # -- consumer side -------------------------------------------------------
    def read_frame(self, timeout: float | None = 1.0) -> np.ndarray | None:
        """Return one canonical 20 ms frame (320 float32 samples @16 kHz), or None on timeout."""
        with self._lock:
            while len(self._pending) < FRAME_SAMPLES:
                try:
                    block = self._queue.get(timeout=timeout)
                except queue.Empty:
                    return None
                self._pending = np.concatenate([self._pending, self._resampler(block)])
            frame, self._pending = self._pending[:FRAME_SAMPLES], self._pending[FRAME_SAMPLES:]
            return frame
