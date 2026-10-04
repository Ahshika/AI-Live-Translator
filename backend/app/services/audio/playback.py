"""Audio output: a continuously running device stream fed from an in-memory queue.

Why a long-lived stream instead of sd.play() per sentence: opening a WASAPI stream costs
tens of ms and clicks; keeping one open makes each new sentence start immediately, and
lets us *interrupt* (barge-in) by clearing the queue with a short fade instead of a click.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
import sounddevice as sd

from app.services.audio.devices import AudioDevice, find_device
from app.services.audio.format import resample
from app.services.tts.base import AudioChunk


class SpeakerUnavailableError(RuntimeError):
    pass


class AudioPlayer:
    FADE_MS = 30

    def __init__(self, device: AudioDevice | str | int | None = None):
        if device is not None and not isinstance(device, AudioDevice):
            device = find_device(device, kind="output")
        if device is None:
            info = sd.query_devices(kind="output")
            self._device_index, self.device_name = None, info["name"]
            self.rate = int(info["default_samplerate"])
            self.channels = max(1, min(int(info["max_output_channels"]), 2))
        else:
            self._device_index, self.device_name = device.index, device.name
            self.rate = int(device.default_sample_rate)
            # WASAPI shared mode only accepts the endpoint's own channel count (usually stereo):
            # opening a stereo device as mono fails with "Invalid device". We play mono on all channels.
            self.channels = max(1, min(device.max_output_channels, 2))
        self._buffers: deque[np.ndarray] = deque()
        self._offset = 0
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._stream: sd.OutputStream | None = None
        self.underruns = 0
        self.gain = 1.0  # 'ducking': lowered while the listener talks over the translation

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        try:
            self._stream = sd.OutputStream(device=self._device_index, samplerate=self.rate, channels=self.channels,
                                           dtype="float32", latency="low", callback=self._callback)
            self._stream.start()
        except sd.PortAudioError as exc:
            raise SpeakerUnavailableError(f"Cannot open {self.device_name!r}: {exc}") from exc

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "AudioPlayer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- producer side -------------------------------------------------------
    def enqueue(self, chunk: AudioChunk) -> None:
        samples = resample(chunk.samples, chunk.sample_rate, self.rate)
        if samples.size == 0:
            return
        with self._lock:
            self._buffers.append(samples)
            self._idle.clear()

    def stop(self) -> None:
        """Interrupt: fade out what's playing now and drop everything queued."""
        fade = self.rate * self.FADE_MS // 1000
        with self._lock:
            if self._buffers:
                head = self._buffers[0][self._offset:self._offset + fade].copy()
                head *= np.linspace(1.0, 0.0, len(head), dtype=np.float32)
                self._buffers.clear()
                self._buffers.append(head)
                self._offset = 0

    def duck(self, gain: float = 0.3) -> None:
        self.gain = gain

    def unduck(self) -> None:
        self.gain = 1.0

    def wait(self, timeout: float | None = None) -> bool:
        """Block until everything queued has been played."""
        return self._idle.wait(timeout)

    @property
    def queued_seconds(self) -> float:
        with self._lock:
            return (sum(len(b) for b in self._buffers) - self._offset) / self.rate

    # -- device thread -------------------------------------------------------
    def _callback(self, outdata: np.ndarray, frames: int, time, status) -> None:  # noqa: ARG002
        out = outdata[:, 0]  # fill channel 0, then copy it to the others
        filled = 0
        with self._lock:
            while filled < frames and self._buffers:
                buf = self._buffers[0]
                n = min(frames - filled, len(buf) - self._offset)
                out[filled:filled + n] = buf[self._offset:self._offset + n]
                filled += n
                self._offset += n
                if self._offset >= len(buf):
                    self._buffers.popleft()
                    self._offset = 0
            if not self._buffers:
                self._idle.set()
        if filled < frames:
            out[filled:] = 0.0
        if self.gain != 1.0:
            out *= self.gain
        if outdata.shape[1] > 1:
            outdata[:, 1:] = out[:, None]
