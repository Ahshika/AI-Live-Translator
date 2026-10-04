"""Canonical in-engine audio format and conversions.

Everything inside the engine is PCM float32, mono, 16 kHz in the range [-1, 1].
Conversion happens exactly once on the way in (device -> engine) and once on the
way out (engine -> device), never in between.
"""

from __future__ import annotations

import numpy as np
import soxr

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = np.float32
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16/int32/float PCM with shape (n,) or (n, channels) to mono float32."""
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32) if audio.shape[1] > 1 else audio[:, 0]
    return np.ascontiguousarray(audio, dtype=DTYPE)


def resample(audio: np.ndarray, src_rate: int, dst_rate: int = SAMPLE_RATE) -> np.ndarray:
    """High-quality resampling (anti-aliased) of a mono float32 signal."""
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(DTYPE, copy=False)
    return soxr.resample(audio, src_rate, dst_rate, quality="HQ").astype(DTYPE, copy=False)


def to_engine_format(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """Any device/file PCM -> canonical float32 mono 16 kHz."""
    return resample(to_mono_float32(audio), src_rate, SAMPLE_RATE)


class StreamResampler:
    """Stateful resampler for live streams (mic, loopback).

    Resampling each 20 ms block independently with a one-shot resampler creates clicks at
    every block edge (the filter has no history). soxr's stream mode keeps that state.
    """

    def __init__(self, src_rate: int, dst_rate: int = SAMPLE_RATE):
        self.passthrough = src_rate == dst_rate
        self._rs = None if self.passthrough else soxr.ResampleStream(src_rate, dst_rate, 1, dtype="float32",
                                                                     quality="HQ")

    def __call__(self, block: np.ndarray) -> np.ndarray:
        mono = to_mono_float32(block)
        return mono if self.passthrough else self._rs.resample_chunk(mono)


def duration_seconds(audio: np.ndarray, rate: int = SAMPLE_RATE) -> float:
    return len(audio) / rate


def trim_silence(audio: np.ndarray, rate: int, *, threshold: float = 0.02,
                 keep_lead_ms: int = 30, keep_trail_ms: int = 150) -> np.ndarray:
    """Cut leading/trailing near-silence (TTS padding) — leading silence is pure added latency.

    A short trailing pause is kept so consecutive sentences don't run into each other.
    """
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        return audio[:0]
    start = max(0, loud[0] - rate * keep_lead_ms // 1000)
    end = min(len(audio), loud[-1] + 1 + rate * keep_trail_ms // 1000)
    return audio[start:end]


def rms_dbfs(audio: np.ndarray) -> float:
    """Loudness in dBFS; used for level meters and 'is the mic silent?' checks."""
    if audio.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * np.log10(rms) if rms > 0 else float("-inf")
