"""Microphone clean-up before speech recognition — numpy only, a few ms per utterance.

    frames ─► AutoGain (quiet mic? raise it, before the VAD) ─► segmenter ─► utterance
    silent frames ─► NoiseProfile (what the room sounds like)          │
                                                                        ▼
                                    enhance_utterance: high-pass, noise gate, loudness

Why each step:
  * AutoGain: laptop / headset mics are often 20-30 dB too quiet. The VAD then misses soft
    words and the start of sentences. Gain adapts slowly, from finished utterances only.
  * High-pass (80 Hz): removes desk thumps, fan rumble and mains hum; no speech lives there.
  * Noise gate (spectral subtraction against the room's own noise profile, learnt while you're
    silent): fans, keyboards, street noise. Kept mild on purpose — aggressive denoising
    creates "musical" artifacts that hurt Whisper more than the noise did.
  * Loudness: every utterance reaches the recogniser at the same level.
"""

from __future__ import annotations

import numpy as np

from app.services.audio.format import SAMPLE_RATE, rms_dbfs

N_FFT = 512
HOP = N_FFT // 2
_WINDOW = np.sqrt(np.hanning(N_FFT + 1)[:-1]).astype(np.float32)  # sqrt-Hann: perfect overlap-add at 50%

# strength -> (over-subtraction, gain floor)
STRENGTHS = {"light": (1.5, 0.25), "strong": (2.5, 0.08)}


def _stft(x: np.ndarray) -> np.ndarray:
    pad = np.concatenate([np.zeros(HOP, np.float32), x, np.zeros(N_FFT, np.float32)])
    n = 1 + (len(pad) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(pad, (n, N_FFT), (pad.strides[0] * HOP, pad.strides[0]))
    return np.fft.rfft(frames * _WINDOW, axis=1)


def _istft(spec: np.ndarray, length: int) -> np.ndarray:
    frames = np.fft.irfft(spec, n=N_FFT, axis=1).astype(np.float32) * _WINDOW
    out = np.zeros(HOP * (len(frames) + 1), np.float32)
    for i, f in enumerate(frames):
        out[i * HOP:i * HOP + N_FFT] += f
    return out[HOP:HOP + length]


class NoiseProfile:
    """Running average power spectrum of the room while nobody speaks."""

    def __init__(self, smoothing: float = 0.95):
        self.smoothing = smoothing
        self.power: np.ndarray | None = None
        self._buf = np.zeros(0, np.float32)
        self.frames = 0

    def update(self, audio: np.ndarray) -> None:
        self._buf = np.concatenate([self._buf, audio])
        while len(self._buf) >= N_FFT:
            frame, self._buf = self._buf[:N_FFT], self._buf[HOP:]
            p = np.abs(np.fft.rfft(frame * _WINDOW)) ** 2
            self.power = p if self.power is None else self.smoothing * self.power + (1 - self.smoothing) * p
            self.frames += 1

    @property
    def ready(self) -> bool:
        return self.frames >= 20  # ~0.3 s of silence heard

    def reset(self) -> None:
        self.__init__(self.smoothing)


def high_pass(audio: np.ndarray, cutoff_hz: float = 80.0) -> np.ndarray:
    """Zero-phase high-pass on a finished utterance (FFT; fine because we have it all)."""
    if audio.size < 64:
        return audio
    spec = np.fft.rfft(audio - audio.mean())
    freqs = np.fft.rfftfreq(audio.size, 1 / SAMPLE_RATE)
    ramp = np.clip((freqs - cutoff_hz * 0.5) / (cutoff_hz * 0.5), 0.0, 1.0)  # gentle slope, no ringing
    return np.fft.irfft(spec * ramp, n=audio.size).astype(np.float32)


def reduce_noise(audio: np.ndarray, noise_power: np.ndarray, strength: str = "light") -> np.ndarray:
    alpha, floor = STRENGTHS[strength]
    spec = _stft(audio)
    power = np.abs(spec) ** 2
    gain = np.sqrt(np.clip(1.0 - alpha * noise_power[None, :] / np.maximum(power, 1e-12), floor ** 2, 1.0))
    # Smooth the gain over time so it doesn't flutter (the cause of "musical noise").
    for i in range(1, len(gain)):
        gain[i] = np.maximum(gain[i], 0.6 * gain[i - 1] + 0.4 * gain[i])
    return _istft(spec * gain, len(audio))


def normalize_loudness(audio: np.ndarray, target_dbfs: float = -20.0, max_gain_db: float = 24.0,
                       peak: float = 0.95) -> np.ndarray:
    """Bring speech to a steady level without clipping."""
    level = rms_dbfs(audio)
    if not np.isfinite(level):
        return audio
    gain = 10 ** (min(max_gain_db, target_dbfs - level) / 20)
    out = audio * gain
    top = float(np.max(np.abs(out))) if out.size else 0.0
    if top > peak:
        out *= peak / top
    return out.astype(np.float32, copy=False)


def enhance_utterance(audio: np.ndarray, noise: NoiseProfile | None = None,
                      strength: str = "light") -> np.ndarray:
    out = high_pass(audio)
    if strength != "off" and noise is not None and noise.ready:
        out = reduce_noise(out, noise.power, strength)
    return normalize_loudness(out)


class AutoGain:
    """Slow automatic gain for a quiet microphone, applied to the live stream (before the VAD).

    Adapts once per finished utterance toward speech at about -22 dBFS; never more than
    +18 dB, never below unity (a loud mic is left alone, the normaliser handles it).
    """

    TARGET_DBFS = -22.0
    MAX_GAIN_DB = 18.0
    STEP_DB = 6.0

    def __init__(self):
        self.gain_db = 0.0

    @property
    def gain(self) -> float:
        return 10 ** (self.gain_db / 20)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self.gain_db == 0.0:
            return frame
        return np.clip(frame * self.gain, -1.0, 1.0)

    def learn(self, utterance_after_gain: np.ndarray) -> None:
        level = rms_dbfs(utterance_after_gain)
        if not np.isfinite(level):
            return
        error = self.TARGET_DBFS - level
        if error > 3:
            self.gain_db = min(self.MAX_GAIN_DB, self.gain_db + min(self.STEP_DB, error))
        elif error < -6:
            self.gain_db = max(0.0, self.gain_db - self.STEP_DB)


class Conditioner:
    """Everything above for one live source (your microphone)."""

    def __init__(self, noise_reduction: str = "light", auto_gain: bool = True):
        self.strength = noise_reduction
        self.agc = AutoGain() if auto_gain else None
        self.noise = NoiseProfile()

    def frame(self, frame: np.ndarray) -> np.ndarray:
        return self.agc(frame) if self.agc else frame

    def silence(self, frame: np.ndarray) -> None:
        """A frame in which nobody spoke: it teaches the noise profile."""
        if self.strength != "off":
            self.noise.update(frame)

    def utterance(self, audio: np.ndarray) -> np.ndarray:
        if self.agc:
            self.agc.learn(audio)
        return enhance_utterance(audio, self.noise, self.strength)

    def partial(self, audio: np.ndarray) -> np.ndarray:
        return enhance_utterance(audio, self.noise, self.strength)

    def reset(self) -> None:
        self.noise.reset()
