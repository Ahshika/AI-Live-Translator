"""Text-to-Speech provider contract.

TTS audio keeps the provider's native sample rate (e.g. 22.05 kHz for Piper) — resampling
it down to the 16 kHz STT format would only lose quality. The audio sink converts to the
output device's rate exactly once.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np


class TTSUnavailableError(RuntimeError):
    """Provider cannot run (model missing, API down, quota...)."""


class NoVoiceError(TTSUnavailableError):
    """No installed voice for the requested language."""


@dataclass(frozen=True)
class AudioChunk:
    samples: np.ndarray  # float32 mono in [-1, 1]
    sample_rate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass(frozen=True)
class Voice:
    id: str
    language: str  # our base language code ("ar", "de")
    name: str
    gender: str | None  # "male" | "female" | None
    quality: str | None


class TextToSpeechProvider(ABC):
    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Discover voices / open connections. Must be idempotent."""

    @abstractmethod
    def voices(self, language: str | None = None) -> list[Voice]:
        """Installed voices, best first; optionally filtered by our language code."""

    def ensure_voice(self, language: str, progress=None) -> str:
        """Make sure a voice for the language is ready (may download). Raises NoVoiceError."""
        found = self.voices(language)
        if not found:
            raise NoVoiceError(f"No voice for language {language!r}")
        return found[0].id

    @abstractmethod
    def synthesize_stream(self, text: str, language: str, *, voice: str | None = None,
                          speed: float = 1.0) -> Iterator[AudioChunk]:
        """Yield audio as soon as each piece (sentence) is ready — playback can start early."""

    def synthesize(self, text: str, language: str, *, voice: str | None = None,
                   speed: float = 1.0) -> AudioChunk:
        chunks = list(self.synthesize_stream(text, language, voice=voice, speed=speed))
        if not chunks:
            return AudioChunk(np.zeros(0, np.float32), 22_050)
        return AudioChunk(np.concatenate([c.samples for c in chunks]), chunks[0].sample_rate)
