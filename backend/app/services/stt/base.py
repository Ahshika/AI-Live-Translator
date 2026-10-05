"""Speech-to-Text provider contract.

Every STT backend (local Whisper, a cloud API, ...) implements SpeechToTextProvider.
The rest of the engine only ever depends on this module, never on a concrete provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class STTUnavailableError(RuntimeError):
    """Provider cannot run (model missing, GPU failure, API down, quota...)."""


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None
    language_probability: float | None
    duration: float
    segments: list[TranscriptSegment] = field(default_factory=list)
    is_final: bool = True

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class SpeechToTextProvider(ABC):
    name: str = "base"
    # Can it take recent conversation text and expected words as hints (see transcribe)?
    supports_context: bool = False

    @abstractmethod
    def load(self) -> None:
        """Load models / open connections. Called once; must be idempotent."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, language: str | None = None, *, fast: bool = False) -> Transcript:
        """Transcribe canonical audio (float32 mono 16 kHz).

        language: Whisper code ("ar", "de"...) or None for auto-detection.
        fast: trade accuracy for speed (used for live partial subtitles, never for translation).
        Providers with supports_context also accept keyword arguments
            context: what this speaker said just before (continuity, spelling of names)
            hotwords: names / terms the user expects ("Ahmed, Siemens, Kubernetes")
        """

    def supported_languages(self) -> set[str] | None:
        """ISO-639-1 codes, or None if unknown/unrestricted."""
        return None
