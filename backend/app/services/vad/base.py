"""Voice Activity Detection contract: a stream of audio -> a stream of speech probabilities."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VoiceActivityDetector(ABC):
    window_samples: int  # how many 16 kHz samples each probability covers

    @abstractmethod
    def __call__(self, window: np.ndarray) -> float:
        """Speech probability (0..1) for exactly `window_samples` samples."""

    @abstractmethod
    def reset(self) -> None:
        """Forget the recurrent state (new stream)."""
