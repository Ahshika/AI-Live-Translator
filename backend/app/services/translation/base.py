"""Translation provider contract. Codes are our language codes (app.core.languages)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class TranslationUnavailableError(RuntimeError):
    """Provider cannot run (model missing, GPU failure, API down, quota...)."""


@dataclass(frozen=True)
class Translation:
    source_text: str
    text: str
    source_language: str
    target_language: str


class TranslationProvider(ABC):
    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Load models / open connections. Called once; must be idempotent."""

    @abstractmethod
    def translate_batch(self, texts: list[str], source: str, target: str) -> list[str]:
        """Translate independent sentences/phrases in one call (lets the backend batch on GPU)."""

    def translate(self, text: str, source: str, target: str) -> Translation:
        from app.utils.text import split_sentences

        if source == target or not text.strip():
            return Translation(text, text, source, target)
        sentences = split_sentences(text)
        out = self.translate_batch(sentences, source, target)
        return Translation(text, " ".join(s for s in out if s).strip(), source, target)

    def supports(self, code: str) -> bool:
        return True
