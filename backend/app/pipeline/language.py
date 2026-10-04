"""Decide which language a segment was spoken in, with a fallback for unreliable detection.

Whisper's language ID is unreliable on very short clips ("ok", "ja"), names, noise and
code-switching. Rules, in order:
1. Fixed language configured            -> use it, STT is told the language (most accurate).
2. Auto: confident detection             -> use it and remember it for this speaker.
3. Auto: low confidence / too short      -> keep the speaker's last confident language.
4. Auto: nothing known yet               -> best guess, if we support it; else expected_hint.
Detected languages we don't support are treated like low-confidence results.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core import languages
from app.core.languages import AUTO


@dataclass
class LanguageDecision:
    code: str
    detected: str | None
    confidence: float | None
    reason: str  # fixed | confident | fallback-last | fallback-hint | best-guess


class LanguageResolver:
    MIN_WORDS_FOR_TRUST = 2

    def __init__(self, configured: str, *, hint: str | None = None, min_confidence: float = 0.6):
        self.configured = configured
        self.hint = hint  # e.g. "the other person is probably German"
        self.min_confidence = min_confidence
        self.last_confident: str | None = None

    @property
    def stt_language(self) -> str | None:
        """What to pass to STT: a whisper code, or None to auto-detect."""
        return None if self.configured == AUTO else languages.get(self.configured).whisper

    def resolve(self, detected_whisper: str | None, confidence: float | None, text: str) -> LanguageDecision:
        if self.configured != AUTO:
            return LanguageDecision(self.configured, detected_whisper, confidence, "fixed")

        lang = languages.from_whisper(detected_whisper)
        trusted = (
            lang is not None
            and (confidence or 0.0) >= self.min_confidence
            and len(text.split()) >= self.MIN_WORDS_FOR_TRUST
        )
        if trusted:
            self.last_confident = lang.code
            return LanguageDecision(lang.code, detected_whisper, confidence, "confident")
        if self.last_confident:
            return LanguageDecision(self.last_confident, detected_whisper, confidence, "fallback-last")
        if lang is not None:
            return LanguageDecision(lang.code, detected_whisper, confidence, "best-guess")
        fallback = self.hint or "en"
        return LanguageDecision(fallback, detected_whisper, confidence, "fallback-hint")
