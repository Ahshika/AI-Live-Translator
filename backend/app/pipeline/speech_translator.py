"""One direction of the conversation: utterance audio -> transcript -> translation.

Phase 2 scope. TTS (Phase 3) and streaming/VAD (Phases 5-6) plug in around this class;
its job stays the same: take one *final* utterance and produce one translation.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

from app.core import languages
from app.pipeline.language import LanguageDecision, LanguageResolver
from app.services.audio.format import duration_seconds
from app.services.stt.base import SpeechToTextProvider, Transcript
from app.services.translation.base import TranslationProvider
from app.pipeline.text_cleanup import clean_transcript, translation_units


@dataclass
class StageTimer:
    """Wall-clock ms per stage, the raw material for latency metrics."""

    marks: dict[str, float] = field(default_factory=dict)
    _t: float = field(default_factory=time.perf_counter)

    def lap(self, stage: str) -> None:
        now = time.perf_counter()
        self.marks[stage] = (now - self._t) * 1000
        self._t = now

    @property
    def total_ms(self) -> float:
        return sum(self.marks.values())


@dataclass
class TranslatedUtterance:
    transcript: Transcript
    language: LanguageDecision
    target_language: str
    translated_text: str
    audio_seconds: float
    timings: StageTimer

    @property
    def skipped(self) -> bool:
        return self.transcript.is_empty


class SpeechTranslator:
    def __init__(self, stt: SpeechToTextProvider, mt: TranslationProvider, *,
                 source: str, target: str, min_confidence: float = 0.6, source_hint: str | None = None,
                 follow_target: Callable[[], str | None] | None = None, glossary: str = "",
                 clean: bool = True):
        """follow_target: the target language changes over time (e.g. "translate into whatever the
        other person was last heard speaking"); `target` is used until it returns a language."""
        languages.get(target)  # validate early: UnsupportedLanguageError
        if source != languages.AUTO:
            languages.get(source)
        self.stt, self.mt, self.default_target = stt, mt, target
        self.follow_target = follow_target
        self.resolver = LanguageResolver(source, hint=source_hint, min_confidence=min_confidence)
        self.glossary = glossary  # names / terms to help speech recognition spell right
        self.clean = clean
        self._context = ""  # what this speaker said last (a hint for the next utterance)
        self._context_at = 0.0

    @property
    def target(self) -> str:
        followed = self.follow_target() if self.follow_target else None
        return followed if followed in languages.LANGUAGES else self.default_target

    CONTEXT_SECONDS = 30  # older talk is a different topic: no hint
    CONTEXT_CHARS = 200

    def stt_hints(self) -> dict:
        if not getattr(self.stt, "supports_context", False):
            return {}
        # With auto-detect the last sentence may be another person in another language: it would
        # pull detection toward that language, so context is only used for a fixed language.
        recent = (self.resolver.configured != languages.AUTO
                  and time.monotonic() - self._context_at < self.CONTEXT_SECONDS)
        hints = {"context": self._context[-self.CONTEXT_CHARS:] if recent else None,
                 "hotwords": self.glossary_hotwords()}
        return {k: v for k, v in hints.items() if v}

    def glossary_hotwords(self) -> str | None:
        terms = [t.strip() for t in self.glossary.replace("\n", ",").split(",") if t.strip()]
        return ", ".join(terms) or None

    def transcribe(self, audio: np.ndarray) -> tuple[Transcript, LanguageDecision]:
        transcript = self.stt.transcribe(audio, self.resolver.stt_language, **self.stt_hints())
        decision = self.resolver.resolve(transcript.language, transcript.language_probability, transcript.text)
        if self.clean and not transcript.is_empty:
            transcript = dataclasses.replace(transcript, text=clean_transcript(transcript.text, decision.code))
        if not transcript.is_empty:
            self._context, self._context_at = transcript.text, time.monotonic()
        return transcript, decision

    def translate_sentences(self, text: str, source: str, target: str | None = None) -> Iterator[tuple[str, str]]:
        """Yield (source sentence, translation) one at a time, so speech can start on the first.
        Tiny fragments ("Ja. Genau.") are translated together with a neighbour for context."""
        target = target or self.target
        for sentence in translation_units(text):
            if languages.same_base(source, target):
                yield sentence, sentence
            else:
                yield sentence, self.mt.translate_batch([sentence], source, target)[0]

    def process(self, audio: np.ndarray) -> TranslatedUtterance:
        timer = StageTimer()
        target = self.target
        transcript, decision = self.transcribe(audio)
        timer.lap("stt")
        translated = ""
        if not transcript.is_empty:
            translated = self.mt.translate(transcript.text, decision.code, target).text
        timer.lap("mt")
        return TranslatedUtterance(transcript, decision, target, translated,
                                   duration_seconds(audio), timer)
