"""Build a ready-to-run translation direction from Settings (loads and warms all models)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from app.core import languages
from app.core.config import Settings
from app.pipeline.speech_translator import SpeechTranslator
from app.pipeline.voice_translator import VoiceTranslator
from app.services.stt import registry as stt_registry
from app.services.translation import registry as mt_registry
from app.services.tts import registry as tts_registry
from app.services.tts.base import NoVoiceError

log = logging.getLogger(__name__)

Direction = Literal["outgoing", "incoming"]


# Until the other person has been heard, "translate into their language" (auto) means English.
AUTO_TARGET_FALLBACK = "en"


@dataclass(frozen=True)
class DirectionLanguages:
    source: str  # may be "auto"
    target: str  # a real language; see `follows_other` for "auto"
    follows_other: bool = False  # target = whatever the other side was last heard speaking


def direction_languages(settings: Settings, direction: Direction,
                        source: str | None = None, target: str | None = None) -> DirectionLanguages:
    """outgoing: I speak -> other person's language. incoming: they speak -> my language."""
    if direction == "outgoing":
        src, tgt = settings.my_language, settings.other_language
    else:
        src, tgt = settings.other_language, settings.my_language
    src, tgt = source or src, target or tgt
    if src != languages.AUTO and not languages.can_listen(src):
        raise languages.UnsupportedLanguageError(
            f"Speech in {languages.get(src).name} can't be recognised yet (no speech-recognition support)")
    follows = False
    if tgt == languages.AUTO:
        if direction != "outgoing":
            raise languages.UnsupportedLanguageError("Your own language can't be 'auto'")
        tgt, follows = AUTO_TARGET_FALLBACK, True
    languages.get(tgt)
    return DirectionLanguages(src, tgt, follows)


@dataclass
class Providers:
    """One instance of each model, shared by both directions (they're thread-safe)."""

    stt: object
    mt: object
    tts: object

    @classmethod
    def load(cls, settings: Settings) -> "Providers":
        p = cls(stt_registry.create(settings), mt_registry.create(settings), tts_registry.create(settings))
        p.tts.load()
        p.mt.load()
        p.stt.load()
        return p


def build_voice_translator(settings: Settings, direction: Direction, *, source: str | None = None,
                           target: str | None = None, voice: str | None = None,
                           speed: float | None = None, providers: Providers | None = None,
                           follow_target=None) -> VoiceTranslator:
    """Raises UnsupportedLanguageError, STTUnavailableError, TranslationUnavailableError,
    TTSUnavailableError — callers turn these into user messages.

    follow_target: for an "auto" target, a callable returning the language the other side was
    last confidently heard speaking (see TranslationSession)."""
    langs = direction_languages(settings, direction, source, target)
    providers = providers or Providers.load(settings)
    stt, mt, tts = providers.stt, providers.mt, providers.tts
    try:
        tts.ensure_voice(langs.target)  # downloads the voice on first use of a language
        has_voice = True
    except NoVoiceError as exc:
        log.warning("%s — translations will be shown as text only", exc)
        has_voice = False
    hint = settings.other_language if direction == "incoming" and settings.other_language != languages.AUTO else None
    translator = SpeechTranslator(stt, mt, source=langs.source, target=langs.target,
                                  min_confidence=settings.language_min_confidence, source_hint=hint,
                                  follow_target=follow_target if langs.follows_other else None,
                                  glossary=settings.glossary)
    vt = VoiceTranslator(translator, tts, voice=voice,
                         speed=speed if speed is not None else settings.speech_speed)
    vt.speak = has_voice
    if has_voice and hasattr(tts, "warm_up"):
        tts.warm_up(langs.target, voice)
    return vt
