"""One direction, voice to voice: utterance audio -> STT -> MT -> TTS -> audio sink.

Latency trick: translation and speech run sentence by sentence. The first sentence is
spoken as soon as it is translated and synthesised, while later sentences are still being
processed — the listener never waits for the whole utterance.

The sink is any callable taking AudioChunk: headphones (AudioPlayer.enqueue) today, the
virtual microphone in Phase 7. The pipeline doesn't know or care which.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.core import languages
from app.pipeline.language import LanguageDecision
from app.pipeline.speech_translator import SpeechTranslator
from app.services.audio.enhance import normalize_loudness
from app.services.audio.format import duration_seconds
from app.services.stt.base import Transcript
from app.services.tts.base import AudioChunk, NoVoiceError, TextToSpeechProvider

AudioSink = Callable[[AudioChunk], None]


@dataclass
class VoiceResult:
    transcript: Transcript
    language: LanguageDecision
    target_language: str
    pairs: list[tuple[str, str]] = field(default_factory=list)  # (source sentence, translation)
    audio_seconds: float = 0.0  # input speech length
    speech_seconds: float = 0.0  # generated speech length
    # ms measured from the end of the user's speech (= when process() was called)
    stt_ms: float = 0.0
    first_audio_ms: float | None = None  # the latency the listener actually feels
    total_ms: float = 0.0
    text_only: bool = False  # no voice for the target language -> subtitles only
    same_language: bool = False  # spoken in the listener's own language: shown, not re-spoken
    rejected: bool = False  # e.g. the mic only heard an echo of the meeting: dropped

    @property
    def translated_text(self) -> str:
        return " ".join(t for _, t in self.pairs if t)

    @property
    def skipped(self) -> bool:
        return self.transcript.is_empty or self.rejected


class VoiceTranslator:
    def __init__(self, translator: SpeechTranslator, tts: TextToSpeechProvider, *,
                 voice: str | None = None, speed: float = 1.0):
        self.translator, self.tts, self.voice, self.speed = translator, tts, voice, speed
        self.speak = True  # False = subtitles only (no voice for the target language)
        self.voiceless: set[str] = set()  # target languages found to have no voice

    @property
    def target(self) -> str:
        return self.translator.target

    # Every voice (and every sentence) reaches the listener at the same loudness.
    OUTPUT_DBFS = -18.0

    def process(self, audio: np.ndarray, sink: AudioSink, *, speak: bool = True,
                speed_factor: float = 1.0, reject=None) -> VoiceResult:
        """speak=False: translate but don't voice it (backlog / subtitles-only).
        speed_factor: talk a little faster to catch up when the conversation runs ahead."""
        t0 = time.perf_counter()
        ms = lambda: (time.perf_counter() - t0) * 1000  # noqa: E731
        target = self.target  # fixed for this utterance even if the followed language changes
        transcript, decision = self.translator.transcribe(audio)
        result = VoiceResult(transcript, decision, target,
                             audio_seconds=duration_seconds(audio), stt_ms=ms())
        if transcript.is_empty or (reject is not None and reject(transcript.text)):
            result.rejected = not transcript.is_empty
            result.total_ms = ms()
            return result
        # They spoke the listener's own language (common with auto-detect in group calls): the
        # listener already understood it, so show it but don't say it again over them.
        result.same_language = languages.same_base(decision.code, target)
        voice = speak and self.speak and target not in self.voiceless and not result.same_language
        for source_sentence, translated in self.translator.translate_sentences(transcript.text, decision.code,
                                                                               target):
            result.pairs.append((source_sentence, translated))
            if not voice:
                result.text_only = True
                continue
            try:
                for chunk in self.tts.synthesize_stream(translated, target, voice=self.voice,
                                                        speed=min(2.0, self.speed * speed_factor)):
                    chunk = AudioChunk(normalize_loudness(chunk.samples, self.OUTPUT_DBFS, max_gain_db=20.0),
                                       chunk.sample_rate)
                    if result.first_audio_ms is None:
                        result.first_audio_ms = ms()
                    result.speech_seconds += chunk.seconds
                    sink(chunk)
            except NoVoiceError:
                self.voiceless.add(target)
                voice, result.text_only = False, True
        result.total_ms = ms()
        return result
