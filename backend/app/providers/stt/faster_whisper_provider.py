"""Local STT using faster-whisper (CTranslate2). Free, offline, 99 languages."""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path

import numpy as np

from app.core.config import models_dir
from app.services.audio.format import SAMPLE_RATE
from app.services.stt.base import (
    SpeechToTextProvider,
    STTUnavailableError,
    Transcript,
    TranscriptSegment,
)
from app.utils.cuda import cuda_runtime_ready, register_cuda_dlls

log = logging.getLogger(__name__)


def resolve_model(model: str) -> str:
    """Prefer a pre-downloaded copy in backend/models/faster-whisper-<name> (resumable
    curl download on slow links); otherwise faster-whisper fetches it from Hugging Face."""
    local = models_dir() / f"faster-whisper-{model}"
    return str(local) if (local / "model.bin").exists() else model


# Whisper was trained on subtitled video, so on silence/noise it "hears" subtitle credits.
# In a live call these would be translated and spoken to the other person — never allow that.
# These phrases are specific enough to drop any short segment that contains them.
_HALLUCINATIONS = (
    "ترجمة نانسي قنقر", "نانسي قنقر", "اشتركوا في القناة", "اشترك في القناة", "شكرا للمشاهدة",
    "شكرا على المشاهدة",
    "untertitel der amara.org-community", "untertitel im auftrag des zdf", "untertitelung des zdf",
    "vielen dank fürs zuschauen", "thank you for watching", "thanks for watching",
    "please subscribe", "subtitles by the amara.org community", "sous-titres réalisés par",
    "sous-titrage st'", "¡suscríbete!", "продолжение следует",
)


# Whole-segment outputs Whisper produces for hum/noise/music. Only an EXACT match is dropped:
# "أنا بحب الموسيقى" or "سبحان الله وبحمده" said in a real sentence must still be translated.
# ("Thank you." is NOT here: people really say it. The VAD gate keeps noise from reaching STT.)
_EXACT_HALLUCINATIONS = {"you", "you you", "hmm", "mm", "uh", "موسيقى", "music", "musik", "سبحان الله وبحمده"}
_BRACKETED = re.compile(r"^[\[(♪*].*[\])♪*]$")  # "[موسيقى]", "(Musik)", "♪ ... ♪"


def is_hallucination(segment) -> bool:
    raw = segment.text.strip()
    if _BRACKETED.match(raw):
        return True
    text = raw.lower().strip(" .!?،؟")
    if not text or text in _EXACT_HALLUCINATIONS:
        return True
    if any(h in text for h in _HALLUCINATIONS) and len(text) < 60:
        return True
    # Model itself thinks there was no speech and isn't confident in the words.
    return segment.no_speech_prob > 0.6 and segment.avg_logprob < -1.0


def _echoes(segments, context: str) -> bool:
    text = " ".join(s.text for s in segments).strip().lower().strip(" .!?،؟")
    return len(text) >= 4 and text in context.lower()


class FasterWhisperProvider(SpeechToTextProvider):
    name = "faster-whisper"
    supports_context = True

    def __init__(self, model: str = "large-v3-turbo", device: str = "auto", compute_type: str = "auto"):
        self.model_name = model
        self.requested_device = device
        self.requested_compute_type = compute_type
        self.device: str | None = None
        self.compute_type: str | None = None
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        register_cuda_dlls()
        from faster_whisper import WhisperModel

        attempts: list[tuple[str, str]] = []
        if self.requested_device == "cuda" or (self.requested_device == "auto" and cuda_runtime_ready()):
            attempts.append(("cuda", "float16" if self.requested_compute_type == "auto" else self.requested_compute_type))
        if self.requested_device in ("auto", "cpu"):
            attempts.append(("cpu", "int8" if self.requested_compute_type == "auto" else self.requested_compute_type))

        errors = []
        for device, compute_type in attempts:
            try:
                t0 = time.perf_counter()
                model = WhisperModel(resolve_model(self.model_name), device=device, compute_type=compute_type)
                # Warm-up: the first CUDA call compiles kernels; do it now, not on the user's first sentence.
                list(model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="en")[0])
                self._model, self.device, self.compute_type = model, device, compute_type
                log.info("Loaded %s on %s/%s in %.1fs", self.model_name, device, compute_type, time.perf_counter() - t0)
                return
            except Exception as exc:  # GPU missing, DLL missing, OOM...
                log.warning("faster-whisper failed on %s/%s: %s", device, compute_type, exc)
                errors.append(f"{device}: {exc}")
        raise STTUnavailableError("Could not load Whisper model: " + " | ".join(errors))

    def transcribe(self, audio: np.ndarray, language: str | None = None, *, fast: bool = False,
                   context: str | None = None, hotwords: str | None = None) -> Transcript:
        if self._model is None:
            self.load()
        with self._lock:  # one CTranslate2 model instance -> serialise calls
            segments, info = self._model.transcribe(
                audio,
                language=language,
                # On a CPU, beam search costs 2-4x the time for a small accuracy gain: in a live
                # call a fast answer beats a perfect one that arrives after they moved on.
                beam_size=1 if fast or self.device == "cpu" else 5,
                vad_filter=False,  # VAD is our own pipeline stage (Phase 6)
                condition_on_previous_text=False,  # avoids hallucination loops on short clips
                without_timestamps=False,
                # The previous sentence keeps topic, names and punctuation style consistent
                # across utterances; hotwords bias recognition toward the user's own terms.
                initial_prompt=context or None,
                hotwords=hotwords or None,
            )
            segs = [TranscriptSegment(s.start, s.end, s.text.strip())
                    for s in segments if not is_hallucination(s)]
            if context and segs and _echoes(segs, context):
                segs = []  # on near-silence Whisper sometimes just repeats its prompt
        return Transcript(
            text=" ".join(s.text for s in segs if s.text).strip(),
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            segments=segs,
        )

    def supported_languages(self) -> set[str] | None:
        from faster_whisper.tokenizer import _LANGUAGE_CODES

        return set(_LANGUAGE_CODES)
