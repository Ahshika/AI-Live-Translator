"""A live, continuously running translation direction.

    AudioSource ──frames──► segmenter ──utterances──► queue ──► VoiceTranslator ──► sink
                               │
                               └─partials──► fast STT (only when the GPU is free) ──► subtitles

Three threads per direction:
  * capture: reads frames, runs VAD/segmenter (cheap, never blocks on models)
  * worker:  translates finished utterances in order
  * partial: live subtitles for what is being said right now (best effort, skippable)

Backlog policy (never fall minutes behind the conversation): at most MAX_PENDING utterances
wait; when exceeded the oldest waiting one is translated as text only (shown, not spoken).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from app.pipeline.segmenter import PartialAudio, SpeechStarted, UtteranceEnded, UtteranceSegmenter
from app.pipeline.voice_translator import AudioSink, VoiceTranslator

log = logging.getLogger(__name__)
EventCallback = Callable[[dict[str, Any]], None]


class AudioSource(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read_frame(self, timeout: float | None = 1.0) -> np.ndarray | None: ...


@dataclass
class _Utterance:
    audio: np.ndarray
    ended_at: float  # perf_counter when the speaker actually stopped talking
    forced: bool
    speak: bool = True
    cancelled: bool = False


class LiveDirection:
    MAX_PENDING = 2

    def __init__(self, name: str, source: AudioSource, segmenter: UtteranceSegmenter,
                 translator: VoiceTranslator, sink: AudioSink, *, on_event: EventCallback | None = None,
                 pause_when: Callable[[], bool] | None = None, partials: bool = True):
        self.name, self.source, self.segmenter = name, source, segmenter
        self.translator, self.sink = translator, sink
        self.on_event = on_event or (lambda e: None)
        self.pause_when = pause_when  # e.g. echo guard: our own speaker is playing
        self.partials = partials
        self._queue: queue.Queue[_Utterance | None] = queue.Queue()
        self._pending: list[_Utterance] = []
        self._pending_lock = threading.Lock()
        self._partial_slot: np.ndarray | None = None
        self._partial_event = threading.Event()
        self._busy = threading.Event()  # worker is using the models
        self._running = threading.Event()
        self._muted = False
        self._threads: list[threading.Thread] = []
        self.stats = {"utterances": 0, "dropped_to_text": 0, "errors": 0}

    # -- control -------------------------------------------------------------
    def start(self) -> None:
        self.source.start()
        self._running.set()
        for target in (self._capture_loop, self._worker_loop, self._partial_loop):
            t = threading.Thread(target=target, name=f"{self.name}-{target.__name__}", daemon=True)
            t.start()
            self._threads.append(t)
        self._emit("status", state="listening")

    def stop(self) -> None:
        self._running.clear()
        self._queue.put(None)
        self._partial_event.set()
        for t in self._threads:
            t.join(timeout=5)
        self._threads.clear()
        self.source.stop()
        self._emit("status", state="stopped")

    @property
    def muted(self) -> bool:
        return self._muted

    @muted.setter
    def muted(self, value: bool) -> None:
        """Muted = keep the stream open but ignore everything heard (privacy 'mute' button)."""
        self._muted = value
        self.segmenter.reset()
        if value:  # what was waiting to be translated must not be spoken after "mute"/"pause"
            with self._pending_lock:
                for u in self._pending:
                    u.cancelled = True
        self._emit("status", state="muted" if value else "listening")

    # -- threads -------------------------------------------------------------
    def _capture_loop(self) -> None:
        while self._running.is_set():
            try:
                frame = self.source.read_frame(timeout=0.5)
            except Exception as exc:  # device unplugged etc.
                self._emit("error", code="audio_source", message=str(exc))
                time.sleep(1)
                continue
            if frame is None:
                continue
            if self._muted:
                continue
            self.segmenter.paused = bool(self.pause_when and self.pause_when())
            for ev in self.segmenter.feed(frame):
                if isinstance(ev, SpeechStarted):
                    self._emit("speech_started")
                elif isinstance(ev, PartialAudio) and self.partials:
                    self._partial_slot = ev.audio
                    self._partial_event.set()
                elif isinstance(ev, UtteranceEnded):
                    self._partial_slot = None
                    # The speaker actually stopped end_silence_ms ago (that's how we knew):
                    # measure latency from there, i.e. what the listener really waits.
                    waited = 0 if ev.forced else self.segmenter.cfg.end_silence_ms / 1000
                    self._enqueue(_Utterance(ev.audio, time.perf_counter() - waited, ev.forced))

    def _enqueue(self, utt: _Utterance) -> None:
        with self._pending_lock:
            waiting = [u for u in self._pending if u.speak]
            if len(waiting) >= self.MAX_PENDING:
                waiting[0].speak = False  # too far behind: oldest becomes subtitles only
                self.stats["dropped_to_text"] += 1
            self._pending.append(utt)
        self._queue.put(utt)

    def _worker_loop(self) -> None:
        while True:
            utt = self._queue.get()
            if utt is None or not self._running.is_set():
                return
            with self._pending_lock:
                if utt in self._pending:
                    self._pending.remove(utt)
            if utt.cancelled:
                continue
            self._busy.set()
            try:
                self._translate(utt)
            except Exception as exc:
                self.stats["errors"] += 1
                log.exception("translation failed")
                self._emit("error", code="pipeline", message=str(exc))
            finally:
                self._busy.clear()

    def _translate(self, utt: _Utterance) -> None:
        first_audio: list[float] = []

        def sink(chunk):
            if not first_audio:
                first_audio.append(time.perf_counter())
            self.sink(chunk)

        r = self.translator.process(utt.audio, sink, speak=utt.speak)
        if r.skipped:
            return
        self.stats["utterances"] += 1
        latency = (first_audio[0] - utt.ended_at) * 1000 if first_audio else None
        self._emit("final", source_language=r.language.code, target_language=r.target_language,
                   source_text=r.transcript.text, translated_text=r.translated_text,
                   spoken=bool(first_audio), text_only=r.text_only or not utt.speak,
                   detected=r.language.detected, detection=r.language.reason,
                   latency_ms=latency, stt_ms=r.stt_ms, total_ms=r.total_ms, forced_cut=utt.forced)

    def _partial_loop(self) -> None:
        stt = self.translator.translator.stt
        language = lambda: self.translator.translator.resolver.stt_language  # noqa: E731
        while self._running.is_set():
            self._partial_event.wait(timeout=0.5)
            self._partial_event.clear()
            audio = self._partial_slot
            if audio is None or self._busy.is_set():  # finals always win the GPU
                continue
            try:
                t = stt.transcribe(audio, language(), fast=True)
            except Exception:
                continue
            if t.text and self._partial_slot is not None:
                self._emit("partial", text=t.text)

    def _emit(self, kind: str, **data) -> None:
        try:
            self.on_event({"type": kind, "direction": self.name, **data})
        except Exception:
            log.exception("event callback failed")
