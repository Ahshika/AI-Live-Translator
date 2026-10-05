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
from app.services.audio.format import rms_dbfs

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


class ModelGate:
    """Shared by both directions: real translations always go before live subtitles.

    Both directions use the same speech model (one at a time). Without this, the meeting
    side's subtitle refresh could hold the model while your finished sentence waits.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._finals = 0

    def add(self, n: int = 1) -> None:
        with self._lock:
            self._finals += n

    def done(self) -> None:
        with self._lock:
            self._finals = max(0, self._finals - 1)

    @property
    def finals_waiting(self) -> bool:
        return self._finals > 0


class LiveDirection:
    MAX_PENDING = 2
    RECOVER_AFTER_S = (1, 2, 5, 10)  # back-off between attempts to reopen a failed audio source

    def __init__(self, name: str, source: AudioSource, segmenter: UtteranceSegmenter,
                 translator: VoiceTranslator, sink: AudioSink, *, on_event: EventCallback | None = None,
                 pause_when: Callable[[], bool] | None = None, partials: bool = True,
                 conditioner: Any = None, gate: ModelGate | None = None,
                 sink_backlog: Callable[[], float] | None = None):
        """conditioner: optional mic clean-up (app.services.audio.enhance.Conditioner).
        gate: shared with the other direction so finals win the model over subtitles.
        sink_backlog: seconds of translated speech still waiting to be played."""
        self.gate = gate or ModelGate()
        self.sink_backlog = sink_backlog
        self.name, self.source, self.segmenter = name, source, segmenter
        self.translator, self.sink = translator, sink
        self.on_event = on_event or (lambda e: None)
        self.pause_when = pause_when  # e.g. echo guard: our own speaker is playing
        self.partials = partials
        self.conditioner = conditioner
        self._level_sum, self._level_n, self._level_peak = 0.0, 0, 0.0
        self._queue: queue.Queue[_Utterance | None] = queue.Queue()
        self._pending: list[_Utterance] = []
        self._pending_lock = threading.Lock()
        self._partial_slot: np.ndarray | None = None
        self._partial_event = threading.Event()
        self._busy = threading.Event()  # worker is using the models
        self._running = threading.Event()
        self._muted = False
        self._reset_segmenter = False
        self._threads: list[threading.Thread] = []
        self.stats = {"utterances": 0, "dropped_to_text": 0, "errors": 0}
        self.quiet_utterances = 0  # in a row; triggers a "your mic is very quiet" hint

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
        self._partial_slot = None
        self._reset_segmenter = True  # done on the capture thread, which owns the segmenter
        if value:  # what was waiting to be translated must not be spoken after "mute"/"pause"
            with self._pending_lock:
                for u in self._pending:
                    u.cancelled = True
        self._emit("status", state="muted" if value else "listening")

    # -- threads -------------------------------------------------------------
    def _capture_loop(self) -> None:
        failures = 0
        while self._running.is_set():
            try:
                frame = self.source.read_frame(timeout=0.5)
            except Exception as exc:  # device unplugged, capture helper died...
                failures += 1
                if failures == 1:  # tell the user once, not every second
                    log.warning("%s audio source failed: %s", self.name, exc)
                    self._emit("error", code="audio_source", message=str(exc))
                self._recover_source(failures)
                continue
            if failures:
                failures = 0
                self._emit("recovered")
            if self._reset_segmenter:
                self._reset_segmenter = False
                self.segmenter.reset()
            if frame is None:
                continue
            if self._muted:
                continue
            try:
                self._segment(frame)
            except Exception:  # never let one bad frame kill this direction's capture thread
                log.exception("%s segmenter failed", self.name)
                self.segmenter.reset()

    CATCH_UP_SPEED = 1.15
    LEVEL_EVERY = 6  # frames (120 ms) per level-meter update

    def _meter(self, frame: np.ndarray) -> None:
        self._level_sum += float(np.mean(frame * frame))
        self._level_peak = max(self._level_peak, float(np.max(np.abs(frame))))
        self._level_n += 1
        if self._level_n >= self.LEVEL_EVERY:
            mean = self._level_sum / self._level_n
            db = 10 * np.log10(mean) if mean > 0 else -100.0
            self._emit("level", db=round(max(db, -100.0), 1), clipping=self._level_peak >= 0.99,
                       speaking=self.segmenter.in_speech)
            self._level_sum, self._level_n, self._level_peak = 0.0, 0, 0.0

    def _segment(self, frame: np.ndarray) -> None:
        if self.conditioner is not None:
            frame = self.conditioner.frame(frame)
            if not self.segmenter.in_speech:
                self.conditioner.silence(frame)
        self._meter(frame)
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
                waited = 0 if ev.forced else ev.silence_ms / 1000
                self._enqueue(_Utterance(self._prepare(ev.audio), time.perf_counter() - waited, ev.forced))

    QUIET_DBFS = -45.0

    def _prepare(self, audio: np.ndarray) -> np.ndarray:
        """Mic clean-up for a finished utterance, plus the 'mic too quiet' check."""
        if rms_dbfs(audio) < self.QUIET_DBFS:
            self.quiet_utterances += 1
            if self.quiet_utterances == 3:
                self._emit("warning", code="mic_quiet")
        else:
            self.quiet_utterances = 0
        if self.conditioner is None:
            return audio
        try:
            return self.conditioner.utterance(audio)
        except Exception:
            log.exception("audio enhancement failed; using the raw audio")
            return audio

    def _recover_source(self, failures: int) -> None:
        """Wait a little, then reopen the source (a re-plugged headset, a restarted helper)."""
        delay = self.RECOVER_AFTER_S[min(failures, len(self.RECOVER_AFTER_S)) - 1]
        deadline = time.monotonic() + delay
        while self._running.is_set() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not self._running.is_set():
            return
        try:
            self.source.stop()
            self.source.start()
            self.segmenter.reset()
        except Exception as exc:
            log.info("%s audio source still unavailable: %s", self.name, exc)

    def _enqueue(self, utt: _Utterance) -> None:
        with self._pending_lock:
            waiting = [u for u in self._pending if u.speak]
            if len(waiting) >= self.MAX_PENDING:
                waiting[0].speak = False  # too far behind: oldest becomes subtitles only
                self.stats["dropped_to_text"] += 1
            self._pending.append(utt)
        self.gate.add()
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
                self.gate.done()
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
                self.gate.done()

    def _translate(self, utt: _Utterance) -> None:
        first_audio: list[float] = []

        def sink(chunk):
            if not first_audio:
                first_audio.append(time.perf_counter())
            self.sink(chunk)

        speak, speed = self._catch_up(utt)
        r = self.translator.process(utt.audio, sink, speak=speak, speed_factor=speed,
                                    reject=self.reject_text)
        if r.skipped:
            if r.rejected:
                self._emit("echo_dropped", text=r.transcript.text)
            return
        self.stats["utterances"] += 1
        latency = (first_audio[0] - utt.ended_at) * 1000 if first_audio else None
        log.info("%s: %.1fs speech -> stt %.0f ms, first audio %s ms after they stopped, total %.0f ms%s",
                 self.name, len(utt.audio) / 16_000, r.stt_ms,
                 f"{latency:.0f}" if latency is not None else "-", r.total_ms,
                 "" if speak else " (text only: running behind)")
        self._emit("final", source_language=r.language.code, target_language=r.target_language,
                   source_text=r.transcript.text, translated_text=r.translated_text,
                   spoken=bool(first_audio), text_only=r.text_only or not utt.speak,
                   same_language=r.same_language,
                   detected=r.language.detected, detection=r.language.reason,
                   latency_ms=latency, stt_ms=r.stt_ms, total_ms=r.total_ms, forced_cut=utt.forced)

    def _partial_loop(self) -> None:
        stt = self.translator.translator.stt
        language = lambda: self.translator.translator.resolver.stt_language  # noqa: E731
        while self._running.is_set():
            self._partial_event.wait(timeout=0.5)
            self._partial_event.clear()
            audio = self._partial_slot
            if audio is None or self.gate.finals_waiting:  # finals (either side) win the model
                continue
            try:
                if self.conditioner is not None:
                    audio = self.conditioner.partial(audio)
                t = stt.transcribe(audio, language(), fast=True)
            except Exception:
                continue
            if t.text and self._partial_slot is not None:
                self._emit("partial", text=t.text)

    # Translated speech already waiting to be played. Beyond these, we catch up first by
    # talking faster, then by showing a sentence as text instead of piling up audio — the
    # listener should never hear translations seconds after the conversation moved on.
    FASTER_AFTER_S = 2.5
    TEXT_ONLY_AFTER_S = 8.0
    reject_text: Callable[[str], bool] | None = None  # e.g. echo filter (set by the session)

    def _catch_up(self, utt: _Utterance) -> tuple[bool, float]:
        with self._pending_lock:
            waiting = any(u.speak and not u.cancelled for u in self._pending)
        backlog = 0.0
        if self.sink_backlog is not None:
            try:
                backlog = float(self.sink_backlog())
            except Exception:
                backlog = 0.0
        if backlog > self.TEXT_ONLY_AFTER_S:
            self.stats["dropped_to_text"] += 1
            return False, 1.0
        if backlog > self.FASTER_AFTER_S:
            return utt.speak, 1.25
        return utt.speak, self.CATCH_UP_SPEED if waiting else 1.0

    def _emit(self, kind: str, **data) -> None:
        try:
            self.on_event({"type": kind, "direction": self.name, **data})
        except Exception:
            log.exception("event callback failed")
