"""Universal Translator Mode: a full two-way translated conversation.

    your mic ──► [outgoing: my language → their language] ──► virtual mic ──► any meeting app
    meeting audio (app or all-but-us) ──► [incoming: their language → mine] ──► your headphones

Interruption policy ("smart"):
  * they start talking while your translation is still being spoken to them
        -> stop it: they've clearly moved on, and talking over them is rude.
  * you start talking while their translation plays in your headphones
        -> duck it (30% volume) so you can still catch the end; restore when you finish.
Echo policy: incoming capture never contains our own process' audio (see process_capture),
and without headphones the mic is ignored while a translation plays on your speakers.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from app.core import languages
from app.core.config import Settings
from app.pipeline.factory import Providers, build_voice_translator
from app.pipeline.live import AudioSource, LiveDirection
from app.pipeline.segmenter import SegmenterConfig, UtteranceSegmenter
from app.services.audio.playback import AudioPlayer
from app.services.tts.base import AudioChunk

log = logging.getLogger(__name__)
EventCallback = Callable[[dict[str, Any]], None]


@dataclass
class Message:
    """One line of the conversation (what the UI shows as history)."""

    id: int
    speaker: str  # "me" | "other"
    source_language: str
    source_text: str
    target_language: str
    translated_text: str
    timestamp: float
    latency_ms: float | None
    spoken: bool


class SessionError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class TranslationSession:
    def __init__(self, settings: Settings, *, on_event: EventCallback | None = None,
                 providers: Providers | None = None,
                 mic: AudioSource | None = None, meeting: AudioSource | None = None,
                 outgoing_sink: Any = None, headphones: AudioPlayer | None = None,
                 vad_factory: Callable[[], Any] | None = None):
        """Every device can be injected (tests, previews); by default real devices are opened."""
        self.settings = settings
        self._on_event = on_event or (lambda e: None)
        self._providers = providers
        self._mic, self._meeting = mic, meeting
        self._outgoing_sink, self._headphones = outgoing_sink, headphones
        self._vad_factory = vad_factory
        self.outgoing: LiveDirection | None = None
        self.incoming: LiveDirection | None = None
        self.history: deque[Message] = deque(maxlen=500)
        self._next_id = 1
        self._lock = threading.Lock()
        self.state = "idle"
        self.mic_name_for_meeting: str | None = None

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self.state in ("starting", "running"):
            return
        self._set_state("starting")
        try:
            self._start()
        except Exception:
            self._set_state("idle")
            self._close_devices()
            raise
        self._set_state("running")

    def _start(self) -> None:
        s = self.settings
        if s.my_language == languages.AUTO:
            raise SessionError("language", "Choose your own language (it can't be Auto)")
        self._emit("status", state="loading_models")
        if self._providers is None:
            self._providers = Providers.load(s)

        # Devices (lazy imports keep tests free of real hardware)
        if self._headphones is None:
            self._headphones = AudioPlayer(s.output_device)
        self._headphones.start()
        if self._outgoing_sink is None:
            from app.services.audio.virtual_mic import VirtualMicrophone, VirtualMicUnavailableError

            try:
                monitor = self._headphones if s.hear_my_translation else None
                self._outgoing_sink = VirtualMicrophone(monitor=monitor)
            except VirtualMicUnavailableError as exc:
                raise SessionError("virtual_mic_missing", str(exc)) from exc
        if hasattr(self._outgoing_sink, "start"):
            self._outgoing_sink.start()
        self.mic_name_for_meeting = getattr(self._outgoing_sink, "mic_name", None)
        if self._mic is None:
            from app.services.audio.capture import MicrophoneCapture
            from app.services.audio.devices import find_device

            from app.services.audio.virtual_mic import is_virtual_device

            if s.input_device and is_virtual_device(s.input_device):
                raise SessionError("mic_is_virtual",
                                   "The virtual cable can't be your microphone — choose your real mic in Settings")
            self._mic = MicrophoneCapture(device=find_device(s.input_device) if s.input_device else None)
        if self._meeting is None:
            from app.services.audio.process_capture import MeetingAudioUnavailableError, ProcessAudioCapture

            app = None if s.meeting_app in ("system", "", None) else s.meeting_app
            try:
                self._meeting = ProcessAudioCapture(app)
            except MeetingAudioUnavailableError as exc:
                code = "meeting_app_not_running" if "not running" in str(exc) else "meeting_capture"
                raise SessionError(code, str(exc)) from exc

        # Live subtitles re-run speech recognition every second: fine on a GPU, but on a CPU
        # they'd steal the time the real translation needs.
        partials = s.live_subtitles and getattr(self._providers.stt, "device", "cuda") != "cpu"
        out_vt = build_voice_translator(s, "outgoing", providers=self._providers)
        in_vt = build_voice_translator(s, "incoming", providers=self._providers)

        def segmenter() -> UtteranceSegmenter:
            if self._vad_factory:
                vad = self._vad_factory()
            else:
                from app.services.vad.silero import SileroVAD

                vad = SileroVAD()
            return UtteranceSegmenter(vad, SegmenterConfig.for_mode(
                s.latency_mode, partial_interval_s=1.0 if partials else None))

        echo_guard = None if s.headphones else (lambda: self._headphones.queued_seconds > 0)
        self.outgoing = LiveDirection("outgoing", self._mic, segmenter(), out_vt, self._outgoing_sink,
                                      on_event=self._direction_event, pause_when=echo_guard,
                                      partials=partials)
        self.incoming = LiveDirection("incoming", self._meeting, segmenter(), in_vt, self._headphones.enqueue,
                                      on_event=self._direction_event, partials=partials)
        self.outgoing.start()
        self.incoming.start()
        self._emit("ready", mic_for_meeting=self.mic_name_for_meeting,
                   outgoing_voice=out_vt.speak, incoming_voice=in_vt.speak,
                   my_language=s.my_language, other_language=s.other_language)

    def stop(self) -> None:
        if self.state == "idle":
            return
        self._set_state("stopping")
        for d in (self.outgoing, self.incoming):
            if d is not None:
                d.stop()
        self.outgoing = self.incoming = None
        self._close_devices()
        self._set_state("idle")

    def _close_devices(self) -> None:
        for dev in (self._outgoing_sink, self._headphones):
            close = getattr(dev, "close", None)
            if close:
                try:
                    close()
                except Exception:
                    log.exception("closing device failed")

    # -- controls ------------------------------------------------------------
    def set_mic_muted(self, muted: bool) -> None:
        if self.outgoing:
            self.outgoing.muted = muted

    def set_paused(self, paused: bool) -> None:
        """Pause = stop translating in both directions (streams stay open; instant resume)."""
        for d in (self.outgoing, self.incoming):
            if d:
                d.muted = paused
        self._set_state("paused" if paused else "running")

    def replay_last(self, speaker: str = "other") -> bool:
        """Say the last translation again (incoming -> your headphones, outgoing -> the meeting)."""
        with self._lock:
            msg = next((m for m in reversed(self.history) if m.speaker == speaker), None)
        if msg is None or self._providers is None:
            return False
        sink = self._headphones.enqueue if speaker == "other" else self._outgoing_sink
        tts = self._providers.tts
        try:
            for chunk in tts.synthesize_stream(msg.translated_text, msg.target_language,
                                               speed=self.settings.speech_speed):
                sink(chunk)
        except Exception as exc:
            self._emit("error", code="replay", message=str(exc))
            return False
        return True

    # -- events --------------------------------------------------------------
    def _direction_event(self, event: dict) -> None:
        kind, direction = event["type"], event["direction"]
        if kind == "speech_started" and self.settings.interruptions == "smart":
            if direction == "incoming" and getattr(self._outgoing_sink, "speaking", False):
                self._outgoing_sink.stop()  # they're talking: stop reading my old sentence to them
                self._emit("interrupted", direction="outgoing")
            elif direction == "outgoing" and self._headphones.queued_seconds > 0:
                self._headphones.duck()
        if kind == "final":
            if direction == "outgoing":
                self._headphones.unduck()
            with self._lock:
                msg = Message(self._next_id, "me" if direction == "outgoing" else "other",
                              event["source_language"], event["source_text"], event["target_language"],
                              event["translated_text"], time.time(), event["latency_ms"], event["spoken"])
                self._next_id += 1
                self.history.append(msg)
            event = {**event, "message": msg.__dict__}
        self._emit_raw(event)

    def _set_state(self, state: str) -> None:
        self.state = state
        self._emit("status", state=state)

    def _emit(self, kind: str, **data) -> None:
        self._emit_raw({"type": kind, **data})

    def _emit_raw(self, event: dict) -> None:
        try:
            self._on_event(event)
        except Exception:
            log.exception("session event callback failed")


class NullSink:
    """Outgoing sink for previews without a virtual mic: discard (or forward) the audio."""

    def __init__(self, forward: Callable[[AudioChunk], None] | None = None):
        self.forward = forward
        self.speaking = False

    def __call__(self, chunk: AudioChunk) -> None:
        if self.forward:
            self.forward(chunk)

    def stop(self) -> None:
        pass
