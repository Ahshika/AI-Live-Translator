"""Turn a continuous audio stream into utterances using VAD probabilities.

State machine (per 32 ms window):
    SILENCE --(prob >= start_threshold for min_speech_ms)--> SPEECH      emit SpeechStarted
    SPEECH  --(prob <  end_threshold  for end_silence_ms)--> SILENCE     emit UtteranceEnded
    SPEECH  --(utterance longer than max_utterance_s)------> cut         emit UtteranceEnded(forced)

* Hysteresis (start 0.5 / end 0.35) stops flapping on borderline frames.
* pre_roll keeps ~300 ms before the detected start so the first syllable isn't clipped.
* end_silence_ms is THE latency/accuracy dial: shorter = faster but may cut mid-thought
  ("أنا عايز... أروح المطار" becomes two translations). It's exposed as the latency mode.
* Forced cuts happen at the quietest recent window, not mid-word.
* Partial snapshots (for live subtitles) are emitted every partial_interval_s while speaking.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Union

import numpy as np

from app.services.audio.format import SAMPLE_RATE
from app.services.vad.base import VoiceActivityDetector


@dataclass(frozen=True)
class SpeechStarted:
    t: float  # stream time, seconds


@dataclass(frozen=True)
class PartialAudio:
    audio: np.ndarray  # everything said so far in this utterance
    t: float


@dataclass(frozen=True)
class UtteranceEnded:
    audio: np.ndarray
    t: float
    forced: bool = False


SegmentEvent = Union[SpeechStarted, PartialAudio, UtteranceEnded]

LATENCY_MODES = {  # end-of-utterance silence in ms
    "fast": 400,
    "balanced": 600,
    "accurate": 900,
}


@dataclass
class SegmenterConfig:
    start_threshold: float = 0.5
    end_threshold: float = 0.35
    min_speech_ms: int = 160
    end_silence_ms: int = 600
    pre_roll_ms: int = 300
    max_utterance_s: float = 15.0
    min_utterance_ms: int = 300  # shorter blips (coughs, clicks) are dropped
    partial_interval_s: float | None = 1.0  # None = no partials

    @classmethod
    def for_mode(cls, mode: str, **kw) -> "SegmenterConfig":
        return cls(end_silence_ms=LATENCY_MODES[mode], **kw)


class UtteranceSegmenter:
    def __init__(self, vad: VoiceActivityDetector, config: SegmenterConfig | None = None):
        self.vad, self.cfg = vad, config or SegmenterConfig()
        self.win = vad.window_samples
        self._ms_per_win = self.win * 1000 / SAMPLE_RATE
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self._pending = np.zeros(0, dtype=np.float32)
        # pre-roll = the voiced windows that triggered the start + pre_roll_ms of audio before them
        self._pre = deque(maxlen=max(1, int((self.cfg.pre_roll_ms + self.cfg.min_speech_ms) / self._ms_per_win)))
        self._speech: list[np.ndarray] = []
        self._win_energy: list[float] = []
        self._in_speech = False
        self._voiced_run = 0
        self._silent_run = 0
        self._samples_seen = 0
        self._last_partial_len = 0
        self.paused = False

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def _now(self) -> float:
        return self._samples_seen / SAMPLE_RATE

    def feed(self, audio: np.ndarray) -> list[SegmentEvent]:
        """Feed any amount of canonical audio; returns the events it produced."""
        events: list[SegmentEvent] = []
        self._pending = np.concatenate([self._pending, audio])
        while len(self._pending) >= self.win:
            window, self._pending = self._pending[:self.win], self._pending[self.win:]
            self._samples_seen += self.win
            events.extend(self._step(window))
        return events

    def _step(self, window: np.ndarray) -> list[SegmentEvent]:
        if self.paused:  # e.g. half-duplex echo guard while our own TTS plays on speakers
            if self._in_speech:
                self._close()
            self._pre.clear()
            return []
        prob = self.vad(window)
        if not self._in_speech:
            self._pre.append(window)
            self._voiced_run = self._voiced_run + 1 if prob >= self.cfg.start_threshold else 0
            if self._voiced_run * self._ms_per_win >= self.cfg.min_speech_ms:
                self._in_speech, self._silent_run = True, 0
                self._speech = list(self._pre)
                self._win_energy = [float(np.mean(w * w)) for w in self._speech]
                self._pre.clear()
                self._last_partial_len = 0
                return [SpeechStarted(self._now())]
            return []

        self._speech.append(window)
        self._win_energy.append(float(np.mean(window * window)))
        self._silent_run = self._silent_run + 1 if prob < self.cfg.end_threshold else 0
        if self._silent_run * self._ms_per_win >= self.cfg.end_silence_ms:
            # Drop most of the trailing silence; keep a little so the last word isn't cut.
            keep = len(self._speech) - self._silent_run + max(1, int(150 / self._ms_per_win))
            self._speech = self._speech[:keep]
            return self._close()
        if len(self._speech) * self.win >= self.cfg.max_utterance_s * SAMPLE_RATE:
            return self._force_cut()
        if self.cfg.partial_interval_s and self._silent_run == 0:  # no partials during pauses
            n = len(self._speech) * self.win
            if n - self._last_partial_len >= self.cfg.partial_interval_s * SAMPLE_RATE:
                self._last_partial_len = n
                return [PartialAudio(np.concatenate(self._speech), self._now())]
        return []

    def _close(self, forced: bool = False) -> list[SegmentEvent]:
        audio = np.concatenate(self._speech) if self._speech else np.zeros(0, np.float32)
        self._in_speech, self._speech, self._win_energy = False, [], []
        self._voiced_run = self._silent_run = 0
        if len(audio) < self.cfg.min_utterance_ms * SAMPLE_RATE / 1000:
            return []
        return [UtteranceEnded(audio, self._now(), forced)]

    def _force_cut(self) -> list[SegmentEvent]:
        # Cut at the quietest window in the last ~2 s so we don't split a word.
        look = min(len(self._win_energy), int(2000 / self._ms_per_win))
        tail = self._win_energy[-look:]
        cut = len(self._speech) - look + int(np.argmin(tail)) + 1
        rest = self._speech[cut:]
        self._speech = self._speech[:cut]
        events = self._close(forced=True)
        # Speech continues: start the next utterance with what came after the cut.
        self._in_speech, self._speech = True, rest
        self._win_energy = [float(np.mean(w * w)) for w in rest]
        self._last_partial_len = 0
        return events
