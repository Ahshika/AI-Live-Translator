import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from app.pipeline.live import LiveDirection
from app.pipeline.segmenter import SegmenterConfig, UtteranceSegmenter
from app.pipeline.speech_translator import SpeechTranslator
from app.pipeline.voice_translator import VoiceTranslator
from app.services.audio.file_source import FileAudioSource
from tests.test_phase2_pipeline import FakeMT, FakeSTT
from tests.test_phase4_voice import FakeTTS
from tests.test_phase6_segmenter import LoudnessVAD, signal

FIXTURES = Path(__file__).parent / "fixtures"


def wait_for(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def make_live(audio, stt=None, **kw):
    st = SpeechTranslator(stt or FakeSTT("مرحبا بك.", "ar", 0.99), FakeMT(), source="ar", target="de")
    vt = VoiceTranslator(st, FakeTTS())
    events, chunks = [], []
    seg = UtteranceSegmenter(LoudnessVAD(), SegmenterConfig(partial_interval_s=None))
    live = LiveDirection("outgoing", FileAudioSource(audio), seg, vt, chunks.append,
                         on_event=events.append, partials=False, **kw)
    return live, events, chunks


def finals(events):
    return [e for e in events if e["type"] == "final"]


def test_each_spoken_utterance_is_translated_and_spoken_once():
    audio = signal(("s", 1000), ("q", 1200), ("s", 800), ("q", 1200))
    live, events, chunks = make_live(audio)
    live.start()
    assert wait_for(lambda: len(finals(events)) == 2)
    live.stop()
    f = finals(events)
    assert [e["translated_text"] for e in f] == ["<de>مرحبا بك."] * 2
    assert all(e["spoken"] and e["latency_ms"] >= 0 for e in f)
    assert len(chunks) == 2
    assert [e["type"] for e in events].count("speech_started") == 2
    assert events[-1] == {"type": "status", "direction": "outgoing", "state": "stopped"}


def test_echo_guard_ignores_audio_while_paused():
    audio = signal(("s", 1000), ("q", 1200))
    live, events, chunks = make_live(audio, pause_when=lambda: True)
    live.start()
    time.sleep(1.0)
    live.stop()
    assert finals(events) == [] and chunks == []


def test_mute_ignores_speech():
    live, events, chunks = make_live(signal(("s", 1000), ("q", 1200)))
    live.muted = True
    live.start()
    time.sleep(1.0)
    live.stop()
    assert finals(events) == []


class SlowSTT(FakeSTT):
    def transcribe(self, audio, language=None, *, fast=False):
        time.sleep(0.4)
        return super().transcribe(audio, language)


def test_backlog_turns_oldest_into_subtitles_instead_of_falling_behind():
    # 5 quick utterances while the translator needs 0.4 s each
    audio = np.concatenate([signal(("s", 400), ("q", 700)) for _ in range(5)] + [signal(("q", 800))])
    live, events, chunks = make_live(audio, stt=SlowSTT("جملة.", "ar", 0.99))
    live.start()
    assert wait_for(lambda: len(finals(events)) == 5, timeout=15)
    live.stop()
    f = finals(events)
    assert any(e["text_only"] for e in f), "backlog should have downgraded something to text"
    assert len(chunks) == sum(e["spoken"] for e in f) < 5


def test_errors_are_reported_and_the_direction_keeps_running():
    class Boom(FakeSTT):
        calls = 0

        def transcribe(self, audio, language=None, *, fast=False):
            Boom.calls += 1
            if Boom.calls == 1:
                raise RuntimeError("GPU hiccup")
            return super().transcribe(audio, language)

    audio = signal(("s", 800), ("q", 1200), ("s", 800), ("q", 1200))
    live, events, chunks = make_live(audio, stt=Boom("نعم.", "ar", 0.99))
    live.start()
    assert wait_for(lambda: len(finals(events)) == 1)
    live.stop()
    assert any(e["type"] == "error" and "GPU hiccup" in e["message"] for e in events)


@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1")
def test_real_models_live_from_arabic_file():
    from app.core.config import Settings
    from app.pipeline.factory import build_voice_translator
    from app.services.vad.silero import SileroVAD

    vt = build_voice_translator(Settings(my_language="ar", other_language="de"), "outgoing")
    events, chunks = [], []
    src = FileAudioSource(FIXTURES / "ar_explain_project.wav", realtime=True)
    live = LiveDirection("outgoing", src, UtteranceSegmenter(SileroVAD()), vt, chunks.append,
                         on_event=events.append)
    live.start()
    assert wait_for(lambda: finals(events), timeout=20)
    live.stop()
    f = finals(events)[0]
    assert "Projekt" in f["translated_text"] and f["spoken"]
    assert f["latency_ms"] < 3000
    assert any(e["type"] == "partial" for e in events)  # live subtitles while "speaking"


def test_mute_cancels_utterances_still_waiting():
    audio = np.concatenate([signal(("s", 400), ("q", 700)) for _ in range(3)] + [signal(("q", 500))])
    live, events, chunks = make_live(audio, stt=SlowSTT("جملة.", "ar", 0.99))
    live.start()
    assert wait_for(lambda: any(e["type"] == "final" for e in events), timeout=10)
    live.muted = True
    time.sleep(1.5)
    live.stop()
    assert len(finals(events)) < 3
