import os
import time
from pathlib import Path

import numpy as np
import pytest

from app.core.config import Settings
from app.engine.session import SessionError, TranslationSession
from app.pipeline.factory import Providers
from app.services.audio.file_source import FileAudioSource
from app.services.stt.base import SpeechToTextProvider, Transcript
from app.services.tts.base import AudioChunk, TextToSpeechProvider, Voice
from tests.test_phase2_pipeline import FakeMT
from tests.test_phase6_live import wait_for
from tests.test_phase6_segmenter import LoudnessVAD, signal

FIXTURES = Path(__file__).parent / "fixtures"


class ByLanguageSTT(SpeechToTextProvider):
    """Returns Arabic text when asked for Arabic, German when asked for German."""

    def load(self):
        pass

    def transcribe(self, audio, language=None, *, fast=False):
        text = {"ar": "أنا أريد أن أشرح لك المشروع.", "de": "Ich habe eine Frage."}[language]
        return Transcript(text, language, 0.99, len(audio) / 16_000)


class VoicedTTS(TextToSpeechProvider):
    def load(self):
        pass

    def voices(self, language=None):
        return [Voice("v", (language or "x").split("-")[0], "v", None, "medium")]

    def synthesize_stream(self, text, language, *, voice=None, speed=1.0):
        yield AudioChunk(np.full(1600, 0.1, np.float32), 16_000)


class FakePlayer:
    def __init__(self):
        self.chunks, self.queued_seconds, self.gain, self.started, self.closed = [], 0.0, 1.0, False, False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def enqueue(self, chunk):
        self.chunks.append(chunk)

    def duck(self, gain=0.3):
        self.gain = gain

    def unduck(self):
        self.gain = 1.0


class FakeVirtualMic:
    mic_name = "CABLE Output"

    def __init__(self):
        self.chunks, self.speaking, self.stopped = [], False, 0

    def __call__(self, chunk):
        self.chunks.append(chunk)

    def stop(self):
        self.stopped += 1


def make_session(mic_audio, meeting_audio, **settings):
    events = []
    s = Settings(my_language="ar", other_language="de", **settings)
    providers = Providers(ByLanguageSTT(), FakeMT(), VoicedTTS())
    vmic, phones = FakeVirtualMic(), FakePlayer()
    session = TranslationSession(s, on_event=events.append, providers=providers,
                                 mic=FileAudioSource(mic_audio), meeting=FileAudioSource(meeting_audio),
                                 outgoing_sink=vmic, headphones=phones, vad_factory=LoudnessVAD)
    return session, events, vmic, phones


def test_two_way_conversation_routes_each_side_correctly():
    me = signal(("s", 800), ("q", 1500))
    them = signal(("q", 1500), ("s", 800), ("q", 1500))
    session, events, vmic, phones = make_session(me, them, live_subtitles=False)
    session.start()
    assert wait_for(lambda: len(session.history) == 2)
    session.stop()
    mine, theirs = sorted(session.history, key=lambda m: m.speaker)
    assert (mine.speaker, mine.translated_text) == ("me", "<de>أنا أريد أن أشرح لك المشروع.")
    assert (theirs.speaker, theirs.translated_text) == ("other", "<ar>Ich habe eine Frage.")
    assert len(vmic.chunks) == 1 and len(phones.chunks) == 1  # German -> meeting, Arabic -> me
    ready = next(e for e in events if e["type"] == "ready")
    assert ready["mic_for_meeting"] == "CABLE Output"
    assert [e["state"] for e in events if e["type"] == "status" and "direction" not in e][-1] == "idle"
    assert phones.closed


def test_they_interrupt_my_translation():
    me = signal(("s", 800), ("q", 3000))
    them = signal(("q", 1600), ("s", 800), ("q", 1200))
    session, events, vmic, phones = make_session(me, them, live_subtitles=False)
    vmic.speaking = True  # my translation is still being read out when they start
    session.start()
    assert wait_for(lambda: any(e["type"] == "interrupted" for e in events))
    session.stop()
    assert vmic.stopped >= 1


def test_paused_session_translates_nothing():
    session, events, vmic, phones = make_session(signal(("q", 500), ("s", 800), ("q", 1500)),
                                                 signal(("q", 500), ("s", 800), ("q", 1500)), live_subtitles=False)
    session._mic.realtime = session._meeting.realtime = True
    session.start()
    session.set_paused(True)
    time.sleep(1.5)
    session.stop()
    assert not session.history


def test_auto_as_my_language_is_rejected():
    session, events, *_ = make_session(signal(("q", 100)), signal(("q", 100)))
    session.settings = Settings(my_language="auto")
    with pytest.raises(SessionError):
        session.start()
    assert session.state == "idle"


def test_replay_last_incoming_translation():
    session, events, vmic, phones = make_session(signal(("q", 2500)), signal(("s", 800), ("q", 1500)),
                                                 live_subtitles=False)
    session.start()
    assert wait_for(lambda: len(session.history) == 1)
    before = len(phones.chunks)
    assert session.replay_last("other")
    session.stop()
    assert len(phones.chunks) == before + 1


@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1")
def test_real_models_two_way():
    """Arabic speaker on the mic, German speaker in the meeting, real models end to end."""
    events = []
    s = Settings(my_language="ar", other_language="de")
    vmic, phones = FakeVirtualMic(), FakePlayer()
    session = TranslationSession(
        s, on_event=events.append,
        mic=FileAudioSource(FIXTURES / "ar_explain_project.wav", realtime=True),
        meeting=FileAudioSource(FIXTURES / "de_question.wav", realtime=True, lead_silence_s=7.5),
        outgoing_sink=vmic, headphones=phones)
    session.start()
    assert wait_for(lambda: len(session.history) == 2, timeout=40)
    session.stop()
    by = {m.speaker: m for m in session.history}
    assert "Projekt" in by["me"].translated_text and vmic.chunks
    assert "سؤال" in by["other"].translated_text and phones.chunks
