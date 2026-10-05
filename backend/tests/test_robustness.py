"""Regressions found in the full code review: each test pins one bug that reached users."""

import asyncio
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import InvalidSettingError, Settings, settings_path
from app.engine.session import SessionError, TranslationSession
from app.main import Engine, _offer, create_app
from app.pipeline.factory import Providers, direction_languages
from app.pipeline.language import LanguageResolver
from app.pipeline.speech_translator import SpeechTranslator
from app.pipeline.voice_translator import VoiceTranslator
from app.services.audio.file_source import FileAudioSource
from app.services.stt.base import SpeechToTextProvider, Transcript
from app.utils.text import split_sentences, word_count
from tests.test_phase2_pipeline import FakeMT, FakeSTT
from tests.test_phase6_live import wait_for
from tests.test_phase6_segmenter import LoudnessVAD, signal
from tests.test_phase9_session import FakePlayer, FakeVirtualMic, VoicedTTS


# ---- "other language = auto" -------------------------------------------------------------
class DetectingSTT(SpeechToTextProvider):
    """Arabic when told Arabic; with auto-detect it 'hears' a confident German sentence."""

    def load(self):
        pass

    def transcribe(self, audio, language=None, *, fast=False):
        if language == "ar":
            return Transcript("أنا أريد أن أشرح لك المشروع.", "ar", 0.99, len(audio) / 16_000)
        return Transcript("Ich habe eine Frage zum Projekt.", "de", 0.97, len(audio) / 16_000)


def test_auto_other_language_can_start_and_follows_the_detected_language():
    """Regression: choosing 'Auto-detect' for the other side made every start fail."""
    assert direction_languages(Settings(other_language="auto"), "outgoing").follows_other
    them = signal(("s", 800), ("q", 4000))
    me = signal(("q", 2500), ("s", 800), ("q", 1500))  # I answer after they've been heard
    events = []
    session = TranslationSession(
        Settings(my_language="ar", other_language="auto", live_subtitles=False), on_event=events.append,
        providers=Providers(DetectingSTT(), FakeMT(), VoicedTTS()),
        mic=FileAudioSource(me, realtime=True), meeting=FileAudioSource(them, realtime=True),
        outgoing_sink=FakeVirtualMic(), headphones=FakePlayer(), vad_factory=LoudnessVAD)
    session.start()
    assert wait_for(lambda: len(session.history) == 2, timeout=15)
    session.stop()
    by = {m.speaker: m for m in session.history}
    assert by["other"].source_language == "de" and by["other"].target_language == "ar"
    assert by["me"].target_language == "de"  # followed what they were heard speaking


def test_auto_target_falls_back_to_english_until_someone_is_heard():
    st = SpeechTranslator(FakeSTT("مرحبا بك.", "ar", 0.99), FakeMT(), source="ar", target="en",
                          follow_target=lambda: None)
    assert st.process(np.zeros(16_000, np.float32)).translated_text == "<en>مرحبا بك."


def test_own_language_spoken_by_them_is_shown_not_respoken():
    st = SpeechTranslator(FakeSTT("مرحبا يا صديقي.", "ar", 0.99), FakeMT(), source="auto", target="ar-EG")
    got = []
    r = VoiceTranslator(st, VoicedTTS()).process(np.zeros(16_000, np.float32), got.append)
    assert r.same_language and r.text_only and got == []


# ---- controls ------------------------------------------------------------------------------
def _session(**kw):
    s = Settings(my_language="ar", other_language="de", live_subtitles=False)
    quiet = signal(("q", 500))
    return TranslationSession(s, providers=Providers(DetectingSTT(), FakeMT(), VoicedTTS()),
                              mic=kw.get("mic") or FileAudioSource(quiet, realtime=True, tail_silence_s=30),
                              meeting=kw.get("meeting") or FileAudioSource(quiet, realtime=True, tail_silence_s=30),
                              outgoing_sink=FakeVirtualMic(), headphones=FakePlayer(), vad_factory=LoudnessVAD,
                              on_event=kw.get("on_event"))


def test_resume_does_not_unmute_a_muted_mic():
    session = _session()
    session.start()
    try:
        session.set_mic_muted(True)
        session.set_paused(True)
        assert session.outgoing.muted and session.incoming.muted
        session.set_paused(False)
        assert session.outgoing.muted and not session.incoming.muted  # mic stays muted
        session.set_mic_muted(False)
        assert not session.outgoing.muted
    finally:
        session.stop()


class BrokenSource:
    def __init__(self):
        self.started = self.stopped = 0

    def start(self):
        raise RuntimeError("helper missing")

    def stop(self):
        self.stopped += 1

    def read_frame(self, timeout=1.0):
        return None


class TrackingSource(FileAudioSource):
    def __init__(self):
        super().__init__(signal(("q", 500)), realtime=True, tail_silence_s=30)
        self.stopped = False

    def stop(self):
        self.stopped = True


def test_failed_start_releases_the_microphone():
    """Regression: if meeting capture failed, the mic and its threads kept running."""
    mic = TrackingSource()
    session = _session(mic=mic, meeting=BrokenSource())
    with pytest.raises(SessionError) as err:
        session.start()
    assert err.value.code == "engine" and session.state == "idle"
    assert mic.stopped and session.outgoing is None and session._headphones.closed


def test_missing_saved_device_falls_back_to_default():
    events = []
    session = _session(on_event=events.append)

    def open_device(name):
        if name:
            raise LookupError(f"No output device matching {name!r}")
        return "default-device"

    assert session._open_with_fallback("output_device", open_device, "USB Headset") == "default-device"
    assert any(e["type"] == "warning" and e["code"] == "device_missing" for e in events)


class FlakySource:
    """Fails a few reads (unplugged), then recovers."""

    def __init__(self):
        self.fail, self.restarts = 3, 0

    def start(self):
        pass

    def stop(self):
        self.restarts += 1

    def read_frame(self, timeout=1.0):
        if self.fail:
            self.fail -= 1
            raise OSError("device unplugged")
        time.sleep(0.01)
        return np.zeros(320, np.float32)


def test_audio_source_failure_is_reported_once_and_recovers(monkeypatch):
    from app.pipeline.live import LiveDirection
    from app.pipeline.segmenter import UtteranceSegmenter

    monkeypatch.setattr(LiveDirection, "RECOVER_AFTER_S", (0.05,))
    events = []
    src = FlakySource()
    vt = VoiceTranslator(SpeechTranslator(DetectingSTT(), FakeMT(), source="ar", target="de"), VoicedTTS())
    d = LiveDirection("outgoing", src, UtteranceSegmenter(LoudnessVAD()), vt, lambda c: None,
                      on_event=events.append, partials=False)
    d.start()
    assert wait_for(lambda: any(e["type"] == "recovered" for e in events), timeout=5)
    d.stop()
    assert [e["type"] for e in events].count("error") == 1 and src.restarts >= 3


# ---- settings ------------------------------------------------------------------------------
def test_corrupted_settings_file_does_not_stop_the_app():
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"speech_speed": "fast", "latency_mode": "warp", "my_language": "fr"}), "utf-8")
    s = Settings.from_env()
    assert (s.speech_speed, s.latency_mode, s.my_language) == (1.0, "balanced", "fr")
    path.write_text("[1, 2", "utf-8")
    assert Settings.from_env().my_language == "ar"


def test_invalid_settings_are_rejected_up_front():
    with pytest.raises(InvalidSettingError):
        Settings().updated(latency_mode="warp")
    with pytest.raises(InvalidSettingError):
        Settings().updated(speech_speed=9)
    with TestClient(create_app(Engine(Settings()), "t")) as c:
        h = {"x-token": "t"}
        assert c.put("/api/settings", headers=h, json={"latency_mode": "warp"}).status_code == 400
        assert c.put("/api/settings", headers=h, json={"speech_speed": "fast"}).status_code == 400
        assert c.put("/api/settings", headers=h, json={"my_language": "auto"}).status_code == 400
        assert c.put("/api/settings", headers=h, json={"other_language": "auto"}).status_code == 200


# ---- text & language -----------------------------------------------------------------------
@pytest.mark.parametrize("text,dropped", [
    ("أنا بحب الموسيقى جدًا", False), ("قلت سبحان الله وبحمده", False), ("موسيقى", True),
    ("[موسيقى]", True), ("(Musik)", True), ("♪ ♪", True), ("شكرا للمشاهدة", True), ("Thank you.", False),
])
def test_hallucination_filter_keeps_real_sentences(text, dropped):
    """Regression: any short sentence mentioning 'music' in Arabic was silently deleted."""
    from types import SimpleNamespace

    from app.providers.stt.faster_whisper_provider import is_hallucination

    assert is_hallucination(SimpleNamespace(text=text, no_speech_prob=0.0, avg_logprob=-0.2)) is dropped


def test_chinese_and_japanese_are_split_and_trusted():
    assert split_sentences("你好。我是学生！你呢？") == ["你好。", "我是学生！", "你呢？"]
    assert word_count("我想问一个问题") >= 2 and word_count("ok") == 1
    r = LanguageResolver("auto", hint="de")
    assert r.resolve("zh", 0.95, "我想问一个问题").reason == "confident"


# ---- offline & events ----------------------------------------------------------------------
def test_voice_list_offline_fails_fast_and_is_not_retried_immediately(tmp_path, monkeypatch):
    """Regression: offline, the language list waited ~8 minutes for voices.json."""
    from app.providers.tts import piper_provider

    calls = []

    def offline(url, dest, **kw):
        calls.append(kw)
        raise ConnectionError("no internet")

    monkeypatch.setattr(piper_provider, "download", offline)
    t0 = time.perf_counter()
    # The language list endpoint makes a new provider per request: the memory must outlive it.
    for _ in range(3):
        assert piper_provider.PiperProvider(voices_dir=tmp_path).downloadable_languages() == set()
    assert time.perf_counter() - t0 < 1 and len(calls) == 1 and calls[0]["retries"] <= 3


def test_slow_ui_client_drops_oldest_events():
    q = asyncio.Queue(maxsize=2)
    for i in range(5):
        _offer(q, str(i))
    assert [q.get_nowait(), q.get_nowait()] == ["3", "4"]
