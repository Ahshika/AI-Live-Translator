"""Delay and overlapping voices: endpointing, model priority, playback backlog, echo."""

import time

import numpy as np

from app.core.config import Settings
from app.engine.echo import EchoFilter, normalize
from app.engine.session import TranslationSession
from app.pipeline.factory import Providers
from app.pipeline.live import LiveDirection, ModelGate
from app.pipeline.segmenter import SegmenterConfig, UtteranceEnded, UtteranceSegmenter
from app.pipeline.speech_translator import SpeechTranslator
from app.pipeline.voice_translator import VoiceTranslator
from app.services.audio.devices import AudioDevice
from app.services.audio.file_source import FileAudioSource
from app.services.audio.playback import AudioPlayer
from app.services.tts.base import AudioChunk
from tests.test_phase2_pipeline import FakeMT, FakeSTT
from tests.test_phase6_segmenter import LoudnessVAD, signal
from tests.test_phase9_session import FakePlayer, FakeVirtualMic, VoicedTTS


def _ends(audio, **cfg):
    seg = UtteranceSegmenter(LoudnessVAD(), SegmenterConfig(partial_interval_s=None, **cfg))
    out = []
    for i in range(0, len(audio), 320):
        out += [e for e in seg.feed(audio[i:i + 320]) if isinstance(e, UtteranceEnded)]
    return out


# ---- endpointing -------------------------------------------------------------------------------
def test_long_talk_is_cut_at_a_short_breath_instead_of_waiting():
    """Regression: someone who talks on with only short breaths was translated after 15 s."""
    talk = signal(("s", 4000), ("q", 350), ("s", 4000), ("q", 1000))  # a 350 ms breath after 4 s
    ends = _ends(talk)
    assert len(ends) == 2 and not ends[0].forced
    assert abs(len(ends[0].audio) / 16_000 - 4.15) < 0.3
    assert len(_ends(talk, adaptive_end=False)) == 1  # the old behaviour waited for 600 ms


def test_short_sentences_still_wait_for_the_full_pause():
    # "أنا عايز... أروح المطار": a 350 ms hesitation early on must not split the sentence
    assert len(_ends(signal(("s", 1200), ("q", 350), ("s", 1200), ("q", 1000)))) == 1


def test_nonstop_talk_is_cut_within_nine_seconds():
    ends = _ends(signal(("s", 20_000), ("q", 1000)))
    assert ends[0].forced and len(ends[0].audio) / 16_000 <= 9.1


def test_end_silence_is_reported_for_latency():
    assert _ends(signal(("s", 1000), ("q", 1000)))[0].silence_ms >= 600


# ---- model priority & backlog -------------------------------------------------------------------
class CountingSTT(FakeSTT):
    def __init__(self):
        super().__init__("Das ist ein ganz normaler Satz.", "de", 0.99)
        self.fast_calls = 0

    def transcribe(self, audio, language=None, *, fast=False):
        if fast:
            self.fast_calls += 1
        return super().transcribe(audio, language)


def _direction(gate=None, backlog=None, events=None):
    stt = CountingSTT()
    vt = VoiceTranslator(SpeechTranslator(stt, FakeMT(), source="de", target="ar"), VoicedTTS())
    d = LiveDirection("incoming", FileAudioSource(signal(("q", 100))), UtteranceSegmenter(LoudnessVAD()), vt,
                      lambda c: None, on_event=(events if events is not None else []).append,
                      gate=gate, sink_backlog=backlog)
    return d, stt


def test_subtitles_never_take_the_model_while_the_other_side_waits_for_a_translation():
    gate = ModelGate()
    d, stt = _direction(gate)
    d.start()
    try:
        gate.add()  # the OTHER direction has a finished sentence waiting
        d._partial_slot = np.zeros(16_000, np.float32)
        d._partial_event.set()
        time.sleep(0.3)
        assert stt.fast_calls == 0
        gate.done()
        d._partial_event.set()
        time.sleep(0.3)
        assert stt.fast_calls == 1
    finally:
        d.stop()


def test_playback_backlog_speeds_up_then_falls_back_to_text():
    from app.pipeline.live import _Utterance

    backlog = [0.0]
    d, _ = _direction(backlog=lambda: backlog[0])
    utt = _Utterance(np.zeros(1), 0.0, False)
    assert d._catch_up(utt) == (True, 1.0)
    backlog[0] = 4.0
    assert d._catch_up(utt) == (True, 1.25)
    backlog[0] = 12.0
    assert d._catch_up(utt) == (False, 1.0)


# ---- echo ---------------------------------------------------------------------------------------
def test_echo_filter_matches_what_was_just_played():
    f = EchoFilter()
    f.heard("Ich habe eine Frage zum Projekt.", "لديَّ سؤالٌ عن المشروع.")
    assert f.is_echo("لدي سؤال عن المشروع")  # same words, tashkeel/punctuation aside
    assert f.is_echo("ich habe eine frage zum projekt")
    assert not f.is_echo("أنا موافق ونبدأ بكرة الصبح")
    assert not f.is_echo("نعم")  # too short to judge
    assert normalize("إلى المشروعِ!") == "الى المشروع"


def test_echo_from_speakers_is_dropped_and_switches_to_speaker_mode():
    events = []
    s = Settings(my_language="ar", other_language="de", live_subtitles=False, headphones=True)
    session = TranslationSession(
        s, on_event=events.append, providers=Providers(FakeSTT("x", "ar", 0.9), FakeMT(), VoicedTTS()),
        mic=FileAudioSource(signal(("q", 100)), tail_silence_s=30, realtime=True),
        meeting=FileAudioSource(signal(("q", 100)), tail_silence_s=30, realtime=True),
        outgoing_sink=FakeVirtualMic(), headphones=FakePlayer(), vad_factory=LoudnessVAD)
    session.start()
    try:
        session._direction_event({"type": "final", "direction": "incoming", "source_language": "de",
                                  "source_text": "Ich habe eine Frage zum Projekt.", "target_language": "ar",
                                  "translated_text": "لدي سؤال عن المشروع.", "latency_ms": 900, "spoken": True})
        assert not session.speaker_mode
        assert session._is_echo("لدي سؤال عن المشروع") and session._is_echo("لدي سؤال عن المشروع.")
        assert not session._is_echo("طيب هبعتلك الملف بكرة")
        assert session.speaker_mode and any(e.get("code") == "echo_detected" for e in events)
        session._headphones.queued_seconds = 2.0
        assert session._echo_guard()  # mic ignored while a translation plays on the speakers
    finally:
        session.stop()


def test_echo_text_is_not_translated_or_spoken():
    mt, got = FakeMT(), []
    vt = VoiceTranslator(SpeechTranslator(FakeSTT("لدي سؤال عن المشروع.", "ar", 0.99), mt, source="ar",
                                          target="de"), VoicedTTS())
    r = vt.process(np.zeros(16_000, np.float32), got.append, reject=lambda t: True)
    assert r.skipped and r.rejected and mt.calls == [] and got == []


# ---- playback -----------------------------------------------------------------------------------
def test_monitor_audio_is_mixed_not_queued_in_front_of_your_translation():
    p = AudioPlayer(AudioDevice(0, "Speakers", "Windows WASAPI", 0, 2, 16_000.0, False, True))
    p.enqueue(AudioChunk(np.full(320, 0.5, np.float32), 16_000))
    p.enqueue(AudioChunk(np.full(320, 0.1, np.float32), 16_000), mix=True)
    out = np.zeros((320, 2), np.float32)
    p._callback(out, 320, None, None)
    assert np.allclose(out[:, 0], 0.6) and np.allclose(out[:, 1], 0.6)
    assert p.queued_seconds == 0  # the monitor never counts as "translation still playing"
