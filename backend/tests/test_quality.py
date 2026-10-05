"""Translation, microphone and voice quality: transcript clean-up, recognition hints,
mic conditioning, voice choice and output loudness."""

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import Engine, create_app, preview_sentence
from app.pipeline.factory import Providers
from app.pipeline.speech_translator import SpeechTranslator
from app.pipeline.text_cleanup import clean_transcript, translation_units
from app.pipeline.voice_translator import VoiceTranslator
from app.services.audio.enhance import AutoGain, Conditioner, NoiseProfile, enhance_utterance, normalize_loudness
from app.services.audio.format import SAMPLE_RATE, rms_dbfs
from app.services.stt.base import SpeechToTextProvider, Transcript
from app.services.tts.base import AudioChunk, TextToSpeechProvider, Voice
from tests.test_phase2_pipeline import FakeMT, FakeSTT


# ---- transcript clean-up -----------------------------------------------------------------
@pytest.mark.parametrize("text,lang,expected", [
    ("Uh, I I I think we should, um, go.", "en", "I think we should go."),
    ("Ähm, ich, ich will das, äh, morgen machen.", "de", "Ich will das morgen machen."),
    ("آه، أنا موافق امم بس بكرة.", "ar-EG", "آه، أنا موافق بس بكرة."),  # «آه» = yes: kept
    ("Er hat recht.", "de", "Er hat recht."),  # "er" is a filler only in English
    ("Um.", "en", ""),
    ("This is very very good.", "en", "This is very very good."),  # emphasis, not a stutter
    ("este libro es bueno", "es", "este libro es bueno"),
])
def test_clean_transcript(text, lang, expected):
    assert clean_transcript(text, lang) == expected


def test_short_fragments_are_translated_with_their_neighbour():
    assert translation_units("Ja. Genau. Das machen wir morgen früh im Büro. Und dann gehen wir nach Hause, okay?") == [
        "Ja. Genau. Das machen wir morgen früh im Büro.", "Und dann gehen wir nach Hause, okay?"]


def test_filler_only_utterance_is_not_translated_or_spoken():
    mt = FakeMT()
    r = SpeechTranslator(FakeSTT("Ähm... äh.", "de", 0.99), mt, source="de", target="ar").process(
        np.zeros(16_000, np.float32))
    assert r.skipped and mt.calls == []


# ---- recognition hints -------------------------------------------------------------------
class HintedSTT(SpeechToTextProvider):
    supports_context = True

    def __init__(self, texts):
        self.texts, self.calls = list(texts), []

    def load(self):
        pass

    def transcribe(self, audio, language=None, *, fast=False, context=None, hotwords=None):
        self.calls.append({"context": context, "hotwords": hotwords})
        return Transcript(self.texts.pop(0), language, 0.99, 1.0)


def test_previous_sentence_and_glossary_are_given_to_speech_recognition():
    stt = HintedSTT(["Wir treffen Ahmed bei Siemens.", "Er bringt die Unterlagen mit."])
    st = SpeechTranslator(stt, FakeMT(), source="de", target="ar", glossary="Ahmed, Siemens\nKubernetes")
    st.process(np.zeros(16_000, np.float32))
    st.process(np.zeros(16_000, np.float32))
    assert stt.calls[0] == {"context": None, "hotwords": "Ahmed, Siemens, Kubernetes"}
    assert stt.calls[1]["context"] == "Wir treffen Ahmed bei Siemens."


def test_providers_without_context_support_get_plain_calls():
    st = SpeechTranslator(FakeSTT("Guten Morgen zusammen.", "de", 0.99), FakeMT(), source="de", target="ar",
                          glossary="Ahmed")
    st.process(np.zeros(16_000, np.float32))
    assert st.process(np.zeros(16_000, np.float32)).translated_text  # FakeSTT takes no hints: no TypeError


def test_whisper_echoing_its_prompt_is_dropped():
    from app.providers.stt.faster_whisper_provider import _echoes

    seg = lambda t: SimpleNamespace(text=t)  # noqa: E731
    assert _echoes([seg("Er bringt die Unterlagen mit.")], "Wir treffen Ahmed. Er bringt die Unterlagen mit.")
    assert not _echoes([seg("Ja.")], "Ja, genau.")  # too short to judge
    assert not _echoes([seg("Morgen um neun.")], "Wir treffen Ahmed.")


# ---- microphone ------------------------------------------------------------------------------
def _speechlike(seconds=2.0, amp=0.05):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    gate = (np.sin(2 * np.pi * 3 * t) > 0).astype(np.float32)
    return (amp * np.sin(2 * np.pi * 220 * t) * gate).astype(np.float32), gate.astype(bool)


def test_noise_reduction_improves_snr_and_keeps_length():
    rng = np.random.default_rng(1)
    speech, voiced = _speechlike()
    noisy = speech + (0.01 * rng.standard_normal(speech.size)).astype(np.float32) \
        + (0.05 * np.sin(2 * np.pi * 50 * np.arange(speech.size) / SAMPLE_RATE)).astype(np.float32)  # mains hum
    profile = NoiseProfile()
    profile.update((0.01 * rng.standard_normal(SAMPLE_RATE)).astype(np.float32))
    out = enhance_utterance(noisy, profile, "light")

    def snr(x):
        return 10 * np.log10(np.mean(x[voiced] ** 2) / np.mean(x[~voiced] ** 2))

    assert out.size == noisy.size and snr(out) > snr(noisy) + 5
    assert abs(rms_dbfs(out) - (-20)) < 1 and np.max(np.abs(out)) <= 0.95


def test_noise_reduction_off_still_normalises_but_leaves_clean_speech_alone():
    speech, _ = _speechlike(amp=0.003)  # very quiet mic
    out = enhance_utterance(speech, NoiseProfile(), "off")
    assert rms_dbfs(out) > rms_dbfs(speech) + 15
    assert np.corrcoef(out, speech)[0, 1] > 0.99


def test_auto_gain_raises_a_quiet_mic_slowly_and_never_lowers_below_unity():
    agc = AutoGain()
    quiet = np.full(SAMPLE_RATE, 0.003, np.float32)  # ~-50 dBFS speech
    for _ in range(10):
        agc.learn(agc(quiet))
    assert agc.gain_db == AutoGain.MAX_GAIN_DB
    loud = AutoGain()
    loud.learn(np.full(SAMPLE_RATE, 0.5, np.float32))
    assert loud.gain_db == 0.0


def test_conditioner_learns_noise_only_when_enabled():
    c = Conditioner("off")
    c.silence(np.zeros(16_000, np.float32))
    assert c.noise.frames == 0
    c = Conditioner("light")
    c.silence(np.zeros(16_000, np.float32))
    assert c.noise.ready


def test_quiet_microphone_warning_after_three_utterances():
    from app.pipeline.live import LiveDirection
    from app.pipeline.segmenter import UtteranceSegmenter
    from tests.test_phase6_segmenter import LoudnessVAD

    events = []
    d = LiveDirection("outgoing", None, UtteranceSegmenter(LoudnessVAD()), None, None, on_event=events.append)
    for _ in range(4):
        d._prepare(np.full(16_000, 0.001, np.float32))
    assert [e.get("code") for e in events if e["type"] == "warning"] == ["mic_quiet"]


# ---- voice output ----------------------------------------------------------------------------
class LevelTTS(TextToSpeechProvider):
    def __init__(self, amp):
        self.amp, self.speeds = amp, []

    def load(self):
        pass

    def voices(self, language=None):
        return [Voice("v", "de", "v", None, "medium")]

    def synthesize_stream(self, text, language, *, voice=None, speed=1.0):
        self.speeds.append(speed)
        yield AudioChunk(np.full(2205, self.amp, np.float32) * np.sign(np.sin(np.arange(2205))), 22_050)


def test_every_voice_reaches_the_listener_at_the_same_loudness():
    levels = []
    for amp in (0.02, 0.6):
        got = []
        st = SpeechTranslator(FakeSTT("Das ist ein ganz normaler Satz.", "de", 0.99), FakeMT(), source="de", target="ar")
        VoiceTranslator(st, LevelTTS(amp)).process(np.zeros(16_000, np.float32), got.append)
        levels.append(rms_dbfs(got[0].samples))
    assert abs(levels[0] - levels[1]) < 1.0


def test_catch_up_speed_when_someone_is_waiting():
    tts = LevelTTS(0.1)
    st = SpeechTranslator(FakeSTT("Das ist ein ganz normaler Satz.", "de", 0.99), FakeMT(), source="de", target="ar")
    vt = VoiceTranslator(st, tts, speed=1.0)
    vt.process(np.zeros(16_000, np.float32), lambda c: None, speed_factor=1.15)
    assert tts.speeds == [pytest.approx(1.15)]


def test_normalize_loudness_never_clips():
    out = normalize_loudness(np.array([0.001] * 100 + [0.9], np.float32))
    assert np.max(np.abs(out)) <= 0.95


# ---- voice choice API --------------------------------------------------------------------------
class ChoiceTTS(LevelTTS):
    def __init__(self):
        super().__init__(0.1)
        self.installed = []

    def voice_choices(self, language):
        return [{"id": "de_DE-thorsten-high", "name": "thorsten", "quality": "high", "gender": "male",
                 "installed": True, "size_mb": None},
                {"id": "de_DE-kerstin-low", "name": "kerstin", "quality": "low", "gender": "female",
                 "installed": False, "size_mb": 63}]

    def install_voice(self, voice_id, progress=None):
        self.installed.append(voice_id)
        return voice_id


def test_voice_choice_is_listed_saved_and_previewed(monkeypatch):
    engine = Engine(Settings())
    tts = ChoiceTTS()
    engine.providers = Providers(object(), object(), tts)
    played = []

    class FakePlayer:
        def __init__(self, device=None):
            pass

        def start(self):
            pass

        def enqueue(self, chunk):
            played.append(chunk)

        def wait(self, timeout=None):
            return True

        def close(self):
            pass

    monkeypatch.setattr("app.services.audio.playback.AudioPlayer", FakePlayer)
    with TestClient(create_app(engine, "t")) as c:
        h = {"x-token": "t"}
        r = c.get("/api/voices?language=de", headers=h).json()
        assert r["selected"] == "de_DE-thorsten-high" and len(r["voices"]) == 2
        s = c.put("/api/voices", headers=h, json={"language": "de", "voice": "de_DE-kerstin-low"}).json()
        assert Settings(tts_voices=s["tts_voices"]).preferred_voices() == {
            "ar": "ar_JO-kareem-medium", "de": "de_DE-kerstin-low"}
        assert engine.providers is None  # the new voice is loaded on next start
        engine.providers = Providers(object(), object(), tts)
        assert c.post("/api/voices/preview", headers=h, json={"language": "de", "voice": "de_DE-kerstin-low"}).json()["ok"]
        assert tts.installed == ["de_DE-kerstin-low"] and played
        assert c.get("/api/voices?language=xx", headers=h).status_code == 400


def test_preview_sentence_exists_for_every_language():
    from app.core import languages

    assert all(preview_sentence(code) for code in languages.LANGUAGES)
    assert "Übersetzung" in preview_sentence("de")


def test_high_quality_translation_model_is_a_valid_choice():
    from app.engine import assets

    s = Settings().updated(translation_model="nllb-200-3.3B-ct2-int8")
    mt = next(c for c in assets.plan(s) if c.id == "mt")
    assert mt.size_mb > 3000 and s.translation_model in assets.NLLB_REPOS
    with pytest.raises(ValueError):
        Settings().updated(translation_model="something-else")


def test_no_context_hint_when_the_speaker_language_is_auto_detected():
    stt = HintedSTT(["Wir treffen Ahmed bei Siemens.", "On se voit demain matin."])
    st = SpeechTranslator(stt, FakeMT(), source="auto", target="ar")
    st.process(np.zeros(16_000, np.float32))
    st.process(np.zeros(16_000, np.float32))
    assert stt.calls[1]["context"] is None
