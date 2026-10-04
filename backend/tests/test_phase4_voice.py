import os
from pathlib import Path

import numpy as np
import pytest

from app.core.config import Settings
from app.pipeline.factory import build_voice_translator, direction_languages
from app.pipeline.speech_translator import SpeechTranslator
from app.pipeline.voice_translator import VoiceTranslator
from app.services.audio.format import to_engine_format, trim_silence
from app.services.tts.base import AudioChunk, TextToSpeechProvider
from app.core import languages
from tests.test_phase2_pipeline import FakeMT, FakeSTT

FIXTURES = Path(__file__).parent / "fixtures"


def test_trim_silence_keeps_short_margins():
    rate = 1000
    audio = np.concatenate([np.zeros(300), np.full(100, 0.5), np.zeros(500)]).astype(np.float32)
    out = trim_silence(audio, rate, keep_lead_ms=30, keep_trail_ms=150)
    assert len(out) == 30 + 100 + 150
    assert trim_silence(np.zeros(50, np.float32), rate).size == 0


class FakeTTS(TextToSpeechProvider):
    def __init__(self):
        self.spoken = []

    def load(self):
        pass

    def voices(self, language=None):
        return []

    def synthesize_stream(self, text, language, *, voice=None, speed=1.0):
        self.spoken.append((text, language))
        yield AudioChunk(np.full(100, 0.5, np.float32), 1000)


def make(text, lang="ar", source="ar", target="de"):
    st = SpeechTranslator(FakeSTT(text, lang, 0.99), FakeMT(), source=source, target=target)
    tts = FakeTTS()
    return VoiceTranslator(st, tts), tts


def test_each_sentence_is_spoken_in_order_as_soon_as_translated():
    vt, tts = make("جملة أولى. جملة ثانية؟")
    received = []
    r = vt.process(np.zeros(16_000, np.float32), received.append)
    assert tts.spoken == [("<de>جملة أولى.", "de"), ("<de>جملة ثانية؟", "de")]
    assert len(received) == 2 and r.speech_seconds == pytest.approx(0.2)
    assert r.translated_text == "<de>جملة أولى. <de>جملة ثانية؟"
    assert 0 <= r.stt_ms <= r.first_audio_ms <= r.total_ms


def test_silence_produces_no_audio():
    vt, tts = make("   ")
    received = []
    r = vt.process(np.zeros(8000, np.float32), received.append)
    assert r.skipped and received == [] and tts.spoken == [] and r.first_audio_ms is None


def test_direction_languages():
    s = Settings(my_language="ar-EG", other_language="de")
    out, inc = direction_languages(s, "outgoing"), direction_languages(s, "incoming")
    assert (out.source, out.target, inc.source, inc.target) == ("ar-EG", "de", "de", "ar-EG")
    with pytest.raises(languages.UnsupportedLanguageError):
        direction_languages(Settings(my_language="auto"), "incoming")


# ---- real models: full voice -> voice, verified by listening back with Whisper ----
needs_models = pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1")


@needs_models
def test_arabic_voice_becomes_understandable_german_voice():
    import soundfile as sf

    vt = build_voice_translator(Settings(my_language="ar", other_language="de"), "outgoing")
    data, rate = sf.read(FIXTURES / "ar_explain_project.wav", dtype="float32")
    chunks = []
    r = vt.process(to_engine_format(data, rate), chunks.append)
    assert "Projekt" in r.translated_text
    speech = to_engine_format(np.concatenate([c.samples for c in chunks]), chunks[0].sample_rate)
    heard = vt.translator.stt.transcribe(speech)  # what the German listener would hear
    assert heard.language == "de" and "Projekt" in heard.text


class NoVoiceTTS(FakeTTS):
    def synthesize_stream(self, text, language, *, voice=None, speed=1.0):
        from app.services.tts.base import NoVoiceError
        raise NoVoiceError("none")
        yield  # pragma: no cover


def test_missing_voice_falls_back_to_subtitles():
    st = SpeechTranslator(FakeSTT("جملة. وجملة.", "ar", 0.99), FakeMT(), source="ar", target="de")
    vt = VoiceTranslator(st, NoVoiceTTS())
    got = []
    r = vt.process(np.zeros(16_000, np.float32), got.append)
    assert r.text_only and got == [] and len(r.pairs) == 2 and vt.speak is False
