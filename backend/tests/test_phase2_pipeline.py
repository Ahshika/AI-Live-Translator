import numpy as np
import pytest

from app.core import languages
from app.core.config import Settings
from app.pipeline.language import LanguageResolver
from app.pipeline.speech_translator import SpeechTranslator
from app.services.stt.base import SpeechToTextProvider, Transcript
from app.services.translation import registry as mt_registry
from app.services.translation.base import TranslationProvider
from app.utils.text import MAX_WORDS, split_sentences


# ---- text segmentation -------------------------------------------------------
def test_split_latin_and_arabic_punctuation():
    assert split_sentences("Hallo. Wie geht's? Gut!") == ["Hallo.", "Wie geht's?", "Gut!"]
    assert split_sentences("إزيك؟ أنا كويس. وانت") == ["إزيك؟", "أنا كويس.", "وانت"]


def test_split_caps_unpunctuated_run():
    parts = split_sentences(" ".join(["kelma"] * (MAX_WORDS * 2 + 3)))
    assert [len(p.split()) for p in parts] == [MAX_WORDS, MAX_WORDS, 3]


def test_split_keeps_decimals_and_empty():
    assert split_sentences("It costs 3.5 euros") == ["It costs 3.5 euros"]
    assert split_sentences("   ") == []


# ---- languages ---------------------------------------------------------------
def test_language_table():
    assert languages.get("ar-EG").nllb == "arz_Arab"
    assert languages.from_whisper("ar").code == "ar"  # MSA is the default for detected Arabic
    assert languages.from_whisper("xx") is None
    with pytest.raises(languages.UnsupportedLanguageError):
        languages.get("klingon")


# ---- auto-detect fallback ----------------------------------------------------
def test_fixed_language_passes_code_to_stt():
    r = LanguageResolver("ar-EG")
    assert r.stt_language == "ar"
    assert r.resolve("en", 0.99, "hello there").code == "ar-EG"


def test_auto_keeps_last_confident_language_on_short_or_unsure_input():
    r = LanguageResolver("auto", hint="de")
    assert r.stt_language is None
    assert r.resolve("de", 0.97, "Ich habe eine Frage").reason == "confident"
    short = r.resolve("nl", 0.95, "Ja")  # one word: don't trust
    assert (short.code, short.reason) == ("de", "fallback-last")
    unsure = r.resolve("en", 0.40, "Projekt Meeting heute")
    assert (unsure.code, unsure.reason) == ("de", "fallback-last")
    switched = r.resolve("en", 0.95, "Let me switch to English")
    assert (switched.code, switched.reason) == ("en", "confident")


def test_auto_without_history():
    assert LanguageResolver("auto", hint="de").resolve("xx", 0.9, "?? ??").code == "de"
    assert LanguageResolver("auto").resolve("fr", 0.3, "oui").reason == "best-guess"


# ---- pipeline with fake providers -------------------------------------------
class FakeSTT(SpeechToTextProvider):
    def __init__(self, text, lang, prob):
        self.result = (text, lang, prob)
        self.seen_language = "unset"

    def load(self):
        pass

    def transcribe(self, audio, language=None, *, fast=False):
        self.seen_language = language
        text, lang, prob = self.result
        return Transcript(text, language or lang, prob, len(audio) / 16_000)


class FakeMT(TranslationProvider):
    def __init__(self):
        self.calls = []

    def load(self):
        pass

    def translate_batch(self, texts, source, target):
        self.calls.append((texts, source, target))
        return [f"<{target}>{t}" for t in texts]


def test_pipeline_translates_sentence_by_sentence():
    stt, mt = FakeSTT("أنا عايز أشرحلك المشروع. فيه سؤال؟", "ar", 0.99), FakeMT()
    r = SpeechTranslator(stt, mt, source="ar-EG", target="de").process(np.zeros(16_000, np.float32))
    assert stt.seen_language == "ar"
    assert mt.calls == [(["أنا عايز أشرحلك المشروع.", "فيه سؤال؟"], "ar-EG", "de")]
    assert r.translated_text == "<de>أنا عايز أشرحلك المشروع. <de>فيه سؤال؟"
    assert set(r.timings.marks) == {"stt", "mt"} and r.audio_seconds == 1.0


def test_pipeline_skips_mt_on_empty_transcript():
    mt = FakeMT()
    r = SpeechTranslator(FakeSTT("  ", "de", 0.5), mt, source="auto", target="ar").process(
        np.zeros(8000, np.float32))
    assert r.skipped and mt.calls == []


def test_pipeline_rejects_unknown_target():
    with pytest.raises(languages.UnsupportedLanguageError):
        SpeechTranslator(FakeSTT("", "de", 1), FakeMT(), source="ar", target="xx")


def test_same_language_is_passthrough():
    mt = FakeMT()
    assert mt.translate("Hallo", "de", "de").text == "Hallo" and mt.calls == []


def test_translation_registry_rejects_unknown():
    with pytest.raises(ValueError, match="nllb"):
        mt_registry.create(Settings(translation_provider="nope"))


def test_every_language_code_exists_in_the_models():
    from pathlib import Path

    from faster_whisper.tokenizer import _LANGUAGE_CODES
    from tokenizers import Tokenizer

    tok = Path(__file__).parents[1] / "models" / "nllb-200-distilled-1.3B-ct2-int8" / "tokenizer.json"
    if not tok.exists():
        pytest.skip("NLLB model not downloaded")
    vocab = Tokenizer.from_file(str(tok)).get_vocab()
    for lang in languages.LANGUAGES.values():
        assert lang.nllb in vocab, lang
        assert not lang.whisper or lang.whisper in _LANGUAGE_CODES, lang
    assert len(languages.LANGUAGES) >= 100
