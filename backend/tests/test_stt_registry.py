import numpy as np
import pytest

from app.core.config import Settings
from app.services.stt import registry
from app.services.stt.base import SpeechToTextProvider, Transcript


class FakeSTT(SpeechToTextProvider):
    name = "fake"

    def __init__(self):
        self.loaded = False

    def load(self):
        self.loaded = True

    def transcribe(self, audio, language=None, *, fast=False):
        return Transcript(text="hello", language=language or "en", language_probability=1.0,
                          duration=len(audio) / 16_000)


def test_custom_provider_is_selected_by_config():
    registry.register("fake", lambda s: FakeSTT())
    stt = registry.create(Settings(stt_provider="fake"))
    stt.load()
    t = stt.transcribe(np.zeros(16_000, np.float32), "de")
    assert (t.text, t.language, t.duration) == ("hello", "de", 1.0)


def test_unknown_provider_lists_choices():
    with pytest.raises(ValueError, match="faster-whisper"):
        registry.create(Settings(stt_provider="nope"))


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("TRANSLATOR_STT_MODEL", "small")
    s = Settings.from_env(stt_device="cpu")
    assert (s.stt_model, s.stt_device, s.my_language) == ("small", "cpu", "ar")


def test_settings_save_and_reload_with_types(monkeypatch):
    s = Settings(my_language="ar-EG", headphones=False, speech_speed=1.2)
    s.save()
    again = Settings.from_env()
    assert (again.my_language, again.headphones, again.speech_speed) == ("ar-EG", False, 1.2)
    monkeypatch.setenv("TRANSLATOR_HEADPHONES", "false")
    assert Settings.from_env(headphones=None).headphones is False
    assert s.updated(headphones="true", speech_speed="0.9", bogus=1).speech_speed == 0.9
