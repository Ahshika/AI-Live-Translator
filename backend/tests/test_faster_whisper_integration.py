"""Real-model test. Skipped unless RUN_MODEL_TESTS=1 (it loads Whisper on the GPU/CPU)."""

import os
from pathlib import Path

import pytest
import soundfile as sf

from app.core.config import Settings
from app.services.audio.format import to_engine_format
from app.services.stt import registry

pytestmark = pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1")

FIXTURE = Path(__file__).parent / "fixtures" / "en_explain_project.wav"


@pytest.fixture(scope="module")
def stt():
    provider = registry.create(Settings.from_env())
    provider.load()
    return provider


def test_transcribes_english_with_auto_detect(stt):
    data, rate = sf.read(FIXTURE, dtype="float32")
    t = stt.transcribe(to_engine_format(data, rate))
    assert t.language == "en"
    words = t.text.lower()
    assert "explain" in words and "project" in words
