"""Real-model test. Skipped unless RUN_MODEL_TESTS=1."""

import os

import pytest

from app.core.config import Settings
from app.services.translation import registry

pytestmark = pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1")


@pytest.fixture(scope="module")
def mt():
    provider = registry.create(Settings.from_env())
    provider.load()
    return provider


def test_english_to_german(mt):
    out = mt.translate("I want to explain the project to you.", "en", "de").text
    assert "Projekt" in out and "erklären" in out


def test_german_to_arabic(mt):
    out = mt.translate("Ich habe eine Frage zum Projekt.", "de", "ar").text
    assert "سؤال" in out and "مشروع" in out


def test_arabic_to_german(mt):
    out = mt.translate("أنا أريد أن أشرح لك المشروع.", "ar", "de").text
    assert "Projekt" in out
