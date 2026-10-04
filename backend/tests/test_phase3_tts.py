import os

import numpy as np
import pytest

from app.core.config import Settings
from app.services.audio import playback
from app.services.audio.playback import AudioPlayer
from app.services.tts import registry
from app.services.tts.base import AudioChunk, NoVoiceError


# ---- player (no real device: we drive the callback directly) ----------------
@pytest.fixture
def player(monkeypatch):
    monkeypatch.setattr(playback.sd, "query_devices",
                        lambda kind=None: {"name": "Fake", "default_samplerate": 16_000, "max_output_channels": 2})
    return AudioPlayer()


def pull(p: AudioPlayer, frames: int) -> np.ndarray:
    out = np.full((frames, p.channels), 9.0, dtype=np.float32)
    p._callback(out, frames, None, None)
    return out[:, 0].copy()


def test_player_plays_queue_in_order_across_chunks_then_silence(player):
    player.enqueue(AudioChunk(np.full(3, 0.1, np.float32), 16_000))
    player.enqueue(AudioChunk(np.full(4, 0.2, np.float32), 16_000))
    assert not player.wait(timeout=0)
    np.testing.assert_allclose(pull(player, 5), [0.1, 0.1, 0.1, 0.2, 0.2])
    np.testing.assert_allclose(pull(player, 4), [0.2, 0.2, 0.0, 0.0])
    assert player.wait(timeout=0)


def test_player_resamples_to_device_rate(player):
    player.enqueue(AudioChunk(np.zeros(22_050, np.float32), 22_050))
    assert player.queued_seconds == pytest.approx(1.0, abs=0.01)


def test_stop_fades_out_and_drops_queue(player):
    player.enqueue(AudioChunk(np.ones(16_000, np.float32), 16_000))
    player.enqueue(AudioChunk(np.ones(16_000, np.float32), 16_000))
    pull(player, 100)
    player.stop()
    fade = player.rate * AudioPlayer.FADE_MS // 1000
    assert player.queued_seconds == pytest.approx(fade / player.rate)
    tail = pull(player, fade + 10)
    assert tail[0] == pytest.approx(1.0) and np.all(np.diff(tail[:fade]) <= 0) and tail[-1] == 0.0


# ---- config / registry -------------------------------------------------------
def test_preferred_voices_parsing():
    s = Settings(tts_voices=" ar = ar_JO-kareem-medium , de=de_DE-kerstin-low,bad")
    assert s.preferred_voices() == {"ar": "ar_JO-kareem-medium", "de": "de_DE-kerstin-low"}


def test_unknown_tts_provider():
    with pytest.raises(ValueError, match="piper"):
        registry.create(Settings(tts_provider="nope"))


# ---- real Piper voices -------------------------------------------------------
needs_models = pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1")


@pytest.fixture(scope="module")
def tts():
    t = registry.create(Settings())
    t.load()
    return t


@needs_models
def test_voice_catalog_and_preference(tts):
    ids = [v.id for v in tts.voices("de")]
    assert ids[0] == "de_DE-thorsten-high"  # best quality first
    assert all(v.language == "ar" for v in tts.voices("ar-EG"))
    from app.providers.tts.piper_provider import PiperProvider

    offline = PiperProvider(auto_download=False)
    with pytest.raises(NoVoiceError):
        offline._pick("ja", None)


@needs_models
@pytest.mark.parametrize("text,lang", [("Ich möchte dir das Projekt erklären.", "de"),
                                        ("لدي سؤال عن المشروع.", "ar")])
def test_synthesizes_audible_speech(tts, text, lang):
    chunk = tts.synthesize(text, lang)
    assert 1.0 < chunk.seconds < 8.0
    assert np.sqrt(np.mean(chunk.samples ** 2)) > 0.02  # not silence


@needs_models
def test_speed_changes_duration(tts):
    # Piper's length_scale shortens phonemes but not pauses, so the effect is monotonic
    # but sub-linear (measured: speed 2.0 -> ~0.68x duration, not 0.5x).
    text = "Guten Morgen, wie geht es dir heute? Ich möchte dir das Projekt erklären."
    # Piper durations are slightly random (noise_w), so compare well-separated speeds.
    slow, normal, fast = (tts.synthesize(text, "de", speed=sp).seconds for sp in (0.5, 1.0, 2.0))
    assert slow > normal * 1.3 and fast < normal * 0.85


def test_player_fills_every_channel_of_a_stereo_device(player):
    """Regression: WASAPI stereo endpoints (e.g. VB-CABLE) refuse mono streams."""
    assert player.channels == 2
    player.enqueue(AudioChunk(np.full(4, 0.25, np.float32), 16_000))
    out = np.zeros((4, 2), np.float32)
    player._callback(out, 4, None, None)
    np.testing.assert_allclose(out, 0.25)
