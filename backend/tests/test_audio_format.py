import numpy as np
import pytest

from app.services.audio.format import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    rms_dbfs,
    to_engine_format,
    to_mono_float32,
)


def test_frame_is_20ms_at_16k():
    assert FRAME_SAMPLES == 320


def test_int16_scaled_to_unit_range():
    out = to_mono_float32(np.array([0, 16384, -32768], dtype=np.int16))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.0, 0.5, -1.0])


def test_stereo_downmixed():
    stereo = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(to_mono_float32(stereo), [0.5, 0.5])


def test_48k_to_16k_keeps_duration_and_tone():
    t = np.arange(48_000) / 48_000
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    out = to_engine_format(np.stack([tone, tone], axis=1), 48_000)
    assert out.dtype == np.float32
    assert abs(len(out) - SAMPLE_RATE) <= 1
    spectrum = np.abs(np.fft.rfft(out))
    assert abs(np.argmax(spectrum) * SAMPLE_RATE / len(out) - 440) < 2


def test_resampling_removes_content_above_new_nyquist():
    t = np.arange(48_000) / 48_000
    hiss = np.sin(2 * np.pi * 12_000 * t).astype(np.float32)  # above 8 kHz
    assert rms_dbfs(to_engine_format(hiss, 48_000)) < -40  # filtered, not aliased


def test_rms_dbfs():
    assert rms_dbfs(np.zeros(10, np.float32)) == float("-inf")
    assert rms_dbfs(np.ones(10, np.float32)) == pytest.approx(0.0)


def _distortion_db(signal: np.ndarray, rate: int, freq: float) -> float:
    """Energy outside +-20 Hz of the tone, relative to total (dB). Clicks raise it."""
    spec = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
    freqs = np.fft.rfftfreq(len(signal), 1 / rate)
    off = spec[np.abs(freqs - freq) > 20].sum()
    return 10 * np.log10(off / spec.sum())


def test_stream_resampler_has_no_block_edge_artifacts():
    from app.services.audio.format import StreamResampler

    rate = 48_000
    t = np.arange(rate * 2) / rate
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    blocks = [tone[i:i + 960] for i in range(0, len(tone), 960)]  # 20 ms, like the mic
    rs = StreamResampler(rate)
    stream = np.concatenate([rs(b) for b in blocks])[4000:-4000]
    naive = np.concatenate([to_engine_format(b, rate) for b in blocks])[4000:-4000]
    assert _distortion_db(stream, 16_000, 440) < -60
    assert _distortion_db(naive, 16_000, 440) > _distortion_db(stream, 16_000, 440) + 10  # the old bug
