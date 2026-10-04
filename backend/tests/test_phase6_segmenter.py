from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.pipeline.segmenter import (
    PartialAudio,
    SegmenterConfig,
    SpeechStarted,
    UtteranceEnded,
    UtteranceSegmenter,
)
from app.services.audio.format import to_engine_format
from app.services.vad.base import VoiceActivityDetector

W = 512  # samples per window (32 ms)


class LoudnessVAD(VoiceActivityDetector):
    """Fake VAD: 'speech' = window amplitude above 0.1."""
    window_samples = W

    def __call__(self, window):
        return 1.0 if np.abs(window).max() > 0.1 else 0.0

    def reset(self):
        pass


def signal(*parts):
    """parts: ("s"|"q", ms) -> speech (0.5) or quiet (0)."""
    out = [np.full(int(ms * 16), 0.5 if kind == "s" else 0.0, np.float32) for kind, ms in parts]
    return np.concatenate(out)


def run(audio, chunk=320, **cfg):
    seg = UtteranceSegmenter(LoudnessVAD(), SegmenterConfig(partial_interval_s=None, **cfg))
    events = []
    for i in range(0, len(audio), chunk):  # feed in 20 ms frames like the mic does
        events += seg.feed(audio[i:i + chunk])
    return events


def ended(events):
    return [e for e in events if isinstance(e, UtteranceEnded)]


def test_one_utterance_with_pre_roll_and_trimmed_tail():
    ev = run(signal(("q", 1000), ("s", 2000), ("q", 1500)))
    assert [type(e) for e in ev] == [SpeechStarted, UtteranceEnded]
    utt = ev[1].audio
    voiced = np.count_nonzero(utt) / 16_000
    assert voiced == pytest.approx(2.0, abs=0.07)
    assert 0.25 <= np.argmax(utt > 0) / 16_000 <= 0.35  # ~300 ms pre-roll kept
    assert len(utt) / 16_000 < 2.0 + 0.35 + 0.3  # most trailing silence dropped


def test_short_pause_does_not_split_but_long_pause_does():
    one = ended(run(signal(("s", 1000), ("q", 300), ("s", 1000), ("q", 1000))))
    two = ended(run(signal(("s", 1000), ("q", 900), ("s", 1000), ("q", 1000))))
    assert len(one) == 1 and len(two) == 2


def test_latency_mode_changes_end_silence():
    audio = signal(("s", 1000), ("q", 500), ("s", 1000), ("q", 1000))
    assert len(ended(run(audio, end_silence_ms=400))) == 2  # fast mode splits
    assert len(ended(run(audio, end_silence_ms=900))) == 1  # accurate mode waits
    assert SegmenterConfig.for_mode("fast").end_silence_ms == 400


def test_clicks_are_ignored():
    assert run(signal(("q", 500), ("s", 60), ("q", 2000))) == []


def test_forced_cut_on_endless_speech_at_quietest_point():
    # 7 s of speech with a short dip at 4.5 s; max 5 s -> cut near the dip, rest continues
    audio = np.concatenate([signal(("s", 4500)), np.full(3 * W, 0.12, np.float32),
                            signal(("s", 2400), ("q", 1000))])
    ev = ended(run(audio, max_utterance_s=5.0))
    assert len(ev) == 2 and ev[0].forced and not ev[1].forced
    assert 4.4 < len(ev[0].audio) / 16_000 < 5.0


def test_partials_grow_while_speaking():
    seg = UtteranceSegmenter(LoudnessVAD(), SegmenterConfig(partial_interval_s=1.0))
    ev = seg.feed(signal(("s", 3500), ("q", 1000)))
    partials = [e for e in ev if isinstance(e, PartialAudio)]
    assert len(partials) == 3
    assert [round(len(p.audio) / 16_000) for p in partials] == [1, 2, 3]


def test_pause_drops_audio_while_paused():
    seg = UtteranceSegmenter(LoudnessVAD(), SegmenterConfig(partial_interval_s=None))
    seg.paused = True
    assert seg.feed(signal(("s", 2000), ("q", 1000))) == []
    seg.paused = False
    assert len(ended(seg.feed(signal(("s", 1000), ("q", 1000))))) == 1


def test_real_silero_splits_fixture_into_sentences():
    from app.services.vad.silero import SileroVAD

    d, r = sf.read(Path(__file__).parent / "fixtures" / "ar_explain_project.wav", dtype="float32")
    speech = to_engine_format(d, r)
    audio = np.concatenate([np.zeros(16_000, np.float32), speech, np.zeros(16_000, np.float32)])
    seg = UtteranceSegmenter(SileroVAD(), SegmenterConfig(partial_interval_s=None))
    utts = [e for e in seg.feed(audio) if isinstance(e, UtteranceEnded)]
    # the fixture has a ~0.3 s pause between its two sentences: balanced mode keeps them together
    assert len(utts) == 1
    assert abs(len(utts[0].audio) - len(speech)) / 16_000 < 0.6
