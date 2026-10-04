"""Helpers shared by the phase-by-phase CLI scripts."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from app.services.audio import devices  # noqa: E402
from app.services.audio.capture import MicrophoneCapture  # noqa: E402
from app.services.audio.format import FRAME_SAMPLES, SAMPLE_RATE, rms_dbfs, to_engine_format  # noqa: E402

MAX_RECORD_SECONDS = 30


def print_devices() -> None:
    for d in devices.list_devices():
        kind = "/".join(k for k, n in (("in", d.max_input_channels), ("out", d.max_output_channels)) if n)
        print(f"[{d.index:>3}] {kind:<6} {int(d.default_sample_rate):>6} Hz  {d.name}")


def open_mic(query: str | None) -> MicrophoneCapture:
    """Raises LookupError / MicrophoneUnavailableError."""
    mic = MicrophoneCapture(device=devices.find_device(query) if query else None)
    print(f"🎙  Mic: {mic.device.name}  ({int(mic.device.default_sample_rate)} Hz → {SAMPLE_RATE} Hz)")
    return mic


def load_audio_file(path: Path) -> np.ndarray:
    data, rate = sf.read(path, dtype="float32")
    return to_engine_format(data, rate)


def record_push_to_talk(mic: MicrophoneCapture) -> np.ndarray:
    input("▶  Press Enter and speak...")
    mic.clear()
    stop = threading.Event()
    threading.Thread(target=lambda: (input("●  Recording — press Enter to stop"), stop.set()), daemon=True).start()
    frames: list[np.ndarray] = []
    max_frames = MAX_RECORD_SECONDS * SAMPLE_RATE // FRAME_SAMPLES
    while not stop.is_set() and len(frames) < max_frames:
        frame = mic.read_frame(timeout=0.5)
        if frame is not None:
            frames.append(frame)
    if len(frames) >= max_frames:
        print(f"   (stopped at the {MAX_RECORD_SECONDS}s limit — press Enter)")
        stop.wait()
    audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
    level = rms_dbfs(audio)
    if audio.size and level < -60:
        print(f"   ⚠ very quiet input ({level:.0f} dBFS) — is the mic muted / the right one selected?")
    return audio
