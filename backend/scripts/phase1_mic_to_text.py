"""Phase 1 — Microphone -> Speech-to-Text (push-to-talk CLI).

Usage (from backend/):
    .venv/Scripts/python scripts/phase1_mic_to_text.py --list-devices
    .venv/Scripts/python scripts/phase1_mic_to_text.py --language ar
    .venv/Scripts/python scripts/phase1_mic_to_text.py --file sample.wav
    .venv/Scripts/python scripts/phase1_mic_to_text.py --model small --device cpu

Press Enter to start talking, Enter again to stop. The transcript, detected language
and timings are printed. Ctrl+C to quit. Nothing is saved to disk unless --save is given.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import _cli  # noqa: F401  (sets sys.path + utf-8 stdout)
import soundfile as sf

from app.core.config import Settings
from app.services.audio.capture import MicrophoneUnavailableError
from app.services.audio.format import SAMPLE_RATE, duration_seconds
from app.services.stt import registry
from app.services.stt.base import STTUnavailableError, Transcript


def report(t: Transcript, audio_seconds: float, stt_seconds: float) -> None:
    lang = f"{t.language} ({t.language_probability:.0%})" if t.language else "?"
    print(f"\n  📝 {t.text or '(no speech recognised)'}")
    print(f"     language={lang}  audio={audio_seconds:.1f}s  stt={stt_seconds*1000:.0f}ms  "
          f"RTF={stt_seconds / max(audio_seconds, 1e-6):.2f}\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--input", help="input device name substring or index (default: system mic)")
    p.add_argument("--language", default=None, help="Whisper code (ar, de, en...). Omit for auto-detect")
    p.add_argument("--model", default=None, help="whisper model (large-v3-turbo, medium, small...)")
    p.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    p.add_argument("--file", type=Path, help="transcribe an audio file instead of the mic")
    p.add_argument("--save", type=Path, help="save each mic recording as WAV into this folder")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.list_devices:
        _cli.print_devices()
        return 0

    settings = Settings.from_env(stt_model=args.model, stt_device=args.device, input_device=args.input)
    stt = registry.create(settings)
    print(f"Loading STT '{settings.stt_provider}' model '{settings.stt_model}' (first run downloads it)...")
    try:
        stt.load()
    except STTUnavailableError as exc:
        print(f"❌ {exc}")
        return 2
    print(f"✅ STT ready on {getattr(stt, 'device', '?')}/{getattr(stt, 'compute_type', '?')}")

    if args.file:
        audio = _cli.load_audio_file(args.file)
        t0 = time.perf_counter()
        report(stt.transcribe(audio, args.language), duration_seconds(audio), time.perf_counter() - t0)
        return 0

    try:
        mic = _cli.open_mic(settings.input_device)
        with mic:
            n = 0
            while True:
                audio = _cli.record_push_to_talk(mic)
                secs = duration_seconds(audio)
                if secs < 0.3:
                    print("   too short, try again")
                    continue
                if args.save:
                    args.save.mkdir(parents=True, exist_ok=True)
                    n += 1
                    sf.write(args.save / f"utt_{n:03d}.wav", audio, SAMPLE_RATE)
                t0 = time.perf_counter()
                report(stt.transcribe(audio, args.language), secs, time.perf_counter() - t0)
    except (LookupError, MicrophoneUnavailableError) as exc:
        print(f"❌ Microphone: {exc}")
        return 3
    except (KeyboardInterrupt, EOFError):
        print(f"\nbye. capture stats: {mic.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
