"""Phases 5+6 — Live, hands-free translation (one direction).

Just talk. Voice activity detection decides when you start and stop; live subtitles appear
while you speak; each finished sentence is translated and spoken.

Usage (from backend/):
    .venv/Scripts/python scripts/phase6_live.py                          # mic, ar-EG -> de
    .venv/Scripts/python scripts/phase6_live.py --direction incoming     # de -> ar
    .venv/Scripts/python scripts/phase6_live.py --latency fast           # fast | balanced | accurate
    .venv/Scripts/python scripts/phase6_live.py --headphones             # disable the echo guard
    .venv/Scripts/python scripts/phase6_live.py --file clip.wav          # simulate a speaker from a file
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import _cli  # noqa: F401  (sets sys.path + utf-8 stdout)

from app.core import languages
from app.core.config import Settings
from app.pipeline.factory import build_voice_translator
from app.pipeline.live import LiveDirection
from app.pipeline.segmenter import LATENCY_MODES, SegmenterConfig, UtteranceSegmenter
from app.services.audio.capture import MicrophoneUnavailableError
from app.services.audio.file_source import FileAudioSource
from app.services.audio.playback import AudioPlayer, SpeakerUnavailableError
from app.services.stt.base import STTUnavailableError
from app.services.translation.base import TranslationUnavailableError
from app.services.tts.base import TTSUnavailableError
from app.services.vad.silero import SileroVAD


def printer(event: dict) -> None:
    kind = event["type"]
    if kind == "partial":
        sys.stdout.write(f"\r  … {event['text'][-90:]:<90}")
        sys.stdout.flush()
    elif kind == "final":
        src = languages.get(event["source_language"])
        tgt = languages.get(event["target_language"])
        lat = f"{event['latency_ms']:.0f}ms" if event["latency_ms"] is not None else "text only"
        sys.stdout.write("\r" + " " * 96 + "\r")
        print(f"  {src.flag} {event['source_text']}")
        print(f"  {tgt.flag} {event['translated_text']}")
        print(f"     ⏱ you stopped → translation heard after {lat}"
              + ("  (subtitles only)" if event["text_only"] else "") + "\n")
    elif kind == "error":
        print(f"\n  ⚠ {event['code']}: {event['message']}")
    elif kind == "status":
        print(f"  ● {event['state']}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--direction", choices=["outgoing", "incoming"], default="outgoing")
    p.add_argument("--from", dest="source")
    p.add_argument("--to", dest="target")
    p.add_argument("--voice")
    p.add_argument("--speed", type=float)
    p.add_argument("--latency", choices=list(LATENCY_MODES), default="balanced")
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--headphones", action="store_true", help="you use headphones: don't pause while speaking")
    p.add_argument("--no-partials", action="store_true")
    p.add_argument("--file", type=Path, help="simulate a live speaker with an audio file")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    s = Settings.from_env(input_device=args.input, output_device=args.output)
    print("Loading models...")
    try:
        vt = build_voice_translator(s, args.direction, source=args.source, target=args.target,
                                    voice=args.voice, speed=args.speed)
    except languages.UnsupportedLanguageError as exc:
        print(f"❌ {exc}")
        return 1
    except (STTUnavailableError, TranslationUnavailableError, TTSUnavailableError) as exc:
        print(f"❌ {exc}")
        return 2

    try:
        player = AudioPlayer(s.output_device)
        player.start()
        source = FileAudioSource(args.file, realtime=True) if args.file else _cli.open_mic(s.input_device)
    except (LookupError, SpeakerUnavailableError, MicrophoneUnavailableError) as exc:
        print(f"❌ {exc}")
        return 3

    # Echo guard: on speakers the mic hears our translation; stop listening while it plays.
    guard = None if (args.headphones or args.file) else (lambda: player.queued_seconds > 0)
    seg = UtteranceSegmenter(SileroVAD(), SegmenterConfig.for_mode(
        args.latency, partial_interval_s=None if args.no_partials else 1.0))
    live = LiveDirection(args.direction, source, seg, vt, player.enqueue, on_event=printer,
                         pause_when=guard, partials=not args.no_partials)
    src = vt.translator.resolver.configured
    print(f"✅ {'Auto' if src == 'auto' else languages.get(src).name} → {languages.get(vt.target).name}"
          f"   · output: {player.device_name} · latency mode: {args.latency}")
    if not vt.speak:
        print("   (no voice for this language — translations will be shown as text)")
    print("   Just talk. Ctrl+C to stop.\n")
    live.start()
    try:
        if args.file:
            source.finished.wait()
            while live._queue.qsize() or live._busy.is_set():
                time.sleep(0.1)
            player.wait(timeout=30)
        else:
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        live.stop()
        player.close()
        print(f"stats: {live.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
