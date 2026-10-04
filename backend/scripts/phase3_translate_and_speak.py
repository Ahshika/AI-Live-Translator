"""Phase 3 — Translation -> Text-to-Speech.

Type a sentence in your language; it's translated and spoken in the other language.

Usage (from backend/):
    .venv/Scripts/python scripts/phase3_translate_and_speak.py                    # interactive, ar-EG -> de
    .venv/Scripts/python scripts/phase3_translate_and_speak.py --text "أنا عايز أشرحلك المشروع"
    .venv/Scripts/python scripts/phase3_translate_and_speak.py --from de --to ar --text "Ich habe eine Frage."
    .venv/Scripts/python scripts/phase3_translate_and_speak.py --say "Hallo zusammen" --lang de   # TTS only
    .venv/Scripts/python scripts/phase3_translate_and_speak.py --voice de_DE-kerstin-low --speed 1.1
    .venv/Scripts/python scripts/phase3_translate_and_speak.py --save out.wav --text "..."
    .venv/Scripts/python scripts/phase3_translate_and_speak.py --list-voices
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import _cli  # noqa: F401  (sets sys.path + utf-8 stdout)
import numpy as np
import soundfile as sf

from app.core import languages
from app.core.config import Settings
from app.services.audio.playback import AudioPlayer, SpeakerUnavailableError
from app.services.translation import registry as mt_registry
from app.services.translation.base import TranslationUnavailableError
from app.services.tts import registry as tts_registry
from app.services.tts.base import TTSUnavailableError


def speak(tts, player: AudioPlayer | None, text: str, lang: str, voice: str | None, speed: float,
          save: Path | None) -> dict[str, float]:
    t0 = time.perf_counter()
    first = None
    parts = []
    for chunk in tts.synthesize_stream(text, lang, voice=voice, speed=speed):
        if first is None:
            first = time.perf_counter() - t0
        parts.append(chunk)
        if player:
            player.enqueue(chunk)  # playback starts while later sentences are still synthesising
    synth = time.perf_counter() - t0
    audio_s = sum(c.seconds for c in parts)
    if save and parts:
        sf.write(save, np.concatenate([c.samples for c in parts]), parts[0].sample_rate)
        print(f"     saved → {save}")
    return {"ttfa": (first or 0) * 1000, "synth": synth * 1000, "audio": audio_s}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="source")
    p.add_argument("--to", dest="target")
    p.add_argument("--text", help="translate + speak this text, then exit")
    p.add_argument("--say", help="speak this text as-is (no translation)")
    p.add_argument("--lang", help="language of --say text")
    p.add_argument("--voice", help="voice id (see --list-voices)")
    p.add_argument("--speed", type=float, default=None, help="0.5–2.0 (default 1.0)")
    p.add_argument("--output", help="output device name substring or index (default: system speakers)")
    p.add_argument("--save", type=Path, help="also write the speech to this WAV file")
    p.add_argument("--no-play", action="store_true", help="don't play (useful with --save)")
    p.add_argument("--list-voices", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    s = Settings.from_env(output_device=args.output)
    speed = args.speed if args.speed is not None else s.speech_speed
    tts = tts_registry.create(s)
    try:
        tts.load()
    except TTSUnavailableError as exc:
        print(f"❌ {exc}")
        return 2

    if args.list_voices:
        for v in tts.voices():
            print(f"  {v.language}  {v.id:<34} {v.gender or '?':<7} {v.quality}")
        return 0

    player = None
    if not args.no_play:
        try:
            player = AudioPlayer(s.output_device)
            player.start()
            print(f"🔊 Output: {player.device_name} ({player.rate} Hz)")
        except (LookupError, SpeakerUnavailableError) as exc:
            print(f"❌ Speaker: {exc}")
            return 3

    try:
        if args.say:
            lang = args.lang or s.other_language
            m = speak(tts, player, args.say, lang, args.voice, speed, args.save)
            print(f"  🔈 [{lang}] {args.say}\n     tts first audio={m['ttfa']:.0f}ms  "
                  f"synth={m['synth']:.0f}ms  audio={m['audio']:.1f}s")
            return 0

        source, target = args.source or s.my_language, args.target or s.other_language
        try:
            languages.get(source), languages.get(target)
        except languages.UnsupportedLanguageError as exc:
            print(f"❌ {exc}")
            return 1
        mt = mt_registry.create(s)
        try:
            mt.load()
        except TranslationUnavailableError as exc:
            print(f"❌ {exc}")
            return 2
        tts.warm_up(target, args.voice)
        src_l, tgt_l = languages.get(source), languages.get(target)
        print(f"{src_l.name} {src_l.flag}  →  {tgt_l.name} {tgt_l.flag}   (voice: "
              f"{args.voice or s.preferred_voices().get(target.split('-')[0], 'auto')})")

        def run(text: str) -> None:
            t0 = time.perf_counter()
            translated = mt.translate(text, source, target).text
            mt_ms = (time.perf_counter() - t0) * 1000
            print(f"  🌍 {translated}")
            m = speak(tts, player, translated, target, args.voice, speed, args.save)
            print(f"     mt={mt_ms:.0f}ms  tts first audio={m['ttfa']:.0f}ms  "
                  f"→ speech starts after ≈{mt_ms + m['ttfa']:.0f}ms  (audio {m['audio']:.1f}s)\n")

        if args.text:
            run(args.text)
            return 0
        print("Type a sentence and press Enter (Ctrl+C to quit).")
        while True:
            text = input("✏  ").strip()
            if text:
                run(text)
    except (KeyboardInterrupt, EOFError):
        print("\nbye.")
    finally:
        if player:
            player.wait(timeout=30)
            player.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
