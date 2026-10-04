"""Phase 2 — Speech -> Text -> Translation.

Usage (from backend/):
    .venv/Scripts/python scripts/phase2_speech_to_translation.py                  # mic, ar -> de
    .venv/Scripts/python scripts/phase2_speech_to_translation.py --from ar-EG     # Egyptian Arabic
    .venv/Scripts/python scripts/phase2_speech_to_translation.py --direction incoming   # de -> ar
    .venv/Scripts/python scripts/phase2_speech_to_translation.py --from auto --to ar
    .venv/Scripts/python scripts/phase2_speech_to_translation.py --file clip.wav --from auto --to ar
    .venv/Scripts/python scripts/phase2_speech_to_translation.py --text "أنا عايز أشرحلك المشروع" --from ar-EG
    .venv/Scripts/python scripts/phase2_speech_to_translation.py --list-languages

"outgoing" = I speak my language -> other person's language (default)
"incoming" = the other person speaks -> my language
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import _cli  # noqa: F401  (sets sys.path + utf-8 stdout)

from app.core import languages
from app.core.config import Settings
from app.pipeline.speech_translator import SpeechTranslator, TranslatedUtterance
from app.services.audio.capture import MicrophoneUnavailableError
from app.services.stt import registry as stt_registry
from app.services.stt.base import STTUnavailableError
from app.services.translation import registry as mt_registry
from app.services.translation.base import TranslationUnavailableError


def label(code: str) -> str:
    if code == languages.AUTO:
        return "Auto-detect 🌐"
    lang = languages.get(code)
    return f"{lang.name} {lang.flag}"


def report(r: TranslatedUtterance) -> None:
    if r.skipped:
        print("   (no speech recognised)\n")
        return
    d = r.language
    conf = f"{d.confidence:.0%}" if d.confidence is not None else "?"
    print(f"\n  🗣  [{d.code}] {r.transcript.text}")
    print(f"  🌍 [{r.target_language}] {r.translated_text}")
    print(f"     detected={d.detected} ({conf}, {d.reason})  audio={r.audio_seconds:.1f}s  "
          f"stt={r.timings.marks['stt']:.0f}ms  mt={r.timings.marks['mt']:.0f}ms  "
          f"total={r.timings.total_ms:.0f}ms\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--direction", choices=["outgoing", "incoming"], default="outgoing")
    p.add_argument("--from", dest="source", help="source language code or 'auto'")
    p.add_argument("--to", dest="target", help="target language code")
    p.add_argument("--text", help="translate this text directly (skips mic/STT)")
    p.add_argument("--file", type=Path, help="translate speech from an audio file")
    p.add_argument("--input", help="input device name substring or index")
    p.add_argument("--list-languages", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.list_languages:
        for lang in languages.LANGUAGES.values():
            print(f"  {lang.code:<6} {lang.flag}  {lang.name:<22} {lang.native}")
        return 0

    s = Settings.from_env(input_device=args.input)
    default_src, default_tgt = (s.my_language, s.other_language) if args.direction == "outgoing" \
        else (s.other_language, s.my_language)
    source, target = args.source or default_src, args.target or default_tgt
    try:
        if source != languages.AUTO:
            languages.get(source)
        languages.get(target)
    except languages.UnsupportedLanguageError as exc:
        print(f"❌ {exc}. See --list-languages")
        return 1
    print(f"{label(source)}  →  {label(target)}")

    mt = mt_registry.create(s)
    print(f"Loading translation '{s.translation_provider}'...")
    try:
        mt.load()
    except TranslationUnavailableError as exc:
        print(f"❌ {exc}")
        return 2
    print(f"✅ MT ready on {getattr(mt, 'device', '?')}/{getattr(mt, 'compute_type', '?')}")

    if args.text:
        if source == languages.AUTO:
            print("❌ --text needs an explicit --from language")
            return 1
        t0 = time.perf_counter()
        out = mt.translate(args.text, source, target)
        print(f"\n  🌍 {out.text}\n     mt={(time.perf_counter() - t0) * 1000:.0f}ms")
        return 0

    stt = stt_registry.create(s)
    print(f"Loading STT '{s.stt_provider}' ({s.stt_model})...")
    try:
        stt.load()
    except STTUnavailableError as exc:
        print(f"❌ {exc}")
        return 2
    print(f"✅ STT ready on {getattr(stt, 'device', '?')}/{getattr(stt, 'compute_type', '?')}")

    hint = s.other_language if args.direction == "incoming" and s.other_language != languages.AUTO else None
    translator = SpeechTranslator(stt, mt, source=source, target=target,
                                  min_confidence=s.language_min_confidence, source_hint=hint)

    if args.file:
        report(translator.process(_cli.load_audio_file(args.file)))
        return 0

    try:
        mic = _cli.open_mic(s.input_device)
        with mic:
            while True:
                audio = _cli.record_push_to_talk(mic)
                if len(audio) < 0.3 * 16_000:
                    print("   too short, try again")
                    continue
                report(translator.process(audio))
    except (LookupError, MicrophoneUnavailableError) as exc:
        print(f"❌ Microphone: {exc}")
        return 3
    except (KeyboardInterrupt, EOFError):
        print("\nbye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
