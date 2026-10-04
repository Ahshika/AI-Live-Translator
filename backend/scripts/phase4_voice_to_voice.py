"""Phase 4 — Full voice-to-voice translation, one direction, push-to-talk.

You speak (Arabic) -> you hear the translation (German) from the chosen output device.
Use headphones: with speakers the mic can pick up the translated voice.

Usage (from backend/):
    .venv/Scripts/python scripts/phase4_voice_to_voice.py                       # ar-EG -> de
    .venv/Scripts/python scripts/phase4_voice_to_voice.py --direction incoming  # de -> ar
    .venv/Scripts/python scripts/phase4_voice_to_voice.py --from auto --to ar
    .venv/Scripts/python scripts/phase4_voice_to_voice.py --file clip.wav --save out.wav
    .venv/Scripts/python scripts/phase4_voice_to_voice.py --output "Headphones" --voice de_DE-kerstin-low
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import _cli  # noqa: F401  (sets sys.path + utf-8 stdout)
import numpy as np
import soundfile as sf

from app.core import languages
from app.core.config import Settings
from app.pipeline.factory import build_voice_translator
from app.pipeline.voice_translator import VoiceResult
from app.services.audio.capture import MicrophoneUnavailableError
from app.services.audio.playback import AudioPlayer, SpeakerUnavailableError
from app.services.stt.base import STTUnavailableError
from app.services.translation.base import TranslationUnavailableError
from app.services.tts.base import AudioChunk, TTSUnavailableError


def report(r: VoiceResult) -> None:
    if r.skipped:
        print("   (no speech recognised)\n")
        return
    src, tgt = languages.get(r.language.code), languages.get(r.target_language)
    print(f"\n  {src.flag} {r.transcript.text}")
    print(f"  {tgt.flag} {r.translated_text}")
    first = f"{r.first_audio_ms:.0f}ms" if r.first_audio_ms is not None else "-"
    print(f"     ⏱ first translated audio after {first}  (stt {r.stt_ms:.0f}ms, "
          f"all done {r.total_ms:.0f}ms)  · you spoke {r.audio_seconds:.1f}s → speech {r.speech_seconds:.1f}s\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--direction", choices=["outgoing", "incoming"], default="outgoing")
    p.add_argument("--from", dest="source")
    p.add_argument("--to", dest="target")
    p.add_argument("--voice")
    p.add_argument("--speed", type=float)
    p.add_argument("--input", help="mic name substring or index")
    p.add_argument("--output", help="output device name substring or index")
    p.add_argument("--file", type=Path, help="use speech from an audio file instead of the mic")
    p.add_argument("--save", type=Path, help="write the translated speech to a WAV file")
    p.add_argument("--no-play", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    s = Settings.from_env(input_device=args.input, output_device=args.output)
    print("Loading models (speech recognition, translation, voice)...")
    try:
        vt = build_voice_translator(s, args.direction, source=args.source, target=args.target,
                                    voice=args.voice, speed=args.speed)
    except languages.UnsupportedLanguageError as exc:
        print(f"❌ {exc}")
        return 1
    except (STTUnavailableError, TranslationUnavailableError, TTSUnavailableError) as exc:
        print(f"❌ {exc}")
        return 2
    src = vt.translator.resolver.configured
    print(f"✅ Ready:  {'Auto 🌐' if src == languages.AUTO else languages.get(src).name}  →  "
          f"{languages.get(vt.target).name}")

    player = None
    if not args.no_play:
        try:
            player = AudioPlayer(s.output_device)
            player.start()
            print(f"🔊 Output: {player.device_name}")
        except (LookupError, SpeakerUnavailableError) as exc:
            print(f"❌ Speaker: {exc}")
            return 3

    saved: list[AudioChunk] = []

    def sink(chunk: AudioChunk) -> None:
        saved.append(chunk)
        if player:
            player.enqueue(chunk)

    def finish() -> None:
        if args.save and saved:
            sf.write(args.save, np.concatenate([c.samples for c in saved]), saved[0].sample_rate)
            print(f"     saved → {args.save}")
        saved.clear()

    try:
        if args.file:
            report(vt.process(_cli.load_audio_file(args.file), sink))
            finish()
            return 0
        mic = _cli.open_mic(s.input_device)
        print("🎧 Use headphones so the mic doesn't hear the translation.")
        with mic:
            while True:
                audio = _cli.record_push_to_talk(mic)
                if len(audio) < 0.3 * 16_000:
                    print("   too short, try again")
                    continue
                if player:
                    player.stop()  # new utterance takes priority over anything still playing
                report(vt.process(audio, sink))
                finish()
    except (LookupError, MicrophoneUnavailableError) as exc:
        print(f"❌ Microphone: {exc}")
        return 3
    except (KeyboardInterrupt, EOFError):
        print("\nbye.")
    finally:
        if player:
            player.wait(timeout=30)
            player.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
