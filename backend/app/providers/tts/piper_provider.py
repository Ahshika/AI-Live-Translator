"""Local TTS with Piper (VITS on onnxruntime). Free, offline, fast on CPU.

Voices are .onnx + .onnx.json pairs in backend/models/piper (from rhasspy/piper-voices).
Arabic text is diacritised automatically (Piper's built-in tashkeel), which matters a lot:
undiacritised Arabic is pronounced badly by any phoneme-based TTS.
License: piper-tts is GPL-3.0; each voice has its own license (see its MODEL_CARD).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Iterator

import numpy as np

from app.core.config import models_dir
from app.services.audio.format import trim_silence
from app.utils.download import download
from app.services.tts.base import (
    AudioChunk,
    NoVoiceError,
    TextToSpeechProvider,
    TTSUnavailableError,
    Voice,
)

log = logging.getLogger(__name__)

VOICES_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
_QUALITY_RANK = {"high": 0, "medium": 1, "low": 2, "x_low": 3}
# For automatic downloads prefer "medium": nearly as natural as "high", half the size, faster.
_DOWNLOAD_RANK = {"medium": 0, "high": 1, "low": 2, "x_low": 3}
# Piper's voice configs don't record gender; known speakers of the voices we ship.
_GENDER = {"kareem": "male", "thorsten": "male", "karlsson": "male", "pavoque": "male",
           "kerstin": "female", "ramona": "female", "eva_k": "female", "lessac": "female"}


class PiperProvider(TextToSpeechProvider):
    name = "piper"

    def __init__(self, voices_dir: Path | None = None, preferred: dict[str, str] | None = None,
                 auto_download: bool = True):
        self.voices_dir = Path(voices_dir) if voices_dir else models_dir() / "piper"
        self.preferred = preferred or {}  # language -> voice id
        self.auto_download = auto_download
        self._catalog: dict[str, tuple[Voice, Path]] = {}
        self._remote: dict | None = None
        self._loaded: dict[str, object] = {}
        self._lock = threading.Lock()
        self._scanned = False

    def load(self) -> None:
        if self._scanned:
            return
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        for onnx in sorted(self.voices_dir.glob("*.onnx")):
            self._register(onnx)
        self._scanned = True
        if not self._catalog and not self.auto_download:
            raise TTSUnavailableError(f"No Piper voices found in {self.voices_dir}")
        log.info("Piper voices: %s", ", ".join(self._catalog) or "(none yet)")

    def _register(self, onnx: Path) -> None:
        cfg_path = onnx.with_name(onnx.name + ".json")
        if not cfg_path.exists():
            return
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        vid = onnx.stem  # e.g. de_DE-thorsten-high
        speaker = vid.split("-")[1] if vid.count("-") >= 2 else vid
        lang = cfg.get("language", {}).get("family") or vid.split("_")[0]
        quality = cfg.get("audio", {}).get("quality") or vid.rsplit("-", 1)[-1]
        self._catalog[vid] = (Voice(vid, lang, speaker, _GENDER.get(speaker), quality), onnx)

    def voices(self, language: str | None = None) -> list[Voice]:
        """Installed voices, best first."""
        self.load()
        base = language.split("-")[0] if language else None
        found = [v for v, _ in self._catalog.values() if base is None or v.language == base]
        return sorted(found, key=lambda v: (v.language, _QUALITY_RANK.get(v.quality or "", 9), v.id))

    # -- online catalogue (rhasspy/piper-voices) ----------------------------
    def remote_catalog(self) -> dict:
        if self._remote is None:
            path = self.voices_dir / "voices.json"
            if not path.exists():
                download(VOICES_URL + "voices.json", path)
            self._remote = json.loads(path.read_text(encoding="utf-8"))
        return self._remote

    def downloadable_languages(self) -> set[str]:
        try:
            return {v["language"]["family"] for v in self.remote_catalog().values()}
        except (OSError, ConnectionError):
            return set()

    def has_voice(self, language: str) -> bool:
        """Installed, or available for automatic download."""
        base = language.split("-")[0]
        return bool(self.voices(base)) or (self.auto_download and base in self.downloadable_languages())

    def ensure_voice(self, language: str, progress=None) -> str:
        """Return an installed voice id for the language, downloading the best one if needed."""
        self.load()
        base = language.split("-")[0]
        if self.preferred.get(base) in self._catalog:
            return self.preferred[base]
        installed = self.voices(base)
        if installed:
            return installed[0].id
        if not self.auto_download:
            raise NoVoiceError(f"No installed voice for language {language!r}")
        try:
            remote = [v for v in self.remote_catalog().values() if v["language"]["family"] == base]
        except (OSError, ConnectionError) as exc:
            raise NoVoiceError(f"No voice for {language!r} and the voice list can't be downloaded: {exc}") from exc
        if not remote:
            raise NoVoiceError(f"No voice exists for language {language!r}")
        wanted = [v for v in remote if v["key"] == self.preferred.get(base)]
        best = wanted[0] if wanted else min(remote, key=lambda v: (_DOWNLOAD_RANK.get(v["quality"], 9), v["num_speakers"] > 1, v["key"]))
        log.info("Downloading voice %s for %s", best["key"], language)
        for rel in best["files"]:
            if rel.endswith((".onnx", ".onnx.json")):
                download(VOICES_URL + rel, self.voices_dir / Path(rel).name, progress=progress)
        self._register(self.voices_dir / f"{best['key']}.onnx")
        return best["key"]

    def _pick(self, language: str, voice: str | None) -> str:
        if voice:
            self.load()
            if voice not in self._catalog:
                raise NoVoiceError(f"Voice {voice!r} is not installed")
            return voice
        return self.ensure_voice(language)

    def _voice_model(self, vid: str):
        with self._lock:
            if vid not in self._loaded:
                from piper import PiperVoice

                self._loaded[vid] = PiperVoice.load(self._catalog[vid][1], use_cuda=False)
            return self._loaded[vid]

    def warm_up(self, language: str, voice: str | None = None) -> None:
        """Load the voice (and tashkeel for Arabic) before the first real sentence."""
        list(self.synthesize_stream("ok" if not language.startswith("ar") else "نعم", language, voice=voice))

    def synthesize_stream(self, text: str, language: str, *, voice: str | None = None,
                          speed: float = 1.0) -> Iterator[AudioChunk]:
        if not text.strip():
            return
        self.load()
        from piper import SynthesisConfig

        model = self._voice_model(self._pick(language, voice))
        cfg = SynthesisConfig(length_scale=1.0 / max(0.5, min(speed, 2.0)))
        # Piper splits text into sentences itself and yields one chunk per sentence.
        for chunk in model.synthesize(text, syn_config=cfg):
            samples = np.asarray(chunk.audio_float_array, dtype=np.float32).reshape(-1)
            # Arabic voices with tashkeel pad ~0.3 s of silence up front: pure latency.
            samples = trim_silence(samples, chunk.sample_rate)
            if samples.size:
                yield AudioChunk(samples, chunk.sample_rate)
