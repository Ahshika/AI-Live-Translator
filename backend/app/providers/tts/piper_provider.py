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
import time
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
# The online voice list is optional (only needed to download a new language's voice). Offline,
# give up fast and don't ask again for a while — the app must stay instant without internet.
CATALOG_TIMEOUT_S = 8
CATALOG_RETRY_AFTER_S = 300
_catalog_failed_at: dict[Path, float] = {}  # voices folder -> when fetching the list last failed
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
        """rhasspy/piper-voices' voices.json (cached on disk). Raises ConnectionError offline."""
        if self._remote is None:
            path = self.voices_dir / "voices.json"
            if not path.exists():
                failed = _catalog_failed_at.get(self.voices_dir)
                if failed is not None and time.monotonic() - failed < CATALOG_RETRY_AFTER_S:
                    raise ConnectionError("voice list unavailable (offline)")
                try:
                    download(VOICES_URL + "voices.json", path, retries=2, timeout=CATALOG_TIMEOUT_S)
                except (OSError, ConnectionError) as exc:
                    _catalog_failed_at[self.voices_dir] = time.monotonic()
                    raise ConnectionError(f"voice list unavailable: {exc}") from exc
            try:
                self._remote = json.loads(path.read_text(encoding="utf-8"))
            except ValueError as exc:  # truncated / corrupted cache: fetch it again next time
                path.unlink(missing_ok=True)
                raise ConnectionError(f"voice list is corrupted: {exc}") from exc
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

    def voice_choices(self, language: str) -> list[dict]:
        """Installed voices plus (when the list is reachable) the ones that can be downloaded."""
        base = language.split("-")[0]
        out = {v.id: {"id": v.id, "name": v.name, "quality": v.quality, "gender": v.gender,
                      "installed": True, "size_mb": None} for v in self.voices(base)}
        try:
            remote = [v for v in self.remote_catalog().values() if v["language"]["family"] == base]
        except (OSError, ConnectionError):
            remote = []
        for v in remote:
            if v["key"] in out:
                continue
            size = sum(f.get("size_bytes", 0) for name, f in v.get("files", {}).items() if name.endswith(".onnx"))
            out[v["key"]] = {"id": v["key"], "name": v.get("name") or v["key"].split("-")[1],
                             "quality": v.get("quality"), "gender": _GENDER.get(v.get("name", "")),
                             "installed": False, "size_mb": round(size / 1e6) or None,
                             "speakers": v.get("num_speakers", 1)}
        rank = lambda c: (not c["installed"], _QUALITY_RANK.get(c["quality"] or "", 9), c["id"])  # noqa: E731
        return sorted(out.values(), key=rank)

    def install_voice(self, voice_id: str, progress=None) -> str:
        """Download one specific voice (from voice_choices) if it isn't installed yet."""
        self.load()
        if voice_id in self._catalog:
            return voice_id
        entry = self.remote_catalog().get(voice_id)
        if entry is None:
            raise NoVoiceError(f"Unknown voice {voice_id!r}")
        for rel in entry["files"]:
            if rel.endswith((".onnx", ".onnx.json")):
                download(VOICES_URL + rel, self.voices_dir / Path(rel).name, progress=progress)
        self._register(self.voices_dir / f"{voice_id}.onnx")
        return voice_id

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
