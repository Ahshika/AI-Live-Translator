"""Local MT with Meta's NLLB-200 running on CTranslate2. Free, offline, 200 languages.

License note: NLLB weights are CC-BY-NC 4.0 (non-commercial). Fine for the MVP/personal use;
a commercial build must swap this provider (the interface makes that a config change).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from app.core import languages
from app.core.config import models_dir
from app.services.translation.base import TranslationProvider, TranslationUnavailableError
from app.utils.cuda import cuda_runtime_ready, register_cuda_dlls

log = logging.getLogger(__name__)


class NLLBProvider(TranslationProvider):
    name = "nllb"

    def __init__(self, model: str = "nllb-200-distilled-1.3B-ct2-int8", device: str = "auto",
                 beam_size: int = 4):
        self.model_dir = Path(model) if Path(model).is_absolute() else models_dir() / model
        self.requested_device = device
        self.beam_size = beam_size
        self.device: str | None = None
        self.compute_type: str | None = None
        self._translator = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._translator is not None:
            return
        if not (self.model_dir / "model.bin").exists():
            raise TranslationUnavailableError(f"NLLB model not found in {self.model_dir}")
        register_cuda_dlls()
        import ctranslate2
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        attempts = []
        if self.requested_device == "cuda" or (self.requested_device == "auto" and cuda_runtime_ready()):
            attempts.append(("cuda", "int8_float16"))
        if self.requested_device in ("auto", "cpu"):
            attempts.append(("cpu", "int8"))
        errors = []
        for device, compute_type in attempts:
            try:
                t0 = time.perf_counter()
                tr = ctranslate2.Translator(str(self.model_dir), device=device, compute_type=compute_type)
                self._translator, self.device, self.compute_type = tr, device, compute_type
                self.translate_batch(["warm up"], "en", "de")  # first CUDA call is slow; pay it now
                log.info("Loaded NLLB on %s/%s in %.1fs", device, compute_type, time.perf_counter() - t0)
                return
            except Exception as exc:
                self._translator = None
                log.warning("NLLB failed on %s/%s: %s", device, compute_type, exc)
                errors.append(f"{device}: {exc}")
        raise TranslationUnavailableError("Could not load NLLB: " + " | ".join(errors))

    def _encode(self, text: str, src_code: str) -> list[str]:
        # NLLB input format: <src_lang> tokens... </s>
        pieces = self._tokenizer.encode(text, add_special_tokens=False).tokens
        return [src_code, *pieces, "</s>"]

    def _decode(self, tokens: list[str]) -> str:
        ids = [self._tokenizer.token_to_id(t) for t in tokens]
        return self._tokenizer.decode([i for i in ids if i is not None], skip_special_tokens=True).strip()

    def translate_batch(self, texts: list[str], source: str, target: str) -> list[str]:
        if self._translator is None:
            self.load()
        src, tgt = languages.get(source).nllb, languages.get(target).nllb
        batch = [self._encode(t, src) for t in texts]
        with self._lock:
            results = self._translator.translate_batch(
                batch,
                target_prefix=[[tgt]] * len(batch),
                beam_size=self.beam_size if self.device != "cpu" else min(self.beam_size, 2),
                max_decoding_length=max(len(b) for b in batch) * 2 + 16,
                repetition_penalty=1.1,  # NLLB sometimes loops on short/noisy STT input
                no_repeat_ngram_size=4,  # ...and repeats whole phrases ("we go we go we go")
            )
        # Drop the forced target-language token from each hypothesis.
        return [self._decode(r.hypotheses[0][1:]) for r in results]

    def supports(self, code: str) -> bool:
        return code in languages.LANGUAGES
