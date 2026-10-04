"""First-run downloads: AI models (+ the NVIDIA runtime on GPU machines).

The installer stays small (~150 MB); the ~3–5 GB of models come down once, resumably,
with progress shown in the app. Machines without an NVIDIA GPU get a smaller speech model
that runs acceptably on the CPU.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from app.core.config import Settings, is_frozen, models_dir
from app.utils.download import download

log = logging.getLogger(__name__)
HF = "https://huggingface.co"
GPU_STT_MODEL = "large-v3-turbo"
CPU_STT_MODEL = "small"
WHISPER_REPOS = {
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
}
NLLB_REPOS = {"nllb-200-distilled-1.3B-ct2-int8": "OpenNMT/nllb-200-distilled-1.3B-ct2-int8"}
# NVIDIA runtime for CTranslate2 (CUDA 12): same packages the dev environment uses.
CUDA_WHEELS = ("nvidia-cublas-cu12", "nvidia-cudnn-cu12")
Progress = Callable[[dict], None]


def has_nvidia_gpu() -> bool:
    """An NVIDIA driver is installed (doesn't need CUDA itself)."""
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    return (system32 / "nvcuda.dll").exists()


def cuda_dir() -> Path:
    return models_dir().parent / "cuda"


@dataclass
class Component:
    id: str
    title_ar: str
    size_mb: int
    installed: bool
    files: list[tuple[str, str]] = field(default_factory=list, repr=False)  # (url, dest)


def _hf_files(repo: str, dest: Path) -> tuple[list[tuple[str, str]], int]:
    with urllib.request.urlopen(f"{HF}/api/models/{repo}/tree/main", timeout=30) as r:
        tree = json.load(r)
    files = [(f"{HF}/{repo}/resolve/main/{f['path']}", str(dest / f["path"]), f.get("size", 0))
             for f in tree if f["type"] == "file" and not f["path"].startswith(".") and f["path"] != "README.md"]
    return [(u, d) for u, d, _ in files], sum(s for *_, s in files)


def _cuda_installed() -> bool:
    if not is_frozen():
        return True  # dev: the venv's nvidia-* packages provide it
    d = cuda_dir()
    return any(d.glob("cublas64_*.dll")) and any(d.glob("cudnn64_*.dll"))


def plan(settings: Settings) -> list[Component]:
    """What this machine needs, and what's already there (no network needed)."""
    md = models_dir()
    whisper = md / f"faster-whisper-{settings.stt_model}"
    nllb = md / settings.translation_model
    voice_langs = list(settings.preferred_voices())
    piper_dir = md / "piper"

    def has_voice(lang: str) -> bool:  # any voice for the language will do
        return any(piper_dir.glob(f"{lang}_*.onnx"))

    comps = [
        Component("stt", "نموذج فهم الكلام", 1600 if settings.stt_model == GPU_STT_MODEL else 480,
                  (whisper / "model.bin").exists()),
        Component("mt", "نموذج الترجمة", 1380, (nllb / "model.bin").exists()),
        Component("voices", "الأصوات", 130, all(has_voice(lang) for lang in voice_langs)),
    ]
    if has_nvidia_gpu():
        comps.append(Component("cuda", "مكتبات كرت الشاشة NVIDIA", 1300, _cuda_installed()))
    return comps


def recommended_stt_model() -> str:
    return GPU_STT_MODEL if has_nvidia_gpu() else CPU_STT_MODEL


class AssetInstaller:
    """Downloads every missing component; reports progress; safe to call again after a failure."""

    def __init__(self, settings: Settings, on_progress: Progress | None = None):
        self.settings = settings
        self.on_progress = on_progress or (lambda e: None)
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self.error = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="asset-installer")
        self._thread.start()

    def _emit(self, **data) -> None:
        self.on_progress({"type": "setup_progress", **data})

    def _run(self) -> None:
        try:
            for comp in plan(self.settings):
                if comp.installed:
                    continue
                self._emit(component=comp.id, title=comp.title_ar, done=0, total=comp.size_mb * 1_000_000)
                getattr(self, f"_install_{comp.id}")(comp)
            # The GPU check ran (and was cached) before the CUDA libraries existed: redo it.
            from app.utils import cuda

            cuda.register_cuda_dlls.cache_clear()
            cuda.cuda_runtime_ready.cache_clear()
            self._emit(component="all", finished=True)
        except Exception as exc:  # network, disk full, ...
            log.exception("asset download failed")
            self.error = str(exc)
            self._emit(component="all", finished=False, error=str(exc))

    def _progress_for(self, comp: Component, base: int = 0):
        def report(done: int, total: int | None) -> None:
            self._emit(component=comp.id, title=comp.title_ar, done=base + done, total=comp.size_mb * 1_000_000)
        return report

    def _download_repo(self, comp: Component, repo: str, dest: Path) -> None:
        files, _ = _hf_files(repo, dest)
        base = 0
        for url, path in files:
            target = Path(path)
            download(url, target, progress=self._progress_for(comp, base))
            base += target.stat().st_size

    def _install_stt(self, comp: Component) -> None:
        model = self.settings.stt_model
        self._download_repo(comp, WHISPER_REPOS[model], models_dir() / f"faster-whisper-{model}")

    def _install_mt(self, comp: Component) -> None:
        model = self.settings.translation_model
        self._download_repo(comp, NLLB_REPOS[model], models_dir() / model)

    def _install_voices(self, comp: Component) -> None:
        from app.providers.tts.piper_provider import PiperProvider

        tts = PiperProvider(preferred=self.settings.preferred_voices())
        for lang in self.settings.preferred_voices():
            tts.ensure_voice(lang, progress=self._progress_for(comp))

    def _install_cuda(self, comp: Component) -> None:
        dest = cuda_dir()
        dest.mkdir(parents=True, exist_ok=True)
        base = 0
        for package in CUDA_WHEELS:
            url = _wheel_url(package)
            wheel = dest / "downloads" / url.rsplit("/", 1)[1]
            download(url, wheel, progress=self._progress_for(comp, base))
            base += wheel.stat().st_size
            with zipfile.ZipFile(wheel) as z:  # a wheel is a zip: take only the DLLs
                for name in z.namelist():
                    if name.lower().endswith(".dll") and "/bin/" in name:
                        with z.open(name) as src, open(dest / Path(name).name, "wb") as out:
                            shutil.copyfileobj(src, out)
            wheel.unlink()


def _wheel_url(package: str) -> str:
    """The Windows x64 wheel of the version we were built and tested with."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        pinned = version(package)
    except PackageNotFoundError:
        pinned = None
    meta_url = f"https://pypi.org/pypi/{package}/{pinned}/json" if pinned else f"https://pypi.org/pypi/{package}/json"
    with urllib.request.urlopen(meta_url, timeout=30) as r:
        meta = json.load(r)
    for f in meta["urls"]:
        if f["filename"].endswith("win_amd64.whl"):
            return f["url"]
    raise RuntimeError(f"No Windows build of {package}")


def status(settings: Settings) -> dict:
    comps = plan(settings)
    return {"ready": all(c.installed for c in comps),
            "components": [{k: v for k, v in asdict(c).items() if k != "files"} for c in comps]}
