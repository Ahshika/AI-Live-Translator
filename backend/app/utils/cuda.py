"""CUDA runtime discovery shared by every CTranslate2-based provider."""

from __future__ import annotations

import os
import sys
from functools import cache
from pathlib import Path


@cache
def register_cuda_dlls() -> None:
    """Make the pip-installed CUDA libs (nvidia-cublas-cu12 / nvidia-cudnn-cu12) loadable on Windows."""
    if sys.platform != "win32":
        return
    dirs: list[Path] = []
    try:
        import nvidia  # namespace package from the nvidia-* wheels (development)

        dirs += [b for root in map(Path, nvidia.__path__) for b in root.glob("*/bin")]
    except ImportError:
        pass
    from app.engine.assets import cuda_dir  # installed app: downloaded on first run

    if cuda_dir().exists():
        dirs.append(cuda_dir())
    for bin_dir in dirs:
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


# CTranslate2 needs these at run time. If they're missing, a CUDA model can *hang* on its
# first inference instead of raising (seen with faster-whisper), so we check up front.
_RUNTIME_DLLS = ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll")


@cache
def cuda_runtime_ready() -> bool:
    """NVIDIA GPU present AND the CUDA/cuDNN libraries can actually be loaded."""
    register_cuda_dlls()
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() == 0:
            return False
    except Exception:
        return False
    if sys.platform == "win32":
        import ctypes

        for dll in _RUNTIME_DLLS:
            try:
                ctypes.WinDLL(dll)
            except OSError:
                return False
    return True


def cuda_available() -> bool:
    return cuda_runtime_ready()
