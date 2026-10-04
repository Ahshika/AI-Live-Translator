"""Streaming Silero VAD (v6 ONNX, shipped inside faster-whisper — no extra download).

faster-whisper runs this model over a whole file at once; we need it frame by frame, so we
keep its recurrent state (h, c) and the 64-sample context between calls ourselves.
~0.1 ms of CPU per 32 ms window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.services.vad.base import VoiceActivityDetector


def _model_path() -> Path:
    import faster_whisper

    return Path(faster_whisper.__file__).parent / "assets" / "silero_vad_v6.onnx"


class SileroVAD(VoiceActivityDetector):
    window_samples = 512  # 32 ms @ 16 kHz (fixed by the model)
    _CONTEXT = 64

    def __init__(self, model_path: Path | None = None):
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(str(model_path or _model_path()),
                                                     providers=["CPUExecutionProvider"], sess_options=opts)
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(self._CONTEXT, dtype=np.float32)

    def __call__(self, window: np.ndarray) -> float:
        if len(window) != self.window_samples:
            raise ValueError(f"Silero needs exactly {self.window_samples} samples, got {len(window)}")
        x = np.concatenate([self._context, window.astype(np.float32, copy=False)])[None, :]
        out, self._h, self._c = self._session.run(None, {"input": x, "h": self._h, "c": self._c})
        self._context = window[-self._CONTEXT:].astype(np.float32, copy=True)
        return float(np.asarray(out).reshape(-1)[0])
