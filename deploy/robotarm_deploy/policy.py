"""Policy runner — load an exported policy and run deterministic inference.

Framework-agnostic front end over the two supported export formats:

  * **TorchScript** (``policy.pt``) via ``torch.jit.load`` — recommended default: numerics
    identical to training, self-contained, runs anywhere libtorch/torch runs.
  * **ONNX** (``policy.onnx``) via ``onnxruntime`` — for lightweight/edge runtimes without
    a full PyTorch install.

The runner is deliberately dumb: obs vector in -> raw action vector out, plus latency
timing. It knows nothing about ROS, hardware, or the action pipeline.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np

from . import contract as C


class LatencyStats:
    """Rolling inference-latency tracker (milliseconds)."""

    def __init__(self, window: int = 1000) -> None:
        self._buf: deque[float] = deque(maxlen=window)

    def add(self, ms: float) -> None:
        self._buf.append(ms)

    def snapshot(self) -> dict[str, float]:
        if not self._buf:
            return {"count": 0, "mean_ms": float("nan"), "p50_ms": float("nan"),
                    "p99_ms": float("nan"), "max_ms": float("nan")}
        arr = np.array(self._buf)
        return {
            "count": len(arr),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(arr.max()),
        }


class PolicyRunner:
    """Load an exported policy and run single-step deterministic inference."""

    def __init__(self, model_path: str, backend: str | None = None) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"exported policy not found: {self.model_path}")
        self.backend = backend or self._infer_backend(self.model_path)
        self.latency = LatencyStats()
        self._load()

    @staticmethod
    def _infer_backend(path: Path) -> str:
        if path.suffix == ".onnx":
            return "onnx"
        if path.suffix in (".pt", ".jit", ".ts"):
            return "torchscript"
        raise ValueError(f"cannot infer backend from '{path.name}'; pass backend=")

    def _load(self) -> None:
        if self.backend == "torchscript":
            import torch
            self._torch = torch
            self._model = torch.jit.load(str(self.model_path), map_location="cpu")
            self._model.eval()
        elif self.backend == "onnx":
            import onnxruntime as ort
            self._sess = ort.InferenceSession(str(self.model_path),
                                              providers=["CPUExecutionProvider"])
            self._in_name = self._sess.get_inputs()[0].name
        else:
            raise ValueError(f"unknown backend: {self.backend}")

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """Run one deterministic forward pass. ``obs`` is a length-28 vector.

        Returns the raw (un-clipped, un-scaled) 6-dim action mean.
        """
        obs = np.asarray(obs, dtype=np.float32).reshape(1, C.OBS_DIM)
        t0 = time.perf_counter()
        if self.backend == "torchscript":
            with self._torch.inference_mode():
                out = self._model(self._torch.from_numpy(obs)).cpu().numpy()
        else:
            out = self._sess.run(None, {self._in_name: obs})[0]
        self.latency.add((time.perf_counter() - t0) * 1e3)
        return np.asarray(out, dtype=np.float32).reshape(-1)[:C.ACTION_DIM]
