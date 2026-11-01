"""Detector de bovinos para a API. Inferência por ONNX Runtime (sem torch/ultralytics no caminho de inferência)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from cattlecount.onnx_infer import OnnxDetector
from cattlecount.tiling import sliced_predict

Detector = Callable[[np.ndarray, float], np.ndarray]    # (rgb HxWx3, conf) -> (n,5) [x1,y1,x2,y2,score]


class OnnxCattleDetector:
    def __init__(self, path: str | Path, mode: str = "sliced", size: int = 1024):
        self.path, self.mode = str(path), mode
        self.det = OnnxDetector(self.path, size=size)
        self.id = Path(path).stem

    def _tiles(self, conf):
        return lambda tiles: [np.array([[*d.xyxy, d.score] for d in self.det(t, conf=conf, iou=0.6)], float).reshape(-1, 5) for t in tiles]

    def __call__(self, rgb: np.ndarray, conf: float = 0.25) -> np.ndarray:
        if self.mode == "sliced" and max(rgb.shape[:2]) > 1280:
            return sliced_predict(self._tiles(conf), rgb, 1024, 0.2)
        return self._tiles(conf)([rgb])[0]


def load_detector(settings) -> tuple[Detector | None, str]:
    if Path(settings.model_onnx).exists():
        d = OnnxCattleDetector(settings.model_onnx, settings.detect_mode)
        return d, d.id
    return None, ""
