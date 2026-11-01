"""Inferência de borda com ONNX Runtime + pré/pós-processamento próprios (numpy/OpenCV). NÃO importa `ultralytics`.

Saída do YOLO11 exportado: (1, 4+nc, N) com cx,cy,w,h em pixels do tensor de entrada e scores de classe já em [0,1].
Este módulo apenas detecta; não toma decisão de aplicação nem controla atuador.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    xyxy: tuple
    score: float
    cls: int


def letterbox(img: np.ndarray, size: int = 640, pad: int = 114):
    """RGB uint8 HxWx3 → (tensor size×size, escala, (pad_x, pad_y)). Mantém proporção."""
    import cv2
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    out = np.full((size, size, 3), pad, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, (px, py)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.6) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        lt = np.maximum(boxes[i, :2], boxes[rest, :2]); rb = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        inter = np.clip(rb - lt, 0, None).prod(1)
        a = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        iou = inter / np.maximum(a(boxes[i]) + a(boxes[rest]) - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def decode(out: np.ndarray, scale: float, pad: tuple, orig_hw: tuple, conf: float = 0.25, iou: float = 0.6, max_det: int = 100) -> list[Detection]:
    p = out[0].T                                  # (N, 4+nc)
    scores = p[:, 4:]
    cls = scores.argmax(1)
    s = scores[np.arange(len(p)), cls]
    m = s >= conf
    p, cls, s = p[m], cls[m], s[m]
    if not len(p):
        return []
    xyxy = np.c_[p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2]
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad[0]) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad[1]) / scale
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, orig_hw[1]); xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, orig_hw[0])
    keep: list[int] = []
    for c in np.unique(cls):                      # NMS por classe
        idx = np.where(cls == c)[0]
        keep += [int(idx[k]) for k in nms(xyxy[idx], s[idx], iou)]
    keep = sorted(keep, key=lambda i: -s[i])[:max_det]
    return [Detection(tuple(map(float, xyxy[i])), float(s[i]), int(cls[i])) for i in keep]


class OnnxDetector:
    def __init__(self, path: str, providers: list[str] | None = None, size: int = 640, intra_threads: int | None = None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if intra_threads:
            so.intra_op_num_threads = intra_threads
        self.sess = ort.InferenceSession(str(path), so, providers=providers or ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.size = size
        self.providers = self.sess.get_providers()

    def __call__(self, img_rgb: np.ndarray, conf: float = 0.25, iou: float = 0.6, timings: dict | None = None) -> list[Detection]:
        t0 = time.perf_counter()
        x, r, pad = letterbox(img_rgb, self.size)
        x = (x.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        t1 = time.perf_counter()
        out = self.sess.run(None, {self.inp: x})[0]
        t2 = time.perf_counter()
        dets = decode(out, r, pad, img_rgb.shape[:2], conf, iou)
        if timings is not None:
            timings.update(pre_ms=(t1 - t0) * 1e3, infer_ms=(t2 - t1) * 1e3, post_ms=(time.perf_counter() - t2) * 1e3)
        return dets
