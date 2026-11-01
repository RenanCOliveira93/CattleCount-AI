"""Simulação de câmera em movimento sobre fotos REAIS anotadas: uma janela desliza sobre a imagem e gera quadros com GT exato de trilhas.
ATENÇÃO: é movimento de câmera SINTÉTICO (animais estáticos, sem oclusão/comportamento real). Serve para validar rastreador, compensação de
movimento e contagem de linha com verdade exata — NÃO substitui vídeo aéreo real anotado. Todo material gerado aqui deve ser rotulado como sintético."""
from __future__ import annotations

import numpy as np


def pan_windows(img_hw, win_wh, n_frames: int, start_xy, end_xy) -> list[tuple[int, int]]:
    h, w = img_hw
    ww, wh = win_wh
    xs = np.linspace(start_xy[0], end_xy[0], n_frames)
    ys = np.linspace(start_xy[1], end_xy[1], n_frames)
    return [(int(np.clip(round(x), 0, w - ww)), int(np.clip(round(y), 0, h - wh))) for x, y in zip(xs, ys)]


def render_pan(image: np.ndarray, boxes: np.ndarray, win_wh, offsets, min_visible: float = 0.5):
    """Retorna (frames, gt) — gt: (N,6) [frame(1-based), id, x1, y1, x2, y2] em coordenadas da janela; id = índice da caixa (1-based).
    Caixas só entram se ≥ `min_visible` da área estiver dentro da janela."""
    ww, wh = win_wh
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    frames, rows = [], []
    for f, (ox, oy) in enumerate(offsets, 1):
        frames.append(np.ascontiguousarray(image[oy:oy + wh, ox:ox + ww]))
        b = boxes - [ox, oy, ox, oy]
        c = np.stack([np.clip(b[:, 0], 0, ww), np.clip(b[:, 1], 0, wh), np.clip(b[:, 2], 0, ww), np.clip(b[:, 3], 0, wh)], 1)
        vis = np.maximum(c[:, 2] - c[:, 0], 0) * np.maximum(c[:, 3] - c[:, 1], 0) / np.maximum(area, 1e-9)
        for i in np.where(vis >= min_visible)[0]:
            rows.append([f, i + 1, *c[i]])
    return frames, np.array(rows, float).reshape(-1, 6)


def noisy_detections(gt: np.ndarray, n_frames: int, seed: int = 0, recall: float = 0.9, jitter_frac: float = 0.05, fp_per_frame: float = 0.1,
                     occlusion: list[tuple[int, int, int]] | None = None, win_wh=(1280, 720)) -> dict[int, np.ndarray]:
    """Detecções SIMULADAS a partir do GT (perda aleatória, jitter, falsos positivos, janelas de oclusão (id, f0, f1)). Para stress-test do
    rastreador sob condições controladas — não representa um detector real."""
    rng = np.random.default_rng(seed)
    occ = occlusion or []
    out: dict[int, list] = {f: [] for f in range(1, n_frames + 1)}
    for r in gt:
        f, tid, box = int(r[0]), int(r[1]), r[2:6]
        if any(tid == o[0] and o[1] <= f <= o[2] for o in occ) or rng.random() > recall:
            continue
        wh = np.array([box[2] - box[0], box[3] - box[1]])
        j = rng.normal(0, jitter_frac, 4) * np.tile(wh, 2)
        out[f].append([*(box + j), float(np.clip(rng.normal(0.8, 0.1), 0.15, 0.99))])
    for f in out:
        for _ in range(rng.poisson(fp_per_frame)):
            x, y = rng.uniform(0, win_wh[0] - 40), rng.uniform(0, win_wh[1] - 40)
            out[f].append([x, y, x + 25, y + 25, float(rng.uniform(0.15, 0.6))])
    return {f: np.array(v, float).reshape(-1, 5) for f, v in out.items()}
