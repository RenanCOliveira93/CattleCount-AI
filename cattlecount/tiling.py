"""Inferência por tiles (fatiada) na resolução nativa: necessária porque os bovinos aéreos têm ≈ 0,015% da imagem
(≈ 40–55 px em 4000–5280 px de largura; ≈ 7 px se a imagem inteira fosse reduzida para 640).

`sliced_predict` roda um detector (callable sobre lista de tiles) e funde as caixas. Para evitar caixas cortadas na
junção entre tiles, descartam-se detecções que tocam a borda interna de um tile; o vizinho (com sobreposição maior que um
animal) as vê inteiras. Depois, NMS fundindo duplicatas.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .geometry import iou_matrix


def tile_grid(h: int, w: int, tile: int = 1024, overlap: float = 0.2) -> list[tuple[int, int, int, int]]:
    """Lista (x0, y0, x1, y1) que cobre a imagem; os últimos tiles são encostados na borda (nunca menores que `tile`, se a imagem couber)."""
    step = max(1, int(tile * (1 - overlap)))

    def axis(n):
        if n <= tile:
            return [(0, n)]
        starts = list(range(0, n - tile + 1, step))
        if starts[-1] + tile < n:
            starts.append(n - tile)
        return [(s, s + tile) for s in starts]

    return [(x0, y0, x1, y1) for (y0, y1) in axis(h) for (x0, x1) in axis(w)]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5, ios_thr: float | None = 0.8) -> list[int]:
    """NMS por IoU e, opcionalmente, por IoS (interseção sobre a menor área) — remove caixas contidas em outras na emenda."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    for i in order:
        ok = True
        for j in keep:
            iou = iou_matrix(boxes[i], boxes[j])[0, 0]
            if iou > iou_thr:
                ok = False
                break
            if ios_thr is not None:
                lt = np.maximum(boxes[i, :2], boxes[j, :2]); rb = np.minimum(boxes[i, 2:], boxes[j, 2:])
                inter = float(np.clip(rb - lt, 0, None).prod())
                small = min((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]), (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1]))
                if small > 0 and inter / small > ios_thr:
                    ok = False
                    break
        if ok:
            keep.append(int(i))
    return keep


def sliced_predict(detect_tiles: Callable[[list[np.ndarray]], list[np.ndarray]], image: np.ndarray, tile: int = 1024, overlap: float = 0.2,
                   iou_thr: float = 0.5, border_margin: int = 4, batch: int = 4) -> np.ndarray:
    """image RGB HxWx3 → (n,5) [x1,y1,x2,y2,score] em pixels da imagem inteira.
    `detect_tiles(list[tile_img]) -> list[(m,5) em coordenadas do tile]`."""
    h, w = image.shape[:2]
    grid = tile_grid(h, w, tile, overlap)
    allb = []
    for i in range(0, len(grid), batch):
        chunk = grid[i:i + batch]
        res = detect_tiles([image[y0:y1, x0:x1] for x0, y0, x1, y1 in chunk])
        for (x0, y0, x1, y1), d in zip(chunk, res):
            d = np.asarray(d, float).reshape(-1, 5)
            if not len(d):
                continue
            th, tw = y1 - y0, x1 - x0
            touch = np.zeros(len(d), bool)
            if x0 > 0: touch |= d[:, 0] <= border_margin
            if y0 > 0: touch |= d[:, 1] <= border_margin
            if x1 < w: touch |= d[:, 2] >= tw - border_margin
            if y1 < h: touch |= d[:, 3] >= th - border_margin
            d = d[~touch]
            d[:, [0, 2]] += x0
            d[:, [1, 3]] += y0
            allb.append(d)
    if not allb:
        return np.zeros((0, 5))
    d = np.concatenate(allb)
    return d[nms(d[:, :4], d[:, 4], iou_thr)]
