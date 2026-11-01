"""Geometria usada na contagem: caixas, linha virtual, polígono (ROI)."""
from __future__ import annotations

import numpy as np


def xyxy_to_xywh(b):
    b = np.asarray(b, float)
    return np.c_[b[..., 0], b[..., 1], b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]]


def center(b) -> np.ndarray:
    b = np.asarray(b, float).reshape(-1, 4)
    return np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]


def iou_matrix(a, b) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    u = aa[:, None] + ab[None, :] - inter
    return np.where(u > 0, inter / np.maximum(u, 1e-12), 0.0)


def side_of_line(p, a, b) -> np.ndarray:
    """Sinal do lado de p em relação à reta orientada a→b (>0 esquerda no sistema de imagem y↓ é 'direita' visual; usamos só o sinal)."""
    p = np.asarray(p, float).reshape(-1, 2)
    return (b[0] - a[0]) * (p[:, 1] - a[1]) - (b[1] - a[1]) * (p[:, 0] - a[0])


def segments_intersect(p1, p2, a, b) -> bool:
    """O segmento p1→p2 cruza o SEGMENTO a→b (não a reta infinita)?"""
    d1 = side_of_line([p1], a, b)[0]
    d2 = side_of_line([p2], a, b)[0]
    if d1 * d2 > 0 or (d1 == 0 and d2 == 0):
        return False
    d3 = (p2[0] - p1[0]) * (a[1] - p1[1]) - (p2[1] - p1[1]) * (a[0] - p1[0])
    d4 = (p2[0] - p1[0]) * (b[1] - p1[1]) - (p2[1] - p1[1]) * (b[0] - p1[0])
    return d3 * d4 <= 0


def point_in_polygon(pt, poly) -> bool:
    """Ray casting; poly = [(x,y), ...]. Pontos na borda contam como dentro."""
    x, y = pt
    n, inside = len(poly), False
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
        # borda
        if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            if abs((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) < 1e-9:
                return True
    return inside
