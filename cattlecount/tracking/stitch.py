"""Costura LIMITADA de tracklets (re-identificação restrita): une o fim de uma trilha ao início de outra quando o intervalo é curto e
a posição/tamanho (e, opcionalmente, a aparência) são compatíveis. Não é re-ID de longo prazo: bovinos parecidos entre si não
permitem identidade por aparência; trabalhamos só com continuidade espaço-temporal. Pode errar nos dois sentidos (unir animais diferentes)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class StitchConfig:
    max_gap: int = 45                # quadros
    max_dist_norm: float = 1.5       # distância (em "tamanhos de caixa") entre a posição extrapolada e o início da outra trilha
    size_ratio: tuple[float, float] = (0.6, 1.7)
    vel_window: int = 5
    use_appearance: bool = False
    max_appearance_dist: float = 0.5  # Bhattacharyya


def _extrapolate(t, gap, k):
    c = (t["boxes"][:, :2] + t["boxes"][:, 2:]) / 2
    v = (c[-1] - c[max(0, len(c) - k)]) / max(1, min(k, len(c)) - 1) if len(c) > 1 else np.zeros(2)
    return c[-1] + v * gap


def stitch(traj: dict[int, dict], cfg: StitchConfig | None = None, appearance: dict[int, tuple[np.ndarray, np.ndarray]] | None = None) -> dict[int, int]:
    """traj: {id: {frames, boxes}} (coordenadas na MESMA referência; com câmera móvel, use coordenadas estabilizadas).
    Retorna {id_antigo: id_novo}; ids não unidos mapeiam para si mesmos."""
    cfg = cfg or StitchConfig()
    ids = sorted(traj, key=lambda i: traj[i]["frames"][0])
    ends = [i for i in ids]
    starts = [i for i in ids]
    INF = 1e9
    cost = np.full((len(ends), len(starts)), INF)
    for a, ia in enumerate(ends):
        ta = traj[ia]
        for b, ib in enumerate(starts):
            if ia == ib:
                continue
            tb = traj[ib]
            gap = tb["frames"][0] - ta["frames"][-1]
            if gap < 1 or gap > cfg.max_gap:
                continue
            ba, bb = ta["boxes"][-1], tb["boxes"][0]
            sa, sb = np.sqrt((ba[2] - ba[0]) * (ba[3] - ba[1])), np.sqrt((bb[2] - bb[0]) * (bb[3] - bb[1]))
            ratio = sb / max(sa, 1e-6)
            if not cfg.size_ratio[0] <= ratio <= cfg.size_ratio[1]:
                continue
            pred = _extrapolate(ta, gap, cfg.vel_window)
            cb = (bb[:2] + bb[2:]) / 2
            dn = np.hypot(*(pred - cb)) / max(sa, 1e-6)
            if dn > cfg.max_dist_norm:
                continue
            c = dn
            if cfg.use_appearance and appearance and ia in appearance and ib in appearance:
                h1, h2 = appearance[ia][1], appearance[ib][0]
                bc = float(np.sqrt(max(0.0, 1 - np.sum(np.sqrt(h1 * h2)))))
                if bc > cfg.max_appearance_dist:
                    continue
                c += bc
            cost[a, b] = c
    mapping = {i: i for i in ids}
    if (cost < INF).any():
        r, c = linear_sum_assignment(cost)
        parent = {i: i for i in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for a, b in zip(r, c):
            if cost[a, b] < INF:
                ra, rb = find(ends[a]), find(starts[b])
                if ra != rb:
                    parent[rb] = ra
        mapping = {i: find(i) for i in ids}
    return mapping


def apply_mapping(per_frame: dict[int, np.ndarray], mapping: dict[int, int]) -> dict[int, np.ndarray]:
    out = {}
    for f, rows in per_frame.items():
        r = np.asarray(rows, float).reshape(-1, 6).copy()
        r[:, 4] = [mapping.get(int(i), int(i)) for i in r[:, 4]]
        out[f] = r
    return out
