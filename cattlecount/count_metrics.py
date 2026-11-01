"""Erros de contagem: MAE, RMSE, MAPE, viés (com IC por bootstrap) e agrupamento por condição (altitude, densidade, voo...)."""
from __future__ import annotations

import numpy as np


def count_errors(pred, gt) -> dict:
    p, g = np.asarray(pred, float), np.asarray(gt, float)
    e = p - g
    nz = g > 0
    return {"n": int(len(g)), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()),
            "mape": float(np.mean(np.abs(e[nz]) / g[nz])) if nz.any() else None, "gt_total": float(g.sum()), "pred_total": float(p.sum()),
            "total_error_pct": float((p.sum() - g.sum()) / g.sum()) if g.sum() > 0 else None}


def bootstrap_ci(pred, gt, n_boot: int = 1000, seed: int = 0, groups=None) -> dict:
    """IC 95% percentil de MAE/RMSE/viés por reamostragem de IMAGENS (ou de GRUPOS inteiros, ex.: voos, se `groups` for dado — recomendado,
    pois imagens do mesmo voo são correlacionadas)."""
    p, g = np.asarray(pred, float), np.asarray(gt, float)
    rng = np.random.default_rng(seed)
    if groups is None:
        units = [np.array([i]) for i in range(len(p))]
    else:
        gr = np.asarray(groups)
        units = [np.where(gr == k)[0] for k in np.unique(gr)]
    res = {"mae": [], "rmse": [], "bias": []}
    for _ in range(n_boot):
        idx = np.concatenate([units[i] for i in rng.integers(0, len(units), len(units))])
        e = p[idx] - g[idx]
        res["mae"].append(np.abs(e).mean()); res["rmse"].append(np.sqrt((e ** 2).mean())); res["bias"].append(e.mean())
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in res.items()} | {"unidade_bootstrap": "grupo" if groups is not None else "imagem", "n_unidades": len(units)}


def by_group(pred, gt, groups) -> dict:
    gr = np.asarray(groups)
    return {str(k): count_errors(np.asarray(pred)[gr == k], np.asarray(gt)[gr == k]) for k in sorted(set(gr.tolist()))}


def density_bins(gt, edges=(0, 1, 6, 16, 10**9)) -> list[str]:
    """Rótulos de densidade por imagem: 0 | 1–5 | 6–15 | ≥16 animais."""
    names = ["0", "1-5", "6-15", ">=16"]
    return [names[int(np.searchsorted(edges, g, side="right")) - 1] for g in gt]
