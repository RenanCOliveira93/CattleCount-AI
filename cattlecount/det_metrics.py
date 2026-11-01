"""Métricas de detecção próprias (copiadas e adaptadas do projeto 04, onde foram conferidas contra pycocotools; reconferidas em tests/test_det_metrics.py).

Convenções
- GT: array (n, 5) [cls, x1, y1, x2, y2] em pixels.  Predição: array (m, 6) [cls, score, x1, y1, x2, y2] em pixels.
- AP: interpolação de 101 pontos estilo COCO; AP50-95 = média em IoU 0,50:0,05:0,95; mAP = média por classe com ≥ 1 GT.
- Casamento guloso por score decrescente, um GT por predição, no máximo 100 detecções por imagem (como o COCOeval).
- Verificado contra `pycocotools` em tests/test_metrics.py.
"""
from __future__ import annotations

import numpy as np

IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)
MAX_DETS = 100


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre caixas xyxy: (n,4) × (m,4) → (n,m)."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + ab[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _match_one_class(gt_boxes: np.ndarray, pred_boxes: np.ndarray, pred_scores: np.ndarray, thr: float):
    """Retorna (tp bool por predição na ordem de entrada, índice do GT casado ou -1)."""
    order = np.argsort(-pred_scores, kind="stable")[:MAX_DETS]
    tp = np.zeros(len(pred_boxes), bool)
    gt_idx = -np.ones(len(pred_boxes), int)
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return tp, gt_idx
    ious = iou_matrix(pred_boxes, gt_boxes)
    taken = np.zeros(len(gt_boxes), bool)
    for i in order:
        cand = np.where(~taken, ious[i], -1)
        j = int(np.argmax(cand))
        if cand[j] >= thr - 1e-12:
            taken[j] = True
            tp[i] = True
            gt_idx[i] = j
    return tp, gt_idx


def _ap_from_pr(rec: np.ndarray, prec: np.ndarray) -> float:
    """AP por interpolação de 101 pontos (COCO)."""
    if len(rec) == 0:
        return 0.0
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    xs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, xs, side="left")
    q = np.array([prec[i] if i < len(prec) else 0.0 for i in idx])
    return float(q.mean())


def _collect(gts, preds, cls: int, thr: float):
    """Concatena (score, tp) de todas as imagens para uma classe e um IoU."""
    scores, tps, n_gt = [], [], 0
    for g, p in zip(gts, preds):
        g = np.asarray(g, float).reshape(-1, 5)
        p = np.asarray(p, float).reshape(-1, 6)
        gc, pc = g[g[:, 0] == cls], p[p[:, 0] == cls]
        n_gt += len(gc)
        if len(pc) == 0:
            continue
        tp, _ = _match_one_class(gc[:, 1:5], pc[:, 2:6], pc[:, 1], thr)
        order = np.argsort(-pc[:, 1], kind="stable")[:MAX_DETS]
        scores.append(pc[order, 1])
        tps.append(tp[order])
    if not scores:
        return np.array([]), np.array([], bool), n_gt
    return np.concatenate(scores), np.concatenate(tps), n_gt


def pr_curve(gts, preds, cls: int, thr: float = 0.5):
    s, tp, n_gt = _collect(gts, preds, cls, thr)
    if n_gt == 0:
        return np.array([]), np.array([]), np.array([])
    o = np.argsort(-s, kind="stable")
    s, tp = s[o], tp[o]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(~tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    return rec, prec, s


def average_precision(gts, preds, cls: int, thr: float = 0.5) -> float:
    _, _, n_gt = _collect(gts, preds, cls, thr)
    if n_gt == 0:
        return float("nan")  # classe sem GT: AP indefinido
    rec, prec, _ = pr_curve(gts, preds, cls, thr)
    return _ap_from_pr(rec, prec) if len(rec) else 0.0  # há GT mas nenhuma predição: AP = 0


def prf_at_conf(gts, preds, n_classes: int, conf: float, iou: float = 0.5) -> dict:
    """Precisão, revocação e F1 por classe e micro, com limiar de confiança fixo e IoU = 0,5."""
    out = {}
    TP = FP = FN = 0
    for c in range(n_classes):
        tp = fp = fn = 0
        for g, p in zip(gts, preds):
            g = np.asarray(g, float).reshape(-1, 5)
            p = np.asarray(p, float).reshape(-1, 6)
            gc = g[g[:, 0] == c]
            pc = p[(p[:, 0] == c) & (p[:, 1] >= conf)]
            t, _ = _match_one_class(gc[:, 1:5], pc[:, 2:6], pc[:, 1], iou)
            tp += int(t.sum())
            fp += int((~t).sum())
            fn += len(gc) - int(t.sum())
        pr = tp / (tp + fp) if tp + fp else float("nan")
        rc = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * pr * rc / (pr + rc) if tp and pr + rc else (0.0 if (tp + fp) and (tp + fn) else float("nan"))
        out[c] = {"precision": pr, "recall": rc, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        TP, FP, FN = TP + tp, FP + fp, FN + fn
    pr = TP / (TP + FP) if TP + FP else float("nan")
    rc = TP / (TP + FN) if TP + FN else float("nan")
    out["micro"] = {"precision": pr, "recall": rc, "f1": (2 * pr * rc / (pr + rc)) if TP else 0.0, "tp": TP, "fp": FP, "fn": FN}
    return out


def confusion_matrix(gts, preds, n_classes: int, conf: float = 0.25, iou: float = 0.5) -> np.ndarray:
    """Matriz (n+1)×(n+1): linhas = predito, colunas = real; o último índice é 'fundo'.
    FP (predição sem GT) vai na coluna fundo; FN (GT sem predição) vai na linha fundo."""
    cm = np.zeros((n_classes + 1, n_classes + 1), int)
    for g, p in zip(gts, preds):
        g = np.asarray(g, float).reshape(-1, 5)
        p = np.asarray(p, float).reshape(-1, 6)
        p = p[p[:, 1] >= conf]
        ious = iou_matrix(g[:, 1:5], p[:, 2:6])
        pairs = [(ious[i, j], i, j) for i in range(len(g)) for j in range(len(p)) if ious[i, j] >= iou]
        pairs.sort(reverse=True)
        gi, pj = set(), set()
        for _, i, j in pairs:
            if i in gi or j in pj:
                continue
            gi.add(i), pj.add(j)
            cm[int(p[j, 0]), int(g[i, 0])] += 1
        for i in range(len(g)):
            if i not in gi:
                cm[n_classes, int(g[i, 0])] += 1
        for j in range(len(p)):
            if j not in pj:
                cm[int(p[j, 0]), n_classes] += 1
    return cm


def evaluate(gts, preds, class_names: list[str], conf: float = 0.25, img_shapes=None) -> dict:
    """Pacote completo de métricas (serializável em JSON)."""
    n = len(class_names)
    per_class = {}
    for c, name in enumerate(class_names):
        n_gt = int(sum((np.asarray(g, float).reshape(-1, 5)[:, 0] == c).sum() for g in gts))
        if n_gt == 0:
            per_class[name] = {"n_gt": 0, "ap50": None, "ap50_95": None}
            continue
        aps = [average_precision(gts, preds, c, t) for t in IOU_THRS]
        per_class[name] = {"n_gt": n_gt, "ap50": aps[0], "ap50_95": float(np.mean(aps)), "ap75": aps[5]}
    valid = [v for v in per_class.values() if v["n_gt"]]
    prf = prf_at_conf(gts, preds, n, conf)
    # limiar de melhor F1 (micro), escolhido num grade — reportado, mas as métricas "oficiais" usam o conf fixo
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    f1s = [prf_at_conf(gts, preds, n, float(t))["micro"]["f1"] for t in grid]
    best = int(np.nanargmax(f1s))
    curves = {}
    for c, name in enumerate(class_names):
        rec, prec, sc = pr_curve(gts, preds, c, 0.5)
        if len(rec):
            k = np.unique(np.linspace(0, len(rec) - 1, min(len(rec), 200)).astype(int))
            curves[name] = {"recall": rec[k].round(4).tolist(), "precision": prec[k].round(4).tolist(), "score": sc[k].round(4).tolist()}
    cm = confusion_matrix(gts, preds, n, conf)
    return {
        "conf_threshold": conf, "iou_match": 0.5, "n_images": len(gts), "class_names": class_names,
        "map50": float(np.mean([v["ap50"] for v in valid])) if valid else None,
        "map50_95": float(np.mean([v["ap50_95"] for v in valid])) if valid else None,
        "per_class_ap": per_class,
        "prf": {(class_names[k] if isinstance(k, int) else k): v for k, v in prf.items()},
        "best_f1": {"conf": float(grid[best]), "f1": float(f1s[best])},
        "pr_curves": curves, "confusion_matrix": cm.tolist(),
    }


def false_negative_analysis(gts, preds, img_shapes, conf: float = 0.25, iou: float = 0.5) -> dict:
    """Por que o modelo perde objetos? Classifica cada FN em: classe errada, localização ruim, ausente.
    Também mede FN por tamanho relativo do objeto e por aglomeração (vizinhos sobrepostos)."""
    rows = []
    for (g, p, (h, w)) in zip(gts, preds, img_shapes):
        g = np.asarray(g, float).reshape(-1, 5)
        p = np.asarray(p, float).reshape(-1, 6)
        p = p[p[:, 1] >= conf]
        if len(g) == 0:
            continue
        ious_all = iou_matrix(g[:, 1:5], p[:, 2:6])
        gg = iou_matrix(g[:, 1:5], g[:, 1:5])
        taken = set()
        # casamento classe-consciente (mesmo critério de prf_at_conf)
        matched = np.zeros(len(g), bool)
        for c in np.unique(g[:, 0]):
            gi = np.where(g[:, 0] == c)[0]
            pj = np.where(p[:, 0] == c)[0]
            t, idx = _match_one_class(g[gi, 1:5], p[pj, 2:6], p[pj, 1], iou)
            for k, ok in enumerate(t):
                if ok:
                    matched[gi[idx[k]]] = True
        for i in range(len(g)):
            area_rel = (g[i, 3] - g[i, 1]) * (g[i, 4] - g[i, 2]) / (h * w)
            neighbors = int(((gg[i] > 0.0) & (np.arange(len(g)) != i)).sum())
            reason = "detectado"
            if not matched[i]:
                best = ious_all[i].max() if ious_all.shape[1] else 0.0
                j = int(ious_all[i].argmax()) if ious_all.shape[1] else -1
                if best >= iou and p[j, 0] != g[i, 0]:
                    reason = "classe_errada"
                elif best >= 0.1:
                    reason = "localizacao_ruim"
                else:
                    reason = "ausente"
            rows.append((reason, area_rel, neighbors, int(g[i, 0])))
    if not rows:
        return {}
    reasons = np.array([r[0] for r in rows]); area = np.array([r[1] for r in rows]); nb = np.array([r[2] for r in rows])
    # baldes na escala dos dados de plantas daninhas (mediana ≈ 0,1% da imagem; em 1280×720: 0,05% ≈ 21×21 px, 0,25% ≈ 48×48 px)
    bins = [("muito pequeno (<0,05% da imagem)", area < 0.0005), ("pequeno (0,05–0,1%)", (area >= 0.0005) & (area < 0.001)),
            ("médio (0,1–0,25%)", (area >= 0.001) & (area < 0.0025)), ("grande (≥0,25%)", area >= 0.0025)]
    out = {"n_gt": len(rows), "fn_total": int((reasons != "detectado").sum()),
           "motivos": {k: int((reasons == k).sum()) for k in ("classe_errada", "localizacao_ruim", "ausente")},
           "por_tamanho": {n_: {"n": int(m.sum()), "fn": int(((reasons != "detectado") & m).sum()),
                                "taxa_fn": float(((reasons != "detectado") & m).sum() / m.sum()) if m.sum() else None} for n_, m in bins},
           "por_aglomeracao": {}}
    for name, m in [("isolado (0 vizinhos sobrepostos)", nb == 0), ("com vizinhos sobrepostos", nb > 0)]:
        out["por_aglomeracao"][name] = {"n": int(m.sum()), "fn": int(((reasons != "detectado") & m).sum()),
                                        "taxa_fn": float(((reasons != "detectado") & m).sum() / m.sum()) if m.sum() else None}
    return out


def _per_image(gts, preds, cls: int, thr: float):
    """Casamento por imagem (independe da reamostragem): lista de (scores, tp, n_gt)."""
    out = []
    for g, p in zip(gts, preds):
        g = np.asarray(g, float).reshape(-1, 5)
        p = np.asarray(p, float).reshape(-1, 6)
        gc, pc = g[g[:, 0] == cls], p[p[:, 0] == cls]
        if len(pc) == 0:
            out.append((np.array([]), np.array([], bool), len(gc)))
            continue
        tp, _ = _match_one_class(gc[:, 1:5], pc[:, 2:6], pc[:, 1], thr)
        order = np.argsort(-pc[:, 1], kind="stable")[:MAX_DETS]
        out.append((pc[order, 1], tp[order], len(gc)))
    return out


def _ap_from_parts(parts, idx) -> float:
    n_gt = sum(parts[i][2] for i in idx)
    if n_gt == 0:
        return float("nan")
    s = np.concatenate([parts[i][0] for i in idx])
    if len(s) == 0:
        return 0.0
    tp = np.concatenate([parts[i][1] for i in idx])
    o = np.argsort(-s, kind="stable")
    tp = tp[o]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    return _ap_from_pr(ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-12))


def bootstrap_map(gts, preds_by_model: dict, n_boot: int = 200, seed: int = 0, cls_ids=(0,)) -> dict:
    """Bootstrap por IMAGEM (reamostra imagens com reposição, o mesmo índice para todos os modelos → comparação pareada).
    Devolve IC 95% percentil de mAP50 e mAP50-95 por modelo e da diferença (modelo − primeiro modelo).
    Limitação: imagens da mesma sessão são correlacionadas, então o IC é OTIMISTA (estreito demais)."""
    rng = np.random.default_rng(seed)
    n = len(gts)
    names = list(preds_by_model)
    parts = {m: {(c, t): _per_image(gts, preds_by_model[m], c, t) for c in cls_ids for t in IOU_THRS} for m in names}
    res = {m: {"map50": [], "map50_95": []} for m in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for m in names:
            ap = {t: np.nanmean([_ap_from_parts(parts[m][(c, t)], idx) for c in cls_ids]) for t in IOU_THRS}
            res[m]["map50"].append(float(ap[0.5]))
            res[m]["map50_95"].append(float(np.mean(list(ap.values()))))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    out = {m: {k: {"ic95": ci(v), "media": float(np.mean(v))} for k, v in d.items()} for m, d in res.items()}
    base = names[0]
    for m in names[1:]:
        out[m]["diferenca_vs_" + base] = {k: {"ic95": ci(np.array(res[m][k]) - np.array(res[base][k])),
                                              "media": float(np.mean(np.array(res[m][k]) - np.array(res[base][k])))} for k in ("map50", "map50_95")}
    return out
