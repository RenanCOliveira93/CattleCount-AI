"""Métricas de tracking (HOTA, DetA, AssA, IDF1, MOTA, ID switches) sobre o TrackEval oficial (J. Luiten).
Entrada: linhas MOT `[frame, id, x, y, w, h, ...]` (frame a partir de qualquer inteiro; caixas xywh em pixels).
IDF1/MOTA/IDsw conferidos contra `motmetrics` em tests/test_mot_metrics.py."""
from __future__ import annotations

import numpy as np
from trackeval import metrics as TM

from .geometry import iou_matrix, xyxy_to_xywh


def _xywh_to_xyxy(a):
    a = np.asarray(a, float).reshape(-1, 4)
    return np.c_[a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]]


def evaluate_mot(gt: np.ndarray, pred: np.ndarray, iou_thr: float = 0.5) -> dict:
    gt = np.asarray(gt, float).reshape(-1, 6)[:, :6]
    pred = np.asarray(pred, float).reshape(-1, 6)[:, :6]
    frames = sorted(set(gt[:, 0].astype(int)) | set(pred[:, 0].astype(int)))
    gids = {g: i for i, g in enumerate(sorted(set(gt[:, 1].astype(int))))}
    pids = {p: i for i, p in enumerate(sorted(set(pred[:, 1].astype(int))))}
    d = {"num_timesteps": len(frames), "gt_ids": [], "tracker_ids": [], "similarity_scores": [], "num_gt_ids": len(gids), "num_tracker_ids": len(pids),
         "num_gt_dets": len(gt), "num_tracker_dets": len(pred)}
    for f in frames:
        g, p = gt[gt[:, 0] == f], pred[pred[:, 0] == f]
        d["gt_ids"].append(np.array([gids[int(i)] for i in g[:, 1]], int))
        d["tracker_ids"].append(np.array([pids[int(i)] for i in p[:, 1]], int))
        d["similarity_scores"].append(iou_matrix(_xywh_to_xyxy(g[:, 2:6]), _xywh_to_xyxy(p[:, 2:6])) if len(g) and len(p) else np.zeros((len(g), len(p))))
    # CLEAR/Identity usam um limiar único de similaridade
    hota = TM.HOTA().eval_sequence(d)
    clear = TM.CLEAR({"THRESHOLD": iou_thr}).eval_sequence(d)
    ident = TM.Identity({"THRESHOLD": iou_thr}).eval_sequence(d)
    return {"HOTA": float(np.mean(hota["HOTA"])), "DetA": float(np.mean(hota["DetA"])), "AssA": float(np.mean(hota["AssA"])),
            "IDF1": float(ident["IDF1"]), "IDP": float(ident["IDP"]), "IDR": float(ident["IDR"]),
            "MOTA": float(clear["MOTA"]), "IDSW": int(clear["IDSW"]), "FP": int(clear["CLR_FP"]), "FN": int(clear["CLR_FN"]),
            "MT": int(clear["MT"]), "PT": int(clear["PT"]), "ML": int(clear["ML"]), "Frag": int(clear["Frag"]),
            "n_gt_ids": len(gids), "n_pred_ids": len(pids), "n_gt_dets": len(gt), "n_pred_dets": len(pred)}


def tracks_to_rows(per_frame: dict[int, np.ndarray]) -> np.ndarray:
    """{frame: (k,6)[x1,y1,x2,y2,id,score]} → linhas MOT [frame,id,x,y,w,h]."""
    rows = []
    for f, t in sorted(per_frame.items()):
        for r in np.asarray(t).reshape(-1, 6):
            x, y, w, h = xyxy_to_xywh(r[:4])[0]
            rows.append([f, r[4], x, y, w, h])
    return np.array(rows, float).reshape(-1, 6)
