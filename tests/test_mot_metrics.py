import numpy as np
import pytest

from cattlecount.geometry import iou_matrix
from cattlecount.mot_metrics import evaluate_mot


def _xyxy(a):
    return np.c_[a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]]


def _gt(n_frames=30):
    rows = []
    for t in range(n_frames):
        rows += [[t, 1, 10 + 5 * t, 20, 40, 40], [t, 2, 300 - 4 * t, 120, 40, 40]]
    return np.array(rows, float)


def test_rastreamento_perfeito():
    r = evaluate_mot(_gt(), _gt() * [1, 10, 1, 1, 1, 1])  # ids diferentes mas consistentes
    assert r["IDF1"] == pytest.approx(1) and r["HOTA"] == pytest.approx(1) and r["IDSW"] == 0 and r["MOTA"] == pytest.approx(1)


def test_troca_de_id_derruba_idf1_e_assa_mas_nao_deta():
    gt, p = _gt(), _gt().copy()
    p[(p[:, 1] == 1) & (p[:, 0] >= 15), 1] = 99          # o animal 1 troca de ID no meio
    r = evaluate_mot(gt, p)
    assert r["IDSW"] == 1 and r["IDF1"] < 0.95 and r["AssA"] < 0.95 and r["DetA"] == pytest.approx(1)
    assert r["n_pred_ids"] == 3  # ID fragmentado


def test_confere_com_motmetrics():
    import motmetrics as mm
    rng = np.random.default_rng(3)
    gt = _gt(40)
    p = gt.copy()
    p = p[rng.random(len(p)) > 0.15]                      # 15% de FN
    p[(p[:, 1] == 2) & (p[:, 0] >= 20), 1] = 7            # troca de ID
    p[:, 2:4] += rng.normal(0, 2, (len(p), 2))            # ruído de localização
    ours = evaluate_mot(gt, p)
    acc = mm.MOTAccumulator(auto_id=False)
    for f in sorted(set(gt[:, 0])):
        g, q = gt[gt[:, 0] == f], p[p[:, 0] == f]
        # mm.distances.iou_matrix usa np.asfarray (removido no NumPy 2): calculamos a distância 1−IoU aqui; NaN = não pareável (IoU < 0,5)
        iou = iou_matrix(_xyxy(g[:, 2:6]), _xyxy(q[:, 2:6])) if len(q) else np.zeros((len(g), 0))
        dist = np.where(iou >= 0.5, 1 - iou, np.nan)
        acc.update(g[:, 1].astype(int).tolist(), q[:, 1].astype(int).tolist(), dist, frameid=int(f))
    s = mm.metrics.create().compute(acc, metrics=["idf1", "mota", "num_switches", "num_false_positives", "num_misses"], name="x").iloc[0]
    assert ours["IDF1"] == pytest.approx(s["idf1"], abs=1e-6)
    assert ours["MOTA"] == pytest.approx(s["mota"], abs=1e-6)
    assert ours["IDSW"] == s["num_switches"] and ours["FP"] == s["num_false_positives"] and ours["FN"] == s["num_misses"]
