import json
import numpy as np
import pytest

from cattlecount import det_metrics as M


def _coco_ap(gts, preds, n_classes, shapes):
    """AP50 e AP50-95 por classe via pycocotools (referência independente)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images, anns, dets = [], [], []
    aid = 1
    for i, ((h, w), g, p) in enumerate(zip(shapes, gts, preds)):
        images.append({"id": i, "height": h, "width": w})
        for c, x1, y1, x2, y2 in g:
            anns.append({"id": aid, "image_id": i, "category_id": int(c), "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0}); aid += 1
        for c, s, x1, y1, x2, y2 in p:
            dets.append({"image_id": i, "category_id": int(c), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    gt = COCO(); gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": c} for c in range(n_classes)]}; gt.createIndex()
    dt = gt.loadRes(dets) if dets else COCO()
    out = {}
    for c in range(n_classes):
        ev = COCOeval(gt, dt, "bbox"); ev.params.catIds = [c]; ev.params.maxDets = [1, 10, 100]
        ev.evaluate(); ev.accumulate(); ev.summarize()
        out[c] = (float(ev.stats[1]), float(ev.stats[0]))  # AP50, AP50:95
    return out


def _random_dataset(seed, n_img=40, n_classes=2):
    rng = np.random.default_rng(seed)
    gts, preds, shapes = [], [], []
    for _ in range(n_img):
        h, w = 480, 640
        k = rng.integers(0, 8)
        g = []
        for _ in range(k):
            x1, y1 = rng.uniform(0, w - 80), rng.uniform(0, h - 80)
            g.append([rng.integers(0, n_classes), x1, y1, x1 + rng.uniform(15, 120), y1 + rng.uniform(15, 120)])
        g = np.array(g).reshape(-1, 5)
        p = []
        for row in g:
            if rng.random() < 0.8:  # detecta com ruído de localização e às vezes erra a classe
                jit = rng.normal(0, 6, 4)
                cls = row[0] if rng.random() < 0.9 else 1 - row[0]
                p.append([cls, rng.uniform(0.3, 1.0), *(row[1:] + jit)])
        for _ in range(rng.integers(0, 4)):  # falsos positivos
            x1, y1 = rng.uniform(0, w - 80), rng.uniform(0, h - 80)
            p.append([rng.integers(0, n_classes), rng.uniform(0.05, 0.7), x1, y1, x1 + rng.uniform(15, 100), y1 + rng.uniform(15, 100)])
        gts.append(g); preds.append(np.array(p).reshape(-1, 6)); shapes.append((h, w))
    return gts, preds, shapes


def test_iou_basico():
    a = np.array([[0, 0, 10, 10]]); b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert M.iou_matrix(a, b).tolist() == [[1.0, pytest.approx(1 / 3), 0.0]]
    assert M.iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_ap_perfeito_e_vazio():
    g = [np.array([[0, 10, 10, 50, 50]])]
    assert M.average_precision(g, [np.array([[0, 0.9, 10, 10, 50, 50]])], 0) == 1.0
    assert M.average_precision(g, [np.zeros((0, 6))], 0) == 0.0
    assert np.isnan(M.average_precision([np.zeros((0, 5))], [np.zeros((0, 6))], 0))


def test_ap_calculado_a_mao():
    # 2 GT, 3 predições: (0.9 TP), (0.8 FP), (0.7 TP) → rec [.5,.5,1], prec [1,.5,.667]
    g = [np.array([[0, 0, 0, 10, 10], [0, 50, 50, 60, 60]])]
    p = [np.array([[0, 0.9, 0, 0, 10, 10], [0, 0.8, 100, 100, 110, 110], [0, 0.7, 50, 50, 60, 60]])]
    rec, prec, _ = M.pr_curve(g, p, 0)
    assert rec.tolist() == [0.5, 0.5, 1.0] and prec.tolist() == pytest.approx([1.0, 0.5, 2 / 3])
    # precisão interpolada: 1.0 até rec .5; 0.667 de .5+ a 1  → AP101 = (51*1 + 50*0.6667)/101
    assert M.average_precision(g, p, 0) == pytest.approx((51 * 1.0 + 50 * (2 / 3)) / 101, abs=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_bate_com_pycocotools(seed):
    gts, preds, shapes = _random_dataset(seed)
    ref = _coco_ap(gts, preds, 2, shapes)
    for c in range(2):
        mine50 = M.average_precision(gts, preds, c, 0.5)
        mine = np.mean([M.average_precision(gts, preds, c, t) for t in M.IOU_THRS])
        assert mine50 == pytest.approx(ref[c][0], abs=2e-3), (seed, c)
        assert mine == pytest.approx(ref[c][1], abs=2e-3), (seed, c)


def test_prf_e_confusao():
    g = [np.array([[0, 0, 0, 10, 10], [1, 50, 50, 60, 60], [0, 100, 100, 110, 110]])]
    p = [np.array([[0, 0.9, 0, 0, 10, 10],        # TP classe 0
                   [0, 0.8, 50, 50, 60, 60],      # classe errada (GT é 1)
                   [1, 0.7, 300, 300, 310, 310]])]  # FP; GT 2 fica sem detecção (FN)
    prf = M.prf_at_conf(g, p, 2, conf=0.25)
    assert prf[0]["tp"] == 1 and prf[0]["fp"] == 1 and prf[0]["fn"] == 1
    assert prf[1]["tp"] == 0 and prf[1]["fp"] == 1 and prf[1]["fn"] == 1
    cm = M.confusion_matrix(g, p, 2, conf=0.25)
    assert cm[0, 0] == 1 and cm[0, 1] == 1 and cm[1, 2] == 1 and cm[2, 0] == 1  # linhas=pred, colunas=real
    assert cm.sum() == 4
    assert M.prf_at_conf(g, p, 2, conf=0.85)["micro"]["tp"] == 1  # só a predição de 0,9 passa


def test_evaluate_json_serializavel_e_fn():
    gts, preds, shapes = _random_dataset(7)
    r = M.evaluate(gts, preds, ["weed", "crop"], conf=0.25)
    json.dumps(r)
    assert 0 <= r["map50"] <= 1 and r["map50_95"] <= r["map50"]
    fn = M.false_negative_analysis(gts, preds, shapes, conf=0.25)
    assert sum(fn["motivos"].values()) == fn["fn_total"]
    assert sum(v["n"] for v in fn["por_tamanho"].values()) == fn["n_gt"]


def test_bootstrap_ic_contem_estimativa_e_modelo_identico_tem_diferenca_zero():
    gts, preds, _ = _random_dataset(11, n_img=30)
    b = M.bootstrap_map(gts, {"a": preds, "b": preds}, n_boot=40, cls_ids=(0, 1))
    pt = np.nanmean([M.average_precision(gts, preds, c, 0.5) for c in (0, 1)])
    lo, hi = b["a"]["map50"]["ic95"]
    assert lo <= pt <= hi and lo < hi
    d = b["b"]["diferenca_vs_a"]["map50"]
    assert d["ic95"] == [0.0, 0.0] and d["media"] == 0.0


def test_bootstrap_com_indice_identidade_reproduz_ap():
    gts, preds, _ = _random_dataset(5, n_img=25)
    parts = M._per_image(gts, preds, 0, 0.5)
    assert M._ap_from_parts(parts, np.arange(25)) == pytest.approx(M.average_precision(gts, preds, 0, 0.5), abs=1e-12)
