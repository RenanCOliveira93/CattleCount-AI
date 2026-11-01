"""Avalia predições (saída de predict_split.py): AP/mAP, P/R/F1, erros de contagem por imagem com IC (bootstrap por voo) e estratos.
O limiar operacional é escolhido NO VAL (melhor F1) e aplicado sem alteração no TESTE. Uso:
  eval_predictions.py <nome> <val_preds.json> <test_preds.json> [--conf X] [--test-flights f1,f2,..]
  --conf fixa o limiar em vez de escolher no val; --test-flights restringe o TESTE a esses voos (ex.: excluir os voos usados na adaptação)"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cattlecount.count_metrics import bootstrap_ci, by_group, count_errors, density_bins  # noqa: E402
from cattlecount.det_metrics import evaluate, prf_at_conf  # noqa: E402
from training.zenodo_plan import load  # noqa: E402

name, val_p, test_p = sys.argv[1:4]
fixed = float(sys.argv[sys.argv.index("--conf") + 1]) if "--conf" in sys.argv else None
_, _, _, lab, _ = load()
flights = json.load(open(ROOT / "data/raw/zenodo/download_plan.json"))["flights"]


def gt_boxes(r):
    h, w = r["hw"]
    L = lab[r["stem"]]
    return np.c_[np.zeros(len(L)), (L[:, 1] - L[:, 3] / 2) * w, (L[:, 2] - L[:, 4] / 2) * h, (L[:, 1] + L[:, 3] / 2) * w, (L[:, 2] + L[:, 4] / 2) * h] if len(L) else np.zeros((0, 5))


def load_split(p):
    d = json.load(open(p))
    rows = d["images"]
    gts = [gt_boxes(r) for r in rows]
    preds = [np.c_[np.zeros(len(r["pred"])), np.array(r["pred"], float)[:, 4], np.array(r["pred"], float)[:, :4]] if r["pred"] else np.zeros((0, 6)) for r in rows]
    return d, rows, gts, preds


def counts(rows, preds, conf):
    return np.array([int((p[:, 1] >= conf).sum()) for p in preds]), np.array([r["n_gt"] for r in rows])


dv, rv, gv, pv = load_split(val_p)
dt, rt, gt, pt = load_split(test_p)
if "--test-flights" in sys.argv:
    keep = set(sys.argv[sys.argv.index("--test-flights") + 1].split(","))
    idx = [i for i, r in enumerate(rt) if r["flight"] in keep]
    rt, gt, pt = [rt[i] for i in idx], [gt[i] for i in idx], [pt[i] for i in idx]
grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
f1v = {float(t): prf_at_conf(gv, pv, 1, float(t))["micro"]["f1"] for t in grid}
conf = fixed if fixed is not None else max(f1v, key=lambda t: (np.nan_to_num(f1v[t]), -t))
out = {"name": name, "test_flights": sorted({r["flight"] for r in rt}), "weights": dt["weights"], "coco_zero_shot": dt["coco_cow"], "conf_operacional": conf, "regra_conf": "fixado manualmente" if fixed is not None else "melhor F1 micro no VAL",
       "f1_val_por_conf": f1v}
for split, rows, gts, preds in (("val", rv, gv, pv), ("test", rt, gt, pt)):
    ev = evaluate(gts, preds, ["cow"], conf)
    p, g = counts(rows, preds, conf)
    alt = [flights[r["flight"]]["alt"] for r in rows]
    grp = [r["flight"] for r in rows]
    out[split] = {"n_images": len(rows), "n_gt_boxes": int(g.sum()), "map50": ev["map50"], "map50_95": ev["map50_95"], "ap75": ev["per_class_ap"]["cow"].get("ap75"), "prf": ev["prf"]["cow"],
                  "contagem": count_errors(p, g), "contagem_ic95_bootstrap_por_voo": bootstrap_ci(p, g, 1000, 0, grp),
                  "por_voo": by_group(p, g, grp), "por_altitude_m": by_group(p, g, alt), "por_densidade": by_group(p, g, density_bins(g)),
                  "por_fazenda": by_group(p, g, [r["farm"] for r in rows]), "latencia_s_por_imagem": {"mediana": float(np.median([r["sec"] for r in rows])), "n": len(rows)},
                  "imagens": [{"stem": r["stem"], "flight": r["flight"], "gt": int(a), "pred": int(b)} for r, a, b in zip(rows, g, p)]}
Path(ROOT / "experiments").mkdir(exist_ok=True)
Path(ROOT / f"experiments/{name}.json").write_text(json.dumps(out, indent=1, default=float))
for s in ("val", "test"):
    o = out[s]
    print(s, "mAP50=%.3f mAP50-95=%.3f P=%.3f R=%.3f | contagem MAE=%.2f RMSE=%.2f viés=%.2f (n=%d, GT=%d, pred=%d)" % (o["map50"], o["map50_95"], o["prf"]["precision"], o["prf"]["recall"],
          o["contagem"]["mae"], o["contagem"]["rmse"], o["contagem"]["bias"], o["n_images"], o["contagem"]["gt_total"], o["contagem"]["pred_total"]), "conf", conf)
