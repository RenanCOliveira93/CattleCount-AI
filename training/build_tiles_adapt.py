"""Experimento de ADAPTAÇÃO a fazenda nova (declarado antes de rodar): tiles de 2 voos de Jalogny (031 e 058) entram no treino junto com os tiles originais.
Teste da adaptação = os 3 voos restantes de Jalogny (032, 056, 059). Os tiles originais são ligados por link simbólico."""
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cattlecount.tiling import tile_grid  # noqa: E402
from training.zenodo_plan import EX, load  # noqa: E402

ADAPT_FLIGHTS = ["DJI_202306231103_031", "DJI_202309261339_058"]
OUT, SRC = ROOT / "data/yolo_tiles_adapt", ROOT / "data/yolo_tiles"
TILE, OV, MIN_VIS, NEG_KEEP = 1024, 0.2, 0.5, 0.5   # aqui mais negativos: o ponto da adaptação é aprender os falsos positivos da fazenda nova


def link_tree(sub):
    (OUT / sub / "images").mkdir(parents=True, exist_ok=True); (OUT / sub / "labels").mkdir(parents=True, exist_ok=True)
    for kind in ("images", "labels"):
        for f in (SRC / sub / kind).iterdir():
            d = OUT / sub / kind / f.name
            if not d.exists():
                os.symlink(f, d)


def main(seed=0):
    link_tree("train"); link_tree("val")
    _, _, _, lab, _ = load()
    rng = random.Random(seed)
    items = [m for m in json.load(open(SRC / "manifest.json"))["items"] if m["split"] == "test" and m["flight"] in ADAPT_FLIGHTS and m["status"] == "ok"]
    n = [0, 0, 0]
    for o in items:
        img = cv2.imread(str(ROOT / o["path"])); h, w = img.shape[:2]; L = lab[o["stem"]]
        b = np.stack([(L[:, 1] - L[:, 3] / 2) * w, (L[:, 2] - L[:, 4] / 2) * h, (L[:, 1] + L[:, 3] / 2) * w, (L[:, 2] + L[:, 4] / 2) * h], 1) if len(L) else np.zeros((0, 4))
        area = np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-9)
        for x0, y0, x1, y1 in tile_grid(h, w, TILE, OV):
            c = np.stack([np.clip(b[:, 0], x0, x1), np.clip(b[:, 1], y0, y1), np.clip(b[:, 2], x0, x1), np.clip(b[:, 3], y0, y1)], 1)
            vis = (c[:, 2] - c[:, 0]).clip(0) * (c[:, 3] - c[:, 1]).clip(0) / area
            keep = vis >= MIN_VIS
            if not keep.any() and rng.random() > NEG_KEEP:
                continue
            name = f"{o['stem']}_{x0}_{y0}"
            cv2.imwrite(str(OUT / "train/images" / f"{name}.jpg"), img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            tw, th = x1 - x0, y1 - y0
            lines = [f"0 {((r[0] + r[2]) / 2 - x0) / tw:.6f} {((r[1] + r[3]) / 2 - y0) / th:.6f} {(r[2] - r[0]) / tw:.6f} {(r[3] - r[1]) / th:.6f}" for r in c[keep]]
            (OUT / "train/labels" / f"{name}.txt").write_text("\n".join(lines))
            n[0] += 1; n[1] += bool(lines); n[2] += len(lines)
    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: train/images\nval: val/images\nnames:\n  0: cow\n")
    (OUT / "adapt_manifest.json").write_text(json.dumps({"adapt_flights": ADAPT_FLIGHTS, "neg_keep": NEG_KEEP, "tiles_adapt(total,com_gado,caixas)": n, "images": [o["stem"] for o in items]}, indent=1))
    print("tiles de adaptação:", n, "de", len(items), "imagens")


if __name__ == "__main__":
    main()
