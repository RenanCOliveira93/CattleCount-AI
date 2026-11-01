"""Monta o dataset YOLO de tiles nativos (1024 px, 20% de sobreposição) a partir das imagens ICAERUS baixadas.
train/val: tiles; test: só a lista de imagens inteiras + rótulos (avaliação por inferência fatiada). Sem filtragem por resultado de modelo."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cattlecount.tiling import tile_grid  # noqa: E402
from training.zenodo_plan import EX, load  # noqa: E402

OUT = ROOT / "data/yolo_tiles"
TILE, OV, MIN_VIS, NEG_KEEP = 1024, 0.2, 0.5, 0.15


def main(seed: int = 0):
    plan = json.load(open(ROOT / "data/raw/zenodo/download_plan.json"))["order"]
    _, _, _, lab, _ = load()
    rng = random.Random(seed)
    manifest, stats = [], {"train": [0, 0, 0], "val": [0, 0, 0]}   # tiles, tiles com gado, caixas
    for o in plan:
        p = EX / o["member"]
        if not p.exists() or p.stat().st_size != o["size"]:
            manifest.append({**o, "status": "ausente"}); continue
        manifest.append({**o, "status": "ok", "path": str(p.relative_to(ROOT))})
        if o["split"] == "test":
            continue
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        L = lab[o["stem"]]
        b = np.stack([(L[:, 1] - L[:, 3] / 2) * w, (L[:, 2] - L[:, 4] / 2) * h, (L[:, 1] + L[:, 3] / 2) * w, (L[:, 2] + L[:, 4] / 2) * h], 1) if len(L) else np.zeros((0, 4))
        area = np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-9)
        for x0, y0, x1, y1 in tile_grid(h, w, TILE, OV):
            c = np.stack([np.clip(b[:, 0], x0, x1), np.clip(b[:, 1], y0, y1), np.clip(b[:, 2], x0, x1), np.clip(b[:, 3], y0, y1)], 1)
            vis = (c[:, 2] - c[:, 0]).clip(0) * (c[:, 3] - c[:, 1]).clip(0) / area
            keep = vis >= MIN_VIS
            if not keep.any() and (o["split"] == "val" or rng.random() > NEG_KEEP):
                if o["split"] == "train":
                    continue
            name = f"{o['stem']}_{x0}_{y0}"
            d = OUT / o["split"]
            (d / "images").mkdir(parents=True, exist_ok=True); (d / "labels").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / "images" / f"{name}.jpg"), img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            tw, th = x1 - x0, y1 - y0
            lines = [f"0 {((r[0] + r[2]) / 2 - x0) / tw:.6f} {((r[1] + r[3]) / 2 - y0) / th:.6f} {(r[2] - r[0]) / tw:.6f} {(r[3] - r[1]) / th:.6f}" for r in c[keep]]
            (d / "labels" / f"{name}.txt").write_text("\n".join(lines))
            s = stats[o["split"]]; s[0] += 1; s[1] += bool(lines); s[2] += len(lines)
    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: train/images\nval: val/images\nnames:\n  0: cow\n")
    (OUT / "manifest.json").write_text(json.dumps({"tile": TILE, "overlap": OV, "min_visible": MIN_VIS, "neg_keep_train": NEG_KEEP, "seed": seed, "items": manifest, "stats(tiles,tiles_com_gado,caixas)": stats}, indent=1))
    miss = [m for m in manifest if m["status"] != "ok"]
    print(stats, "ausentes:", len(miss), "de", len(manifest))


if __name__ == "__main__":
    main()
