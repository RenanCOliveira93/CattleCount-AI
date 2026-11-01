"""Inferência fatiada (1024 px nativos, 20% de sobreposição) nas imagens INTEIRAS de um split; salva predições brutas (conf>=0,05).
Uso: predict_split.py <pesos.pt> <split: val|test> <saida.json> [--coco-cow]   (--coco-cow: baseline zero-shot, classe 19 'cow' do COCO)"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cattlecount.tiling import sliced_predict  # noqa: E402

w, split, out = sys.argv[1], sys.argv[2], sys.argv[3]
coco = "--coco-cow" in sys.argv
model = YOLO(w)
man = json.load(open(ROOT / "data/yolo_tiles/manifest.json"))["items"]
items = [m for m in man if m["split"] == split and m["status"] == "ok"]


def detect_tiles(tiles):
    res = model.predict([cv2.cvtColor(t, cv2.COLOR_RGB2BGR) for t in tiles], imgsz=1024, conf=0.05, iou=0.6, device="mps", verbose=False,
                        classes=[19] if coco else None, max_det=300)
    return [np.c_[r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()] for r in res]


rows, t0 = [], time.time()
for k, m in enumerate(items):
    img = cv2.cvtColor(cv2.imread(str(ROOT / m["path"])), cv2.COLOR_BGR2RGB)
    t1 = time.time()
    p = sliced_predict(detect_tiles, img, 1024, 0.2)
    rows.append({"stem": m["stem"], "flight": m["flight"], "farm": m["farm"], "hw": list(img.shape[:2]), "n_gt": m["boxes"], "pred": np.round(p, 2).tolist(), "sec": round(time.time() - t1, 2)})
    print(k + 1, len(items), m["stem"], len(p), round(time.time() - t1, 1), flush=True)
Path(out).write_text(json.dumps({"weights": w, "split": split, "coco_cow": coco, "images": rows}))
