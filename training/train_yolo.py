"""Fine-tuning YOLO11 em tiles ICAERUS. Uso: train_yolo.py <variante|pesos.pt> <nome> [epochs] [patience] [data.yaml] [lr0]"""
import json
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
var, name = sys.argv[1], sys.argv[2]
ep = int(sys.argv[3]) if len(sys.argv) > 3 else 40
pat = int(sys.argv[4]) if len(sys.argv) > 4 else 15
data = sys.argv[5] if len(sys.argv) > 5 else str(ROOT / "data/yolo_tiles/data.yaml")
lr0 = float(sys.argv[6]) if len(sys.argv) > 6 else 0.01
m = YOLO(var if var.endswith(".pt") and Path(var).exists() else str(ROOT / f"models/pretrained/{var}.pt"))
m.train(data=data, imgsz=1024, epochs=ep, batch=8, device="mps", workers=4, project=str(ROOT / "runs"), name=name, seed=0,
        patience=pat, lr0=lr0, flipud=0.5, fliplr=0.5, degrees=90, mosaic=1.0, scale=0.3, exist_ok=True, plots=False)
(ROOT / "runs" / name / "train_args.json").write_text(json.dumps({"variant": var, "epochs": ep, "imgsz": 1024, "batch": 8, "seed": 0, "patience": pat, "data": data, "lr0": lr0}))
