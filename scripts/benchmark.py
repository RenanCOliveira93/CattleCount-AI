"""Latência REAL na máquina atual (ONNX Runtime, CPU). Rodar com a máquina ociosa. Saída: experiments/benchmark.json"""
import json
import platform
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.detector import OnnxCattleDetector  # noqa: E402
from cattlecount.tracking import ByteTracker  # noqa: E402

onnx = ROOT / "models/onnx/cattle_yolo11s_1024.onnx"
det = OnnxCattleDetector(onnx, "sliced")
man = [m for m in json.load(open(ROOT / "data/yolo_tiles/manifest.json"))["items"] if m["split"] == "val" and m["status"] == "ok"]
img = cv2.cvtColor(cv2.imread(str(ROOT / max(man, key=lambda m: m["boxes"])["path"])), cv2.COLOR_BGR2RGB)
tile, frame = img[:1024, :1024], img[:720, :1280]


def bench(fn, n, warm=2):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1000)
    return {"n": n, "mediana_ms": round(st.median(t), 1), "p95_ms": round(float(np.percentile(t, 95)), 1)}


out = {"maquina": {"sistema": platform.platform(), "cpu": platform.processor() or platform.machine(), "provider": "CPUExecutionProvider (onnxruntime)"},
       "modelo": onnx.name,
       "tile_1024": bench(lambda: det.det(tile, conf=0.3), 20),
       "quadro_1280x720_inteiro": bench(lambda: det(frame, 0.3), 10),
       "foto_4000x3000_fatiada": bench(lambda: det(img, 0.3), 5, warm=1)}
tr = ByteTracker(); d = np.c_[np.random.default_rng(0).uniform(0, 1000, (30, 2)), np.random.default_rng(0).uniform(1040, 1100, (30, 2)), np.full(30, 0.9)]
d[:, 2:4] = d[:, :2] + 40
out["bytetrack_update_30_deteccoes"] = bench(lambda: tr.update(d), 300)
(ROOT / "experiments/benchmark.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
