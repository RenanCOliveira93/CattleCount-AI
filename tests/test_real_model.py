"""Teste com o modelo REAL (ONNX) e uma imagem REAL do ICAERUS. Pulado se o modelo ou os dados não existirem."""
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app

ROOT = Path(__file__).resolve().parents[1]
ONNX = ROOT / "models/onnx/cattle_yolo11s_1024.onnx"
PREDS = ROOT / "runs/preds/full_val.json"
pytestmark = pytest.mark.skipif(not (ONNX.exists() and PREDS.exists() and (ROOT / "data/yolo_tiles/manifest.json").exists()), reason="modelo/dados ausentes")


def test_api_onnx_fatiado_concorda_com_pytorch(tmp_path):
    rows = json.load(open(PREDS))["images"]
    man = {m["stem"]: m for m in json.load(open(ROOT / "data/yolo_tiles/manifest.json"))["items"]}
    r = max(rows, key=lambda r: r["n_gt"])
    ref = int((np.array(r["pred"]).reshape(-1, 5)[:, 4] >= 0.25).sum())     # contagem do pipeline PyTorch (predict_split.py)
    c = TestClient(create_app(Settings(database_url=f"sqlite:///{tmp_path}/t.db", data_dir=tmp_path / "j", model_onnx=ONNX)))
    assert c.get("/health").json()["model_loaded"] is True
    with open(ROOT / man[r["stem"]]["path"], "rb") as f:
        j = c.post("/infer/image", files={"file": ("a.jpg", f, "image/jpeg")}, data={"conf": "0.25"}).json()
    assert abs(j["visible_count"] - ref) <= max(2, 0.1 * ref), (j["visible_count"], ref)
    assert j["image"] == {"width": 4000, "height": 3000} and all(d["score"] >= 0.25 for d in j["detections"])
