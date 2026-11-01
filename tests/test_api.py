import io
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import Settings
from backend.app.main import create_app
from cattlecount.sim import noisy_detections, pan_windows, render_pan


class FakeImageDetector:
    """Detector falso determinístico para testar a API (NÃO é o modelo real)."""
    def __call__(self, rgb, conf=0.25):
        d = np.array([[10, 10, 50, 50, 0.9], [100, 100, 140, 150, 0.6], [200, 20, 230, 60, 0.1]], float)
        return d[d[:, 4] >= conf]


def _scene():
    img = np.random.default_rng(0).integers(0, 255, (400, 1200, 3), dtype=np.uint8)
    boxes = np.array([[150 + 100 * i, 150 + 20 * (i % 3), 190 + 100 * i, 190 + 20 * (i % 3)] for i in range(10)], float)
    return render_pan(img, boxes, (400, 300), pan_windows((400, 1200), (400, 300), 60, (0, 50), (800, 50)))


class VideoDetector:
    def __init__(self, dets):
        self.dets, self.k = dets, 0

    def __call__(self, rgb, conf=0.1):
        self.k += 1
        return self.dets[self.k]


def _client(tmp_path, det, **kw):
    st = Settings(database_url=f"sqlite:///{tmp_path}/t.db", data_dir=tmp_path / "jobs", **kw)
    return TestClient(create_app(st, det, "fake"))


def _jpg(w=800, h=600):
    b = io.BytesIO(); Image.fromarray(np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(b, "JPEG"); return b.getvalue()


def _video(tmp_path, frames):
    p = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (400, 300))
    for f in frames:
        w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    w.release()
    return p


def test_health_limits_e_tres_contagens(tmp_path):
    c = _client(tmp_path, FakeImageDetector())
    assert c.get("/health").json()["model_loaded"] is True
    lim = c.get("/limits").json()
    assert set(lim["tres_contagens"]) == {"visiveis_instantanea", "passagens_acumuladas", "estimativa_de_rebanho"}
    assert any("não validado" in x for x in lim["limites"])


def test_imagem_conta_visiveis_respeitando_conf(tmp_path):
    c = _client(tmp_path, FakeImageDetector())
    r = c.post("/infer/image", files={"file": ("a.jpg", _jpg(), "image/jpeg")}, data={"conf": "0.25"}).json()
    assert r["visible_count"] == 2 == len(r["detections"]) and r["annotated_jpeg_base64"] and "VISÍVEIS" in r["interpretation"]
    assert r["analysis_id"] == c.get("/analyses").json()[0]["id"]
    assert any("altitude" in w for w in r["warnings"])


def test_entradas_invalidas(tmp_path):
    c = _client(tmp_path, FakeImageDetector())
    assert c.post("/infer/image", files={"file": ("a.jpg", b"lixo", "image/jpeg")}).status_code == 415
    assert c.post("/infer/image", files={"file": ("a.jpg", _jpg(), "image/jpeg")}, data={"conf": "5"}).status_code == 422
    assert c.post("/videos", files={"file": ("v.mp4", b"lixo", "video/mp4")}).status_code == 415
    assert c.get("/analyses/999").status_code == 404
    small = _client(tmp_path / "s", FakeImageDetector(), max_upload_mb=0)
    assert small.post("/infer/image", files={"file": ("a.jpg", _jpg(), "image/jpeg")}).status_code == 413


def test_sem_modelo_retorna_503(tmp_path):
    c = _client(tmp_path, None, model_onnx=tmp_path / "inexistente.onnx")
    assert c.get("/health").json()["model_loaded"] is False
    assert c.post("/infer/image", files={"file": ("a.jpg", _jpg(), "image/jpeg")}).status_code == 503


def test_api_key(tmp_path):
    c = _client(tmp_path, FakeImageDetector(), api_key="segredo")
    f = {"file": ("a.jpg", _jpg(), "image/jpeg")}
    assert c.post("/infer/image", files=f).status_code == 401
    assert c.post("/infer/image", files=f, headers={"x-api-key": "segredo"}).status_code == 200
    assert c.get("/health").status_code == 200


def test_fluxo_video_recontagem_exportacoes_e_auditoria(tmp_path):
    frames, gt = _scene()
    dets = noisy_detections(gt, 60, seed=1, recall=0.97, fp_per_frame=0.0, win_wh=(400, 300))
    c = _client(tmp_path, VideoDetector(dets))
    p = _video(tmp_path, frames)
    cfg = {"lines": [{"a": [200, 0], "b": [200, 300], "name": "L"}]}
    r = c.post("/videos", files={"file": ("v.mp4", p.read_bytes(), "video/mp4")}, data={"conf": "0.25", "count_config": __import__("json").dumps(cfg)})
    assert r.status_code == 202
    aid = r.json()["analysis_id"]
    for _ in range(100):
        a = c.get(f"/analyses/{aid}").json()
        if a["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert a["status"] == "done", a.get("error")
    assert len(a["runs"]) == 1 and a["runs"][0]["totals"]["L"]["total_passagens"] >= 1
    assert "aviso" in a["runs"][0]["herd_bounds"]
    # recontagem com outra linha, sem refazer a detecção (o detector falso só tem 60 quadros: um novo uso falharia)
    r2 = c.post(f"/analyses/{aid}/count", json={"config": {"lines": [{"a": [100, 0], "b": [100, 300], "name": "L2"}]}, "stitch": True}).json()
    assert r2["run_id"] != a["runs"][0]["run_id"] and "L2" in r2["totals"] and r2["stitching"]["aviso"]
    assert c.post(f"/analyses/{aid}/count", json={"config": {"lines": [{"a": [5, 5], "b": [5, 5]}]}}).status_code == 422
    rid = r2["run_id"]
    ev = c.get(f"/analyses/{aid}/runs/{rid}/events").json()
    assert len(ev) == r2["totals"]["L2"]["total_passagens"] and all(e["config"] == r2["config_digest"] for e in ev)
    assert c.get(f"/analyses/{aid}/runs/{rid}/events.csv").text.startswith("event_id,type")
    assert c.get(f"/analyses/{aid}/runs/{rid}/series.csv").text.startswith("k,src_frame")
    assert c.get(f"/analyses/{aid}/tracks.csv").text.startswith("k,src_frame,track_id")
    tr = c.get(f"/analyses/{aid}/tracks").json()
    assert len(tr["frames"]) == 60
    rep = c.get(f"/analyses/{aid}/report").json()
    assert rep["contagem"]["run_id"] == rid and len(rep["historico_de_recontagens"]) == 2 and rep["analysis"]["sha256"] == a["sha256"]
    assert any("IDs únicos" in x for x in rep["limitacoes"])
    au = c.get(f"/analyses/{aid}/audit").json()
    assert [x["run_id"] for x in au["runs"]] == [a["runs"][0]["run_id"], rid]
    assert c.get(f"/analyses/{aid}/report.html").status_code == 200
    assert c.get(f"/analyses/{aid}/preview.jpg").headers["content-type"] == "image/jpeg"
    mp4 = c.get(f"/analyses/{aid}/runs/{rid}/annotated.mp4")
    assert mp4.status_code == 200 and len(mp4.content) > 1000
    assert c.delete(f"/analyses/{aid}").status_code == 200 and c.get(f"/analyses/{aid}").status_code == 404


def test_websocket_recebe_progresso(tmp_path):
    frames, gt = _scene()
    dets = noisy_detections(gt, 60, seed=1, recall=1.0, fp_per_frame=0.0, win_wh=(400, 300))
    c = _client(tmp_path, VideoDetector(dets))
    p = _video(tmp_path, frames)
    with c.websocket_connect("/ws") as ws:
        aid = c.post("/videos", files={"file": ("v.mp4", p.read_bytes(), "video/mp4")}).json()["analysis_id"]
        msgs = []
        while not msgs or msgs[-1].get("status") != "done":
            msgs.append(ws.receive_json())
        assert all(m["analysis_id"] == aid for m in msgs) and msgs[-1]["progress"] == 1.0
