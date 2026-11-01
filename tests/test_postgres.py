"""Integração com PostgreSQL REAL (servidor local via pgserver). Pulado se o pgserver não estiver disponível."""
import time

import pytest

pgserver = pytest.importorskip("pgserver")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.config import Settings  # noqa: E402
from backend.app.main import create_app  # noqa: E402
from tests.test_api import VideoDetector, _scene, _video  # noqa: E402
from cattlecount.sim import noisy_detections  # noqa: E402


def test_fluxo_completo_em_postgres(tmp_path):
    srv = pgserver.get_server(tmp_path / "pg", cleanup_mode="stop")
    url = srv.get_uri().replace("postgresql://", "postgresql+psycopg://", 1)
    frames, gt = _scene()
    dets = noisy_detections(gt, 60, seed=1, recall=1.0, fp_per_frame=0.0, win_wh=(400, 300))
    c = TestClient(create_app(Settings(database_url=url, data_dir=tmp_path / "jobs"), VideoDetector(dets), "fake"))
    p = _video(tmp_path, frames)
    aid = c.post("/videos", files={"file": ("v.mp4", p.read_bytes(), "video/mp4")}).json()["analysis_id"]
    for _ in range(100):
        a = c.get(f"/analyses/{aid}").json()
        if a["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert a["status"] == "done", a.get("error")
    r = c.post(f"/analyses/{aid}/count", json={"config": {"lines": [{"a": [200, 0], "b": [200, 300], "name": "L"}]}}).json()
    assert len(c.get(f"/analyses/{aid}/runs/{r['run_id']}/events").json()) == r["totals"]["L"]["total_passagens"] > 0
    assert c.delete(f"/analyses/{aid}").status_code == 200       # cascade em count_runs/events
    from sqlalchemy import create_engine, text
    with create_engine(url).connect() as cx:
        assert cx.execute(text("select count(*) from events")).scalar() == 0 and cx.execute(text("select version()")).scalar().startswith("PostgreSQL")
