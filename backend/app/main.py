"""API CattleCount AI. Dois modos distintos: contagem de animais VISÍVEIS em imagem; contagem de PASSAGENS em vídeo (linha/região)."""
from __future__ import annotations

import asyncio
import csv
import io
import json
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

from cattlecount.ingest import IngestError
from cattlecount.video import events_csv

from .config import Settings
from .db import Analysis, CountRun, make_engine, make_session_factory
from .detector import load_detector
from .service import ADVISORY, Service


class LineIn(BaseModel):
    a: tuple[float, float]
    b: tuple[float, float]
    name: str = "linha1"
    deadband_px: float = Field(6.0, ge=0, le=200)
    cooldown_frames: int = Field(10, ge=0, le=1000)
    margin: float = Field(0.0, ge=0, le=1)


class RegionIn(BaseModel):
    polygon: list[tuple[float, float]] = Field(min_length=3)
    name: str = "regiao1"
    min_frames: int = Field(3, ge=1, le=100)


class CountConfigIn(BaseModel):
    lines: list[LineIn] = []
    regions: list[RegionIn] = []
    min_track_len: int = Field(3, ge=1, le=1000)
    anchor: str = Field("center", pattern="^(center|bottom)$")

    @field_validator("lines")
    @classmethod
    def _line_len(cls, v):
        for ln in v:
            if ln.a == ln.b:
                raise ValueError("linha com pontas coincidentes")
        return v


class CountRequest(BaseModel):
    config: CountConfigIn
    stitch: bool = False


def create_app(settings: Settings | None = None, detector=None, model_id: str | None = None) -> FastAPI:
    st = settings or Settings()
    engine = make_engine(st.database_url)
    Session = make_session_factory(engine)
    if detector is None:
        detector, model_id = load_detector(st)
    svc = Service(st, Session, detector, model_id or "")
    app = FastAPI(title="CattleCount AI", version="1.0.0",
                  description="Detecção de bovinos e contagem em imagens/vídeos. Estimativas de modelo — não é censo do rebanho.")
    app.add_middleware(CORSMiddleware, allow_origins=st.cors_origins, allow_methods=["*"], allow_headers=["*"])
    app.state.svc, app.state.settings = svc, st

    def auth(x_api_key: str | None = Header(default=None)):
        if st.api_key and x_api_key != st.api_key:
            raise HTTPException(401, "chave de API ausente ou inválida")

    def get(aid: int) -> Analysis:
        with Session() as db:
            a = db.get(Analysis, aid)
        if a is None:
            raise HTTPException(404, "análise não encontrada")
        return a

    async def read_limited(file: UploadFile, to_path: Path | None = None) -> bytes | Path:
        lim, n = st.max_upload_mb * 1_000_000, 0
        buf = io.BytesIO() if to_path is None else open(to_path, "wb")
        try:
            while chunk := await file.read(1 << 20):
                n += len(chunk)
                if n > lim:
                    raise HTTPException(413, f"arquivo acima do limite de {st.max_upload_mb} MB")
                buf.write(chunk)
        finally:
            if to_path is not None:
                buf.close()
        return buf.getvalue() if to_path is None else to_path

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": svc.detector is not None, "model_id": svc.model_id, "auth_required": bool(st.api_key)}

    @app.get("/limits")
    def limits():
        return {"advisory": ADVISORY, "tres_contagens": {
            "visiveis_instantanea": "animais detectados em uma imagem/quadro; não inclui ocultos nem fora do quadro",
            "passagens_acumuladas": "eventos de cruzamento de linha / entrada-saída de região; depende de rastreamento, FPS e posição da linha",
            "estimativa_de_rebanho": "NÃO calculada como número único; apenas faixas de referência com avisos (piso observado e IDs únicos)"},
            "limites": ["Treinado no ICAERUS (bovinos em pastagem vistos de drone, fazendas na Europa, 60–100 m): não validado para outras raças, relevo, altitudes ou iluminação.",
                        "Rastreamento em vídeo AÉREO não foi medido contra verdade real (não há GT público); validação do rastreador feita em vídeo terrestre e sintético.",
                        "Câmera móvel: a linha é referenciada ao quadro.", "Sem autenticação por usuário; defina API_KEY para exigir chave."]}

    @app.get("/models")
    def models():
        p = st.root / "models/registry.json"
        return json.loads(p.read_text()) if p.exists() else []

    @app.get("/evaluations")
    def evaluations():
        p = st.root / "experiments/registry.json"
        return json.loads(p.read_text()) if p.exists() else []

    @app.post("/infer/image", dependencies=[Depends(auth)])
    async def infer_image(file: UploadFile = File(...), conf: float = Form(0.25), with_image: bool = Form(True)):
        if not 0.01 <= conf <= 0.99:
            raise HTTPException(422, "conf deve estar entre 0,01 e 0,99")
        data = await read_limited(file)
        try:
            return await asyncio.to_thread(svc.infer_image, data, file.filename or "", conf, with_image)
        except IngestError as e:
            raise HTTPException(415, str(e))
        except RuntimeError as e:
            raise HTTPException(503, str(e))

    @app.post("/videos", status_code=202, dependencies=[Depends(auth)])
    async def upload_video(file: UploadFile = File(...), conf: float = Form(0.25), stride: int = Form(1), compensate_motion: bool = Form(False),
                           stitch: bool = Form(False), count_config: str = Form("")):
        if not 0.01 <= conf <= 0.99 or not 1 <= stride <= 30:
            raise HTTPException(422, "parâmetros fora do intervalo (conf 0,01–0,99; stride 1–30)")
        cc = None
        if count_config:
            try:
                cc = CountConfigIn.model_validate_json(count_config).model_dump()
            except Exception as e:  # noqa: BLE001
                raise HTTPException(422, f"count_config inválido: {e}")
        tmp = Path(tempfile.mkstemp(suffix=Path(file.filename or "v.mp4").suffix, dir=st.data_dir)[1])
        await read_limited(file, tmp)
        try:
            aid = svc.submit_video(tmp, file.filename or "video.mp4", {"conf": conf, "stride": stride, "compensate_motion": compensate_motion, "stitch": stitch, "count_config": cc})
        except IngestError as e:
            tmp.unlink(missing_ok=True)
            raise HTTPException(415, str(e))
        except RuntimeError as e:
            tmp.unlink(missing_ok=True)
            raise HTTPException(503, str(e))
        return {"analysis_id": aid, "status": "queued"}

    @app.get("/analyses")
    def list_analyses(limit: int = 50):
        with Session() as db:
            rows = db.query(Analysis).order_by(Analysis.id.desc()).limit(min(limit, 200)).all()
            return [{"id": a.id, "kind": a.kind, "filename": a.filename, "status": a.status, "progress": a.progress, "created_at": a.created_at.isoformat(),
                     "n_runs": len(a.runs), "visible_count": a.result.get("visible_count") if a.kind == "image" else None} for a in rows]

    @app.get("/analyses/{aid}")
    def analysis(aid: int):
        a = get(aid)
        with Session() as db:
            runs = db.query(CountRun).filter_by(analysis_id=aid).order_by(CountRun.id).all()
            rr = [{"run_id": r.id, "digest": r.digest, "totals": r.totals, "herd_bounds": r.herd_bounds, "options": r.options, "created_at": r.created_at.isoformat()} for r in runs]
        return {"id": a.id, "kind": a.kind, "filename": a.filename, "sha256": a.sha256, "status": a.status, "progress": a.progress, "error": a.error, "params": a.params,
                "ingest": a.ingest, "result": a.result, "model_id": a.model_id, "runs": rr}

    @app.delete("/analyses/{aid}", dependencies=[Depends(auth)])
    def delete_analysis(aid: int):
        get(aid)
        import shutil
        with Session() as db:
            db.delete(db.get(Analysis, aid)); db.commit()
        shutil.rmtree(st.data_dir / str(aid), ignore_errors=True)
        return {"deleted": aid}

    @app.get("/analyses/{aid}/preview.jpg")
    def preview(aid: int, k: int = 0):
        import cv2
        a = get(aid)
        if a.kind != "video":
            raise HTTPException(404, "apenas vídeos")
        inp = next(svc.job_dir(aid).glob("input.*"), None)
        if inp is None:
            raise HTTPException(404, "arquivo ausente")
        cap = cv2.VideoCapture(str(inp)); cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read(); cap.release()
        if not ok:
            raise HTTPException(404, "quadro inexistente")
        return Response(cv2.imencode(".jpg", f)[1].tobytes(), media_type="image/jpeg")

    @app.get("/analyses/{aid}/tracks")
    def tracks(aid: int, max_frames: int = 5000):
        a = get(aid)
        if a.status != "done":
            raise HTTPException(409, f"status: {a.status}")
        res = svc.load_result(aid)
        return {"fps_efetivo": a.result["fps_efetivo"], "frames": {str(k): [[round(float(v), 1) for v in r[:4]] + [int(r[4]), round(float(r[5]), 3)] for r in rows]
                                                                for k, rows in list(res["tracks"].items())[:max_frames]}}

    @app.post("/analyses/{aid}/count", dependencies=[Depends(auth)])
    async def count(aid: int, req: CountRequest):
        try:
            return await asyncio.to_thread(svc.recount, aid, req.config.model_dump(), {"stitch": req.stitch})
        except KeyError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.get("/analyses/{aid}/runs/{rid}/events")
    def events(aid: int, rid: int):
        try:
            return svc.events(aid, rid)
        except KeyError as e:
            raise HTTPException(404, str(e))

    @app.get("/analyses/{aid}/runs/{rid}/events.csv")
    def events_csv_ep(aid: int, rid: int):
        try:
            ev = svc.events(aid, rid)
        except KeyError as e:
            raise HTTPException(404, str(e))
        buf = io.StringIO()
        import tempfile as tf
        with tf.NamedTemporaryFile("w+", suffix=".csv", newline="") as f:
            events_csv(ev, f.name); buf.write(Path(f.name).read_text())
        return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=eventos_{aid}_{rid}.csv"})

    @app.get("/analyses/{aid}/runs/{rid}/series.csv")
    def series_csv(aid: int, rid: int):
        from .service import count_config_from_dict
        from cattlecount import counting as C
        a = get(aid)
        with Session() as db:
            run = db.get(CountRun, rid)
        if run is None or run.analysis_id != aid:
            raise HTTPException(404, "run não encontrado")
        res = svc.load_result(aid)
        out = C.run_counting(res["tracks"], count_config_from_dict(run.config), a.result["fps_efetivo"], res["n_det"])
        w = io.StringIO(); cw = csv.writer(w); cw.writerow(["k", "src_frame", "t_s", "visiveis_bruto", "visiveis_suavizado"])
        fps = a.result["fps_efetivo"]
        for k, c, s in zip(out["instantaneous"]["frames"], out["instantaneous"]["count"], out["instantaneous"]["count_smoothed"]):
            cw.writerow([k, res["src_index"][k], round(k / fps, 3), int(c), s])
        return Response(w.getvalue(), media_type="text/csv")

    @app.get("/analyses/{aid}/tracks.csv")
    def tracks_csv(aid: int):
        get(aid)
        p = svc.job_dir(aid) / "tracks.csv"
        if not p.exists():
            raise HTTPException(404, "trilhas ainda não geradas")
        return FileResponse(p, media_type="text/csv", filename=f"trilhas_{aid}.csv")

    @app.get("/analyses/{aid}/runs/{rid}/annotated.mp4")
    async def annotated(aid: int, rid: int):
        get(aid)
        try:
            p = await asyncio.to_thread(svc.render_annotated, aid, rid)
        except KeyError as e:
            raise HTTPException(404, str(e))
        return FileResponse(p, media_type="video/mp4")

    @app.get("/analyses/{aid}/report")
    def report(aid: int, run_id: int | None = None):
        get(aid)
        return svc.report(aid, run_id)

    @app.get("/analyses/{aid}/report.html", response_class=HTMLResponse)
    def report_html(aid: int, run_id: int | None = None):
        get(aid)
        r = svc.report(aid, run_id)
        esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;")  # noqa: E731
        return HTMLResponse(f"<!doctype html><meta charset=utf-8><title>Relatório CattleCount #{aid}</title><body style='font-family:sans-serif;max-width:900px;margin:2rem auto'>"
                            f"<h1>Relatório de análise #{aid}</h1><p><b>{esc(r['advisory'])}</b></p><pre style='white-space:pre-wrap'>{esc(json.dumps(r, indent=1, ensure_ascii=False))}</pre></body>")

    @app.get("/analyses/{aid}/audit")
    def audit(aid: int):
        a = get(aid)
        with Session() as db:
            runs = db.query(CountRun).filter_by(analysis_id=aid).order_by(CountRun.id).all()
            return {"analysis_id": aid, "sha256": a.sha256, "model_id": a.model_id, "params": a.params, "ingest_warnings": a.ingest.get("warnings", []),
                    "runs": [{"run_id": r.id, "digest": r.digest, "created_at": r.created_at.isoformat(), "options": r.options, "n_events": len(r.events),
                              "sinalizados": [e.event_no for e in r.events if e.flags]} for r in runs]}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        if st.api_key and sock.query_params.get("api_key") != st.api_key:
            await sock.close(code=4401)
            return
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        push = lambda m: loop.call_soon_threadsafe(q.put_nowait, m)  # noqa: E731
        svc.subs.append(push)
        try:
            while True:
                await sock.send_json(await q.get())
        except Exception:  # noqa: BLE001
            pass
        finally:
            svc.subs.remove(push)

    return app

