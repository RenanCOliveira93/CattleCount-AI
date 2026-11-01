"""Regras de negócio: inferência em imagem, processamento assíncrono de vídeo, (re)contagem, relatórios. Sem HTTP aqui."""
from __future__ import annotations

import base64
import datetime as dt
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from cattlecount import counting as C
from cattlecount.ingest import IngestError, validate_image_bytes, validate_video_file
from cattlecount.tracking.stitch import StitchConfig, apply_mapping, stitch
from cattlecount.video import annotate_video, events_csv, iter_frames, load_tracks, save_tracks, track_frames, video_info

from .db import Analysis, CountRun, EventRow, now

ADVISORY = ("Contagens são ESTIMATIVAS de um modelo. 'Visíveis agora' (instantânea), 'passagens' (acumulada) e 'IDs únicos' (limite de referência) são "
            "grandezas diferentes; nenhuma delas é, por si só, o tamanho do rebanho. Não use como única base de decisões de manejo, venda ou conformidade.")


def count_config_from_dict(d: dict) -> C.CountConfig:
    return C.CountConfig(
        lines=[C.LineSpec(tuple(x["a"]), tuple(x["b"]), x.get("name", f"linha{i + 1}"), x.get("deadband_px", 6.0), x.get("cooldown_frames", 10), x.get("margin", 0.0))
               for i, x in enumerate(d.get("lines", []))],
        regions=[C.RegionSpec([tuple(p) for p in x["polygon"]], x.get("name", f"regiao{i + 1}"), x.get("min_frames", 3)) for i, x in enumerate(d.get("regions", []))],
        min_track_len=d.get("min_track_len", 3), anchor=d.get("anchor", "center"))


def draw_dets(rgb: np.ndarray, dets: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    t = max(1, round(max(img.shape[:2]) / 1500))
    for x1, y1, x2, y2, s in dets:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), t)
    return img


class Service:
    def __init__(self, settings, session_factory, detector, model_id: str):
        self.s, self.Session, self.detector, self.model_id = settings, session_factory, detector, model_id
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.s.data_dir.mkdir(parents=True, exist_ok=True)
        self.subs: list = []
        self.lock = threading.Lock()

    # ---- imagem ----
    def infer_image(self, data: bytes, filename: str, conf: float, with_image: bool) -> dict:
        if self.detector is None:
            raise RuntimeError("modelo ONNX não encontrado")
        rgb, rep = validate_image_bytes(data, filename)
        import time
        t0 = time.perf_counter()
        dets = np.asarray(self.detector(rgb, conf), float).reshape(-1, 5)
        dets = dets[dets[:, 4] >= conf]
        ms = (time.perf_counter() - t0) * 1000
        out = {"mode": "imagem", "visible_count": int(len(dets)), "conf": conf, "image": {"width": rep.width, "height": rep.height}, "warnings": rep.warnings,
               "ingest": rep.dict(), "detections": [{"xyxy": [round(float(v), 1) for v in d[:4]], "score": round(float(d[4]), 4)} for d in dets],
               "timings_ms": {"infer_ms": round(ms, 1)}, "model_id": self.model_id,
               "interpretation": "Número de animais VISÍVEIS nesta imagem segundo o modelo. Não é contagem de rebanho (animais ocultos/fora do quadro não entram).",
               "advisory": ADVISORY}
        if with_image:
            ok, buf = cv2.imencode(".jpg", draw_dets(rgb, dets), [cv2.IMWRITE_JPEG_QUALITY, 85])
            out["annotated_jpeg_base64"] = base64.b64encode(buf.tobytes()).decode()
        with self.Session() as db:
            a = Analysis(kind="image", filename=filename, sha256=rep.sha256, status="done", progress=1.0, params={"conf": conf}, ingest=rep.dict(),
                         result={k: v for k, v in out.items() if k != "annotated_jpeg_base64"}, model_id=self.model_id, finished_at=now())
            db.add(a); db.commit(); out["analysis_id"] = a.id
        return out

    # ---- vídeo ----
    def job_dir(self, aid: int) -> Path:
        d = self.s.data_dir / str(aid)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def submit_video(self, path_tmp: Path, filename: str, params: dict) -> int:
        rep = validate_video_file(path_tmp)          # lança IngestError
        if self.detector is None:
            raise RuntimeError("modelo ONNX não encontrado")
        with self.Session() as db:
            a = Analysis(kind="video", filename=filename, sha256=rep.sha256, status="queued", params=params, ingest=rep.dict(), model_id=self.model_id)
            db.add(a); db.commit(); aid = a.id
        dest = self.job_dir(aid) / ("input" + Path(filename).suffix.lower())
        shutil.move(str(path_tmp), dest)
        self.pool.submit(self._run_video, aid, dest)
        return aid

    def _set(self, aid, **kw):
        with self.Session() as db:
            a = db.get(Analysis, aid)
            for k, v in kw.items():
                setattr(a, k, v)
            db.commit()
        self.publish({"analysis_id": aid, **{k: v for k, v in kw.items() if k in ("status", "progress", "error")}})

    def _run_video(self, aid: int, path: Path):
        try:
            with self.Session() as db:
                a = db.get(Analysis, aid); p = dict(a.params); n_total = a.ingest["meta"].get("n_frames") or 0
            self._set(aid, status="running")
            stride, conf = int(p.get("stride", 1)), float(p.get("conf", self.s.conf))
            from cattlecount.tracking import TrackerConfig
            tcfg = TrackerConfig(high_thresh=max(conf, 0.3), low_thresh=min(0.1, conf), new_track_thresh=max(conf + 0.1, 0.4))
            step = max(1, n_total // stride // 50) if n_total else 10
            res = track_frames(iter_frames(str(path), stride), lambda rgb: self.detector(rgb, min(0.1, conf)), tcfg, bool(p.get("compensate_motion", False)),
                               count_conf=conf, on_progress=lambda k: (k % step == 0) and self._set(aid, progress=min(0.99, k * stride / max(n_total, 1))))
            d = self.job_dir(aid)
            save_tracks(d / "tracks.csv", res)
            (d / "n_det.json").write_text(json.dumps(res["n_det"]))
            info = video_info(str(path))
            fr = {"frames_analisados": len(res["tracks"]), "stride": stride, "fps_origem": info["fps"], "fps_efetivo": info["fps"] / stride,
                  "motion_failures": res["motion_failures"], "tracker": "ByteTrack (implementação própria)", "conf": conf,
                  "compensate_motion": bool(p.get("compensate_motion", False))}
            self._set(aid, status="done", progress=1.0, result=fr, finished_at=now())
            if p.get("count_config"):
                self.recount(aid, p["count_config"], {"stitch": bool(p.get("stitch", False))})
        except Exception as e:  # noqa: BLE001
            self._set(aid, status="error", error=f"{type(e).__name__}: {e}", finished_at=now())

    def load_result(self, aid: int) -> dict:
        d = self.job_dir(aid)
        nd = json.loads((d / "n_det.json").read_text()) if (d / "n_det.json").exists() else None
        r = load_tracks(d / "tracks.csv", {int(k): v for k, v in nd.items()} if nd else None)
        return r

    def recount(self, aid: int, config: dict, options: dict | None = None) -> dict:
        """Recontagem sobre as trilhas JÁ salvas (sem refazer detecção). `options.stitch` aplica costura limitada de tracklets antes."""
        options = options or {}
        with self.Session() as db:
            a = db.get(Analysis, aid)
            if a is None or a.kind != "video":
                raise KeyError("análise de vídeo não encontrada")
            if a.status != "done":
                raise ValueError(f"análise ainda não concluída (status: {a.status})")
            fps = a.result["fps_efetivo"]
        res = self.load_result(aid)
        cfg = count_config_from_dict(config)
        per_frame, mapping_info = res["tracks"], None
        if options.get("stitch"):
            traj = C.build_trajectories(per_frame, "center")
            mapping = stitch(traj, StitchConfig())
            merged = sum(1 for k, v in mapping.items() if k != v)
            per_frame = apply_mapping(per_frame, mapping)
            mapping_info = {"ids_unidos": merged, "aviso": "Costura por continuidade espaço-temporal; pode unir animais diferentes ou deixar de unir o mesmo animal."}
        out = C.run_counting(per_frame, cfg, fps, res["n_det"])
        out["stitching"] = mapping_info
        out["config"] = config
        with self.Session() as db:
            a = db.get(Analysis, aid)
            run = CountRun(analysis_id=aid, config=config, digest=out["config_digest"], options=options, totals=out["totals"], herd_bounds=out["herd_bounds"])
            for e in out["events"]:
                run.events.append(EventRow(event_no=e["event_id"], type=e["type"], target=e.get("line") or e.get("region", ""), direction=e["direction"], track_id=e["track_id"],
                                           frame=e["frame"], t_s=e["t_s"], x=e["x"], y=e["y"], track_len=e["track_len"], mean_score=e["mean_score"], flags=e["flags"]))
            db.add(run); db.commit(); out["run_id"] = run.id
        self.publish({"analysis_id": aid, "run_id": out["run_id"], "event": "recount"})
        return out

    def render_annotated(self, aid: int, run_id: int) -> Path:
        with self.Session() as db:
            run = db.get(CountRun, run_id)
            if run is None or run.analysis_id != aid:
                raise KeyError("run não encontrado")
            config, options, a = run.config, run.options, db.get(Analysis, aid)
            stride = int(a.params.get("stride", 1))
        d = self.job_dir(aid)
        out = d / f"annotated_{run_id}.mp4"
        if not out.exists():
            res = self.load_result(aid)
            cfg = count_config_from_dict(config)
            per_frame = res["tracks"]
            if options.get("stitch"):
                per_frame = apply_mapping(per_frame, stitch(C.build_trajectories(per_frame), StitchConfig()))
                res = {**res, "tracks": per_frame}
            cnt = C.run_counting(per_frame, cfg, a.result["fps_efetivo"], res["n_det"])
            inp = next(d.glob("input.*"))
            tmp = d / f"annotated_{run_id}_raw.mp4"
            annotate_video(str(inp), str(tmp), res, cfg, cnt, stride)
            out = _to_h264(tmp, out)
        return out

    # ---- relatório e auditoria ----
    def report(self, aid: int, run_id: int | None = None) -> dict:
        with self.Session() as db:
            a = db.get(Analysis, aid)
            if a is None:
                raise KeyError("análise não encontrada")
            runs = sorted(a.runs, key=lambda r: r.id)
            run = next((r for r in runs if r.id == run_id), runs[-1] if runs else None) if run_id or runs else None
            rep = {"analysis": {"id": a.id, "kind": a.kind, "filename": a.filename, "sha256": a.sha256, "status": a.status, "created_at": a.created_at.isoformat(),
                                "model_id": a.model_id, "params": a.params, "ingest_warnings": a.ingest.get("warnings", [])},
                   "processamento": a.result if a.kind == "video" else None, "advisory": ADVISORY}
            if a.kind == "image":
                rep["contagem_visivel"] = {"valor": a.result.get("visible_count"), "conf": a.result.get("conf"), "interpretacao": a.result.get("interpretation")}
            if run:
                rep["contagem"] = {"run_id": run.id, "config_digest": run.digest, "config": run.config, "opcoes": run.options,
                                   "passagens_acumuladas": run.totals, "faixas_de_referencia_do_rebanho": run.herd_bounds,
                                   "n_eventos": len(run.events), "eventos_com_trilha_curta": sum("trilha_curta" in (e.flags or []) for e in run.events)}
            rep["historico_de_recontagens"] = [{"run_id": r.id, "digest": r.digest, "criado_em": r.created_at.isoformat(), "opcoes": r.options} for r in runs]
            rep["limitacoes"] = ["IDs únicos não são população: trocas/fragmentação de ID inflam; animais ocultos ou fora do quadro deflacionam.",
                                 "Passagens por linha dependem da posição da linha, do FPS e da oclusão; revise os eventos sinalizados.",
                                 "Câmera móvel: linha/região são referenciadas ao QUADRO, não ao terreno."]
            return rep

    def events(self, aid: int, run_id: int) -> list[dict]:
        with self.Session() as db:
            run = db.get(CountRun, run_id)
            if run is None or run.analysis_id != aid:
                raise KeyError("run não encontrado")
            return [{"event_id": e.event_no, "type": e.type, "line": e.target if e.type == "line_cross" else None, "region": e.target if e.type == "region" else None,
                     "direction": e.direction, "track_id": e.track_id, "frame": e.frame, "t_s": e.t_s, "x": e.x, "y": e.y, "track_len": e.track_len,
                     "mean_score": e.mean_score, "flags": e.flags or [], "config": run.digest} for e in sorted(run.events, key=lambda x: x.event_no)]

    # ---- websocket pub/sub ----
    def publish(self, msg: dict):
        for q in list(self.subs):
            try:
                q(msg)
            except Exception:  # noqa: BLE001
                pass


def _to_h264(src: Path, dst: Path) -> Path:
    """Reencoda para H.264 (compatível com navegadores) se ffmpeg existir; senão devolve o mp4v (pode não tocar no navegador)."""
    ff = shutil.which("ffmpeg")
    if not ff:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            ff = None
    if not ff:
        src.rename(dst)
        return dst
    import subprocess
    r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)])
    if r.returncode == 0:
        src.unlink(missing_ok=True)
        return dst
    src.rename(dst)
    return dst
