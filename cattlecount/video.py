"""Pipeline de vídeo: detecção por quadro → ByteTrack → trilhas persistidas → (re)contagem → vídeo anotado e exportações.
As trilhas são salvas (tracks.csv): linha/ROI podem ser reconfiguradas e recontadas SEM refazer a detecção."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .counting import CountConfig, build_trajectories
from .tracking import ByteTracker, GlobalMotion, TrackerConfig


def iter_frames(path: str, stride: int = 1):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"não foi possível abrir o vídeo: {path}")
    i = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield i, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()


def video_info(path: str) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"não foi possível abrir o vídeo: {path}")
    info = {"fps": cap.get(cv2.CAP_PROP_FPS) or 0.0, "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def track_frames(frames, detector: Callable[[np.ndarray], np.ndarray], tracker_cfg: TrackerConfig | None = None, compensate_motion: bool = False,
                 on_progress: Callable[[int], None] | None = None, count_conf: float | None = None) -> dict:
    """frames: iterável de (índice_origem, RGB). detector(rgb)->(n,5). Retorna {tracks: {k: (m,6)}, n_det: {k: n}, src_index: {k: i}}; k = 1,2,… (analisados).
    `n_det` = detecções com score ≥ `count_conf` (a confiança escolhida pelo usuário). O rastreador pode receber detecções mais fracas (2ª etapa do ByteTrack),
    mas elas NÃO entram na contagem de animais visíveis."""
    tr = ByteTracker(tracker_cfg)
    gm = GlobalMotion() if compensate_motion else None
    tracks, n_det, src = {}, {}, {}
    for k, (i, rgb) in enumerate(frames, 1):
        dets = np.asarray(detector(rgb), float).reshape(-1, 5)
        A = gm.update(rgb) if gm else None
        tracks[k] = tr.update(dets, A)
        n_det[k], src[k] = int(len(dets) if count_conf is None else (dets[:, 4] >= count_conf).sum()), int(i)
        if on_progress:
            on_progress(k)
    return {"tracks": tracks, "n_det": n_det, "src_index": src, "motion_failures": gm.failures if gm else None}


def save_tracks(path: str | Path, res: dict) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "src_frame", "track_id", "x1", "y1", "x2", "y2", "score"])
        for k, rows in res["tracks"].items():
            for r in np.asarray(rows).reshape(-1, 6):
                w.writerow([k, res["src_index"][k], int(r[4]), *[round(float(v), 2) for v in r[:4]], round(float(r[5]), 4)])


def load_tracks(path: str | Path, n_det: dict | None = None) -> dict:
    tracks, src = {}, {}
    with open(path) as f:
        for r in csv.DictReader(f):
            k = int(r["k"]); src[k] = int(r["src_frame"])
            tracks.setdefault(k, []).append([float(r[c]) for c in ("x1", "y1", "x2", "y2")] + [int(r["track_id"]), float(r["score"])])
    ks = range(1, max(src) + 1) if src else []
    return {"tracks": {k: np.array(tracks.get(k, []), float).reshape(-1, 6) for k in ks}, "src_index": src, "n_det": n_det}


def events_csv(events: list[dict], path: str | Path) -> None:
    cols = ["event_id", "type", "line", "region", "direction", "track_id", "frame", "t_s", "x", "y", "track_len", "mean_score", "flags", "config"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for e in events:
            w.writerow({**e, "flags": "|".join(e.get("flags", []))})


def _color(i: int):
    rng = np.random.default_rng(i * 7919)
    return tuple(int(v) for v in rng.integers(60, 255, 3))


def annotate_video(src_path: str, out_path: str, res: dict, cfg: CountConfig, counting: dict, stride: int = 1, trail: int = 40, fourcc: str = "mp4v") -> None:
    """Desenha caixas, IDs, rastros, linha/regiões e contagens (instantânea e acumulada separadas). Quadros pulados repetem a última anotação."""
    info = video_info(src_path)
    wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc), max(info["fps"] / stride, 1.0), (info["width"], info["height"]))
    traj = build_trajectories(res["tracks"], cfg.anchor)
    ev_by_k: dict[int, list] = {}
    for e in counting["events"]:
        ev_by_k.setdefault(e["frame"], []).append(e)
    cum = {}
    inst = dict(zip(counting["instantaneous"]["frames"], counting["instantaneous"]["count_smoothed"]))
    for k, (i, rgb) in enumerate(iter_frames(src_path, stride), 1):
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for ln in cfg.lines:
            cv2.line(img, tuple(map(int, ln.a)), tuple(map(int, ln.b)), (0, 255, 255), 2)
        for rg in cfg.regions:
            cv2.polylines(img, [np.array(rg.polygon, np.int32)], True, (255, 160, 0), 2)
        for r in np.asarray(res["tracks"].get(k, [])).reshape(-1, 6):
            tid, col = int(r[4]), _color(int(r[4]))
            cv2.rectangle(img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), col, 2)
            cv2.putText(img, f"#{tid}", (int(r[0]), max(12, int(r[1]) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            t = traj.get(tid)
            if t:
                m = [j for j, fr in enumerate(t["frames"]) if k - trail <= fr <= k]
                if len(m) > 1:
                    cv2.polylines(img, [t["pts"][m].astype(np.int32)], False, col, 1, cv2.LINE_AA)
        for e in ev_by_k.get(k, []):
            key = e.get("line") or e.get("region")
            cum[(key, e["direction"])] = cum.get((key, e["direction"]), 0) + 1
        txt = [f"visiveis (inst., suavizado): {int(inst.get(k, 0))}"] + [f"{a} {d}: {n}" for (a, d), n in sorted(cum.items())]
        for j, s in enumerate(txt):
            cv2.putText(img, s, (10, 24 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, s, (10, 24 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        wr.write(img)
    wr.release()


def write_meta(path: str | Path, **kw) -> None:
    Path(path).write_text(json.dumps(kw, indent=1, ensure_ascii=False, default=str))
