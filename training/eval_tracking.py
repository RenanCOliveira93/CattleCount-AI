"""Avaliação do rastreador (parâmetros PADRÃO, sem ajuste nestes dados) em dois cenários:
 A) vídeo SINTÉTICO (pan sobre foto aérea real) com detector real e GT exato  → HOTA/IDF1 + contagem de passagens pela linha vs GT;
 B) vídeo TERRESTRE real (Mendeley, GT de 1 clipe com 8 animais, viciado em YOLOv8+DeepSORT) com detector COCO → ByteTrack próprio vs baseline IoU vs BYTETracker do Ultralytics.
Saída: experiments/tracking_eval.json"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.detector import OnnxCattleDetector  # noqa: E402
from cattlecount import counting as C  # noqa: E402
from cattlecount.geometry import iou_matrix  # noqa: E402
from cattlecount.mot_metrics import evaluate_mot, tracks_to_rows  # noqa: E402
from cattlecount.tracking import TrackerConfig  # noqa: E402
from cattlecount.video import iter_frames, track_frames  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

CACHE = ROOT / "runs/tracking"; CACHE.mkdir(parents=True, exist_ok=True)


def bt(th):
    from cattlecount.tracking import ByteTracker
    return lambda: ByteTracker(TrackerConfig(high_thresh=th["high"], low_thresh=th["low"], new_track_thresh=th["new"]))


class IouTracker:
    """Baseline mínimo: associação gulosa/húngara por IoU quadro a quadro, sem Kalman, sem 2ª etapa, sem buffer de perdidos (IoU ≥ 0,3)."""
    def __init__(self, thr=0.3, min_score=0.5):
        self.prev, self.next, self.thr, self.ms = np.zeros((0, 5)), 1, thr, min_score

    def update(self, dets, affine=None):
        d = np.asarray(dets, float).reshape(-1, 5); d = d[d[:, 4] >= self.ms]
        out = np.zeros((len(d), 6)); out[:, :4], out[:, 5] = d[:, :4], d[:, 4]; out[:, 4] = -1
        if len(self.prev) and len(d):
            I = iou_matrix(self.prev[:, :4], d[:, :4]); r, c = linear_sum_assignment(-I)
            for a, b in zip(r, c):
                if I[a, b] >= self.thr: out[b, 4] = self.prev[a, 4]
        for i in np.where(out[:, 4] < 0)[0]:
            out[i, 4] = self.next; self.next += 1
        self.prev = out
        return out


AERIAL = dict(high=0.25, low=0.1, new=0.35)   # derivados do conf operacional do detector aéreo (0,25, escolhido no VAL) — mesma regra da API
COCO_ = dict(high=0.5, low=0.1, new=0.6)    # padrões do ByteTrack original


class UltralyticsByte:
    """BYTETracker do Ultralytics (mesmas detecções, mesmos limiares padrão) — verificação cruzada da implementação própria."""
    def __init__(self, th=COCO_):
        from ultralytics.trackers.byte_tracker import BYTETracker
        self.t = BYTETracker(SimpleNamespace(track_high_thresh=th["high"], track_low_thresh=th["low"], new_track_thresh=th["new"], track_buffer=30, match_thresh=0.8, fuse_score=True))

    def update(self, dets, affine=None):
        r = self.t.update(_R(np.asarray(dets, float).reshape(-1, 5)), None)
        return np.array([[*x[:4], x[4], x[5]] for x in r], float).reshape(-1, 6)


class _R:
    def __init__(self, d):
        self.conf = d[:, 4]; self.cls = np.zeros(len(d)); self.xyxy = d[:, :4]
        self.xywh = np.c_[(d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2, d[:, 2] - d[:, 0], d[:, 3] - d[:, 1]] if len(d) else np.zeros((0, 4))

    def __len__(self): return len(self.conf)
    def __getitem__(self, i): d = np.c_[self.xyxy, self.conf][i]; return _R(d.reshape(-1, 5))


def run(tracker_factory, frames_dets, compensate=False, frames=None):
    tr = tracker_factory(); out = {}
    from cattlecount.tracking import GlobalMotion
    gm = GlobalMotion() if compensate else None
    for k, d in enumerate(frames_dets, 1):
        A = gm.update(frames[k - 1]) if gm else None
        out[k] = tr.update(d, A)
    return out


def detections(video, detector, name, stride=1):
    f = CACHE / f"{name}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True); return list(z["d"]), None
    ds = []
    for _, rgb in iter_frames(video, stride):
        ds.append(np.asarray(detector(rgb), float).reshape(-1, 5))
    np.savez(f, d=np.array(ds, dtype=object)); return ds, None


def score(per_frame, gt):
    rows = tracks_to_rows(per_frame)
    m = evaluate_mot(gt, rows)
    m = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
    m3 = evaluate_mot(gt, rows, iou_thr=0.3)      # objetos minúsculos: IoU 0,5 é severo; reportamos também a 0,3
    m.update({"IDF1_iou0.3": round(m3["IDF1"], 4), "IDSW_iou0.3": m3["IDSW"], "MOTA_iou0.3": round(m3["MOTA"], 4)})
    return m


def passages(per_frame, x_line, h, fps):
    cfg = C.CountConfig([C.LineSpec((x_line, 0), (x_line, h), "L", deadband_px=6)])
    t = C.run_counting(per_frame, cfg, fps)["totals"]["L"]
    return t


res = {}
# ---------------- A) sintético ----------------
vid = str(ROOT / "sample_data/demo_pan_synthetic.mp4")
gtA = np.loadtxt(ROOT / "sample_data/demo_pan_synthetic_gt.txt", delimiter=",")[:, :6]
det = OnnxCattleDetector(ROOT / "models/onnx/cattle_yolo11s_1024.onnx", "sliced")
dA, _ = detections(vid, lambda rgb: det(rgb, 0.1), "synthetic_pan_full")
frames = [f for _, f in iter_frames(vid)]
n = len(dA)
gtpf = {int(k): np.array([[*(r[2:4]), r[2] + r[4], r[3] + r[5], r[1], 1.0] for r in gtA[gtA[:, 0] == k]]).reshape(-1, 6) for k in range(1, n + 1)}
x_line = 640
res["A_sintetico"] = {"aviso": "movimento de câmera SINTÉTICO sobre foto real; animais estáticos; detector aéreo real; parâmetros padrão do tracker", "n_frames": n, "gt_ids": int(len(set(gtA[:, 1]))),
                      "passagens_GT_linha_x640": passages(gtpf, x_line, 720, 10), "metodos": {}}
hi = [d[d[:, 4] >= 0.25] for d in dA]
res["A_sintetico"]["deteccao_isolada"] = {"deteccoes_por_quadro_media": float(np.mean([len(d) for d in hi])), "gt_visiveis_por_quadro_media": float(np.mean([len(gtpf[k]) for k in gtpf]))}
for name, fac, comp in (("ByteTrack próprio", bt(AERIAL), False), ("ByteTrack próprio + compensação de câmera", bt(AERIAL), True),
                        ("ByteTrack Ultralytics", lambda: UltralyticsByte(AERIAL), False), ("Baseline IoU", lambda: IouTracker(0.3, 0.3), False)):
    pf = run(fac, dA, comp, frames)
    m = score(pf, gtA)
    m["passagens_linha_x640"] = passages(pf, x_line, 720, 10)
    res["A_sintetico"]["metodos"][name] = m
    print("A", name, {k: m[k] for k in ("HOTA", "IDF1", "IDF1_iou0.3", "IDSW", "n_pred_ids")}, m["passagens_linha_x640"])
# teto do rastreador: detecções PERFEITAS (as próprias caixas do GT, score 0,9) — isola o erro do rastreador do erro do detector
orc = [np.array([[*r[2:4], r[2] + r[4], r[3] + r[5], 0.9] for r in gtA[gtA[:, 0] == k]]).reshape(-1, 5) for k in range(1, n + 1)]
res["A_sintetico"]["metodos_com_deteccao_perfeita"] = {}
for name, fac in (("ByteTrack próprio", bt(AERIAL)), ("Baseline IoU", lambda: IouTracker(0.3, 0.3))):
    m = score(run(fac, orc), gtA); res["A_sintetico"]["metodos_com_deteccao_perfeita"][name] = m
    print("A-oracle", name, {k: m[k] for k in ("HOTA", "IDF1", "IDSW", "n_pred_ids")})
# ---------------- B) terrestre Mendeley ----------------
from ultralytics import YOLO  # noqa: E402
tv = str(ROOT / "data/raw/mendeley/ex/Ground Truth/Toros.mp4")
gtB = np.loadtxt(ROOT / "data/raw/mendeley/ex/Ground Truth/gt.txt", delimiter=",")[:, :6]; gtB[:, 0] += 1
yo = YOLO(str(ROOT / "models/pretrained/yolo11s.pt"))


def coco(rgb):
    r = yo.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), imgsz=960, conf=0.1, classes=[19], device="mps", verbose=False)[0]
    return np.c_[r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()]


dB, _ = detections(tv, coco, "toros_coco_yolo11s")
res["B_terrestre_mendeley"] = {"aviso": "GT de UM clipe (o arquivo tem 9 identidades; a descrição do autor cita 8 animais) gerado com YOLOv8-COCO+DeepSORT e corrigido: favorece detectores tipo COCO; não é GT aéreo", "n_frames": len(dB), "gt_ids": int(len(set(gtB[:, 1]))), "metodos": {}}
for name, fac in (("ByteTrack próprio", bt(COCO_)), ("ByteTrack Ultralytics", lambda: UltralyticsByte(COCO_)), ("Baseline IoU", IouTracker)):
    pf = run(fac, dB)
    m = score(pf, gtB)
    res["B_terrestre_mendeley"]["metodos"][name] = m
    print("B", name, {k: m[k] for k in ("HOTA", "IDF1", "MOTA", "IDSW", "n_pred_ids")})
(ROOT / "experiments/tracking_eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
