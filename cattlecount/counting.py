"""Motor de eventos de contagem sobre trilhas já calculadas.

TRÊS coisas diferentes (nunca misturar):
  1. contagem INSTANTÂNEA — quantos animais estão visíveis em um quadro/imagem (detecções ou trilhas ativas);
  2. contagem ACUMULADA de PASSAGENS — eventos de cruzamento de linha / entrada-saída de região ao longo do vídeo;
  3. ESTIMATIVA de rebanho — NÃO existe aqui como um número único: só FAIXAS de referência com avisos (`herd_bounds`).
O número de IDs únicos NÃO é a população: inflaciona com trocas/fragmentação e deflaciona com animais ocultos ou fora do quadro.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .geometry import center, point_in_polygon


@dataclass
class LineSpec:
    a: tuple[float, float]
    b: tuple[float, float]
    name: str = "linha"
    deadband_px: float = 6.0       # zona morta: dentro dela o lado anterior é mantido (histerese anti-jitter)
    cooldown_frames: int = 10      # impede reconte do mesmo ID nesta linha logo após um evento
    margin: float = 0.0            # tolerância (fração do comprimento) além das pontas do segmento


@dataclass
class RegionSpec:
    polygon: list[tuple[float, float]]
    name: str = "regiao"
    min_frames: int = 3            # debounce: quadros consecutivos para mudar de estado


@dataclass
class CountConfig:
    lines: list[LineSpec] = field(default_factory=list)
    regions: list[RegionSpec] = field(default_factory=list)
    min_track_len: int = 3         # observações mínimas de um ID para gerar evento (filtra trilhas espúrias)
    anchor: str = "center"         # "center" | "bottom"

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=list).encode()).hexdigest()[:12]


def _points(rows: np.ndarray, anchor: str) -> np.ndarray:
    c = center(rows[:, :4])
    if anchor == "bottom":
        c[:, 1] = rows[:, 3]
    return c


def build_trajectories(per_frame: dict[int, np.ndarray], anchor: str = "center") -> dict[int, dict]:
    """{frame: (k,6)} → {id: {frames, pts, boxes, scores}} em ordem temporal."""
    tr: dict[int, dict] = {}
    for f in sorted(per_frame):
        rows = np.asarray(per_frame[f], float).reshape(-1, 6)
        pts = _points(rows, anchor)
        for r, p in zip(rows, pts):
            t = tr.setdefault(int(r[4]), {"frames": [], "pts": [], "boxes": [], "scores": []})
            t["frames"].append(int(f)); t["pts"].append(p); t["boxes"].append(r[:4]); t["scores"].append(r[5])
    for t in tr.values():
        t["pts"] = np.array(t["pts"]); t["boxes"] = np.array(t["boxes"]); t["scores"] = np.array(t["scores"])
    return tr


def _signed_dist(p, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    n = np.hypot(*d)
    return ((p[:, 0] - a[0]) * d[1] - (p[:, 1] - a[1]) * d[0]) / n, ((p[:, 0] - a[0]) * d[0] + (p[:, 1] - a[1]) * d[1]) / (n * n)


def line_events(traj: dict[int, dict], line: LineSpec, cfg: CountConfig, fps: float) -> list[dict]:
    """Direção 'in' = do lado negativo para o positivo da distância assinada (x−ax)·dy−(y−ay)·dx; para linha vertical de cima (a) para baixo (b) na imagem, é esquerda→direita."""
    ev = []
    for tid, t in traj.items():
        if len(t["frames"]) < cfg.min_track_len:
            continue
        dist, along = _signed_dist(t["pts"], line.a, line.b)
        state, last_idx, last_event = 0, None, -10**9
        for i, d in enumerate(dist):
            if abs(d) <= line.deadband_px:
                continue
            s = 1 if d > 0 else -1
            if state != 0 and s != state:
                # ponto de cruzamento: interpolação entre a última amostra fora da zona morta e a atual
                j = last_idx
                d0, d1 = dist[j], d
                u = d0 / (d0 - d1) if d0 != d1 else 0.5
                p = t["pts"][j] + u * (t["pts"][i] - t["pts"][j])
                al = along[j] + u * (along[i] - along[j])
                inside = -line.margin <= al <= 1 + line.margin
                if inside and t["frames"][i] - last_event >= line.cooldown_frames:
                    ev.append({"type": "line_cross", "line": line.name, "direction": "in" if s > 0 else "out", "track_id": tid, "frame": t["frames"][i],
                               "t_s": t["frames"][i] / fps if fps else None, "x": float(p[0]), "y": float(p[1]), "track_len": len(t["frames"]),
                               "mean_score": float(t["scores"].mean()), "flags": ["trilha_curta"] if len(t["frames"]) < 3 * cfg.min_track_len else []})
                    last_event = t["frames"][i]
            state, last_idx = s, i
    return ev


def region_events(traj: dict[int, dict], reg: RegionSpec, cfg: CountConfig, fps: float) -> list[dict]:
    ev = []
    for tid, t in traj.items():
        if len(t["frames"]) < cfg.min_track_len:
            continue
        inside_raw = [point_in_polygon(tuple(p), reg.polygon) for p in t["pts"]]
        state, run, run_val = None, 0, None
        for i, v in enumerate(inside_raw):
            run = run + 1 if v == run_val else 1
            run_val = v
            if run >= reg.min_frames and v != state:
                if state is not None:
                    ev.append({"type": "region", "region": reg.name, "direction": "enter" if v else "exit", "track_id": tid, "frame": t["frames"][i - reg.min_frames + 1],
                               "t_s": t["frames"][i - reg.min_frames + 1] / fps if fps else None, "x": float(t["pts"][i][0]), "y": float(t["pts"][i][1]),
                               "track_len": len(t["frames"]), "mean_score": float(t["scores"].mean()), "flags": []})
                state = v
    return ev


def region_occupancy(per_frame: dict[int, np.ndarray], reg: RegionSpec, anchor: str = "center") -> dict[int, int]:
    out = {}
    for f, rows in per_frame.items():
        rows = np.asarray(rows, float).reshape(-1, 6)
        out[f] = int(sum(point_in_polygon(tuple(p), reg.polygon) for p in _points(rows, anchor))) if len(rows) else 0
    return out


def run_counting(per_frame: dict[int, np.ndarray], cfg: CountConfig, fps: float, n_det_per_frame: dict[int, int] | None = None, smooth: int = 5) -> dict:
    """Pipeline de contagem. `per_frame`: {frame: (k,6)[x1,y1,x2,y2,id,score]}. Retorna eventos, séries e totais SEPARADOS."""
    traj = build_trajectories(per_frame, cfg.anchor)
    events = []
    for ln in cfg.lines:
        events += line_events(traj, ln, cfg, fps)
    for rg in cfg.regions:
        events += region_events(traj, rg, cfg, fps)
    events.sort(key=lambda e: (e["frame"], e["track_id"]))
    for k, e in enumerate(events):
        e["event_id"] = k + 1
        e["config"] = cfg.digest()
    frames = sorted(per_frame)
    inst = {f: int(len(np.asarray(per_frame[f]).reshape(-1, 6))) for f in frames}   # trilhas ativas por quadro
    det = n_det_per_frame if n_det_per_frame is not None else inst
    series = np.array([det.get(f, 0) for f in frames], float)
    if len(series) >= smooth:
        k = smooth // 2
        sm = np.array([np.median(series[max(0, i - k): i + k + 1]) for i in range(len(series))])
    else:
        sm = series
    totals = {}
    for ln in cfg.lines:
        e = [x for x in events if x["type"] == "line_cross" and x["line"] == ln.name]
        ni, no = sum(x["direction"] == "in" for x in e), sum(x["direction"] == "out" for x in e)
        totals[ln.name] = {"in": ni, "out": no, "net": ni - no, "total_passagens": ni + no}
    for rg in cfg.regions:
        e = [x for x in events if x["type"] == "region" and x["region"] == rg.name]
        occ = region_occupancy(per_frame, rg, cfg.anchor)
        totals[rg.name] = {"enter": sum(x["direction"] == "enter" for x in e), "exit": sum(x["direction"] == "exit" for x in e),
                           "ocupacao_max": max(occ.values()) if occ else 0}
    return {"events": events, "instantaneous": {"frames": frames, "count": series.tolist(), "count_smoothed": sm.tolist()}, "totals": totals,
            "herd_bounds": herd_bounds(sm, traj), "config_digest": cfg.digest()}


def herd_bounds(smoothed_counts, traj: dict[int, dict], min_len: int = 5) -> dict:
    """FAIXAS de referência — NÃO é a população do rebanho.
    • piso observado: maior contagem instantânea suavizada (animais vistos ao mesmo tempo): um rebanho com ocultos/fora do quadro é MAIOR.
    • ids_unicos_filtrados: IDs com ≥ min_len observações. Sobe com troca/fragmentação de ID e falso positivo persistente; desce com animais nunca vistos."""
    sm = np.asarray(smoothed_counts, float)
    return {"piso_observado_simultaneo": int(sm.max()) if len(sm) else 0,
            "ids_unicos_filtrados": int(sum(len(t["frames"]) >= min_len for t in traj.values())),
            "ids_unicos_brutos": len(traj),
            "aviso": "Nenhum destes números é o tamanho do rebanho: animais ocultos ou fora do enquadramento não são contados; "
                     "trocas e fragmentação de ID inflam os IDs únicos."}
