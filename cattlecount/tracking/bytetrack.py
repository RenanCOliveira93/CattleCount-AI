"""ByteTrack (Zhang et al., ECCV 2022) — implementação própria em numpy/scipy.

Ideia: associar em DUAS etapas — (1) detecções de confiança alta às trilhas (IoU); (2) detecções de confiança BAIXA às trilhas
que sobraram, para não perder animais parcialmente ocultos (a confiança cai na oclusão). Trilhas não casadas ficam 'perdidas'
por `track_buffer` quadros e podem ser recuperadas.
Extras opcionais: compensação de movimento de câmera (`update(..., affine=A)`).
Não há reidentificação por aparência aqui (ver `stitch.py` para a costura limitada de tracklets).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..geometry import iou_matrix
from .kalman import KalmanFilterXYAH, xyah_to_xyxy, xyxy_to_xyah
from .motion import apply_affine

NEW, TRACKED, LOST, REMOVED = 0, 1, 2, 3


@dataclass
class TrackerConfig:
    high_thresh: float = 0.5      # detecção "alta"
    low_thresh: float = 0.1       # abaixo disso, ignorada
    new_track_thresh: float = 0.6  # score mínimo para ABRIR trilha nova
    match_thresh: float = 0.8     # custo máx (1−IoU) na 1ª etapa
    low_match_thresh: float = 0.5  # custo máx na 2ª etapa
    unconfirmed_thresh: float = 0.7
    track_buffer: int = 30        # quadros que uma trilha perdida sobrevive
    min_hits: int = 1             # confirmações antes de emitir (1 = comportamento original)


class STrack:
    _next_id = 1

    def __init__(self, xyxy, score, kf: KalmanFilterXYAH):
        self.kf = kf
        self.mean, self.cov = None, None
        self.xyxy0 = np.asarray(xyxy, float)
        self.score = float(score)
        self.state = NEW
        self.activated = False
        self.id = 0
        self.hits = 1
        self.start_frame = self.frame_id = 0
        self.history_scores = [float(score)]

    @property
    def box(self) -> np.ndarray:
        return self.xyxy0 if self.mean is None else xyah_to_xyxy(self.mean[:4])

    def activate(self, frame_id):
        self.id = STrack._next_id; STrack._next_id += 1
        self.mean, self.cov = self.kf.initiate(xyxy_to_xyah(self.xyxy0))
        self.state, self.start_frame, self.frame_id = TRACKED, frame_id, frame_id
        self.activated = frame_id == 1  # no 1º quadro ativa direto (convenção do ByteTrack)

    def predict(self):
        m = self.mean.copy()
        if self.state != TRACKED:
            m[7] = 0.0
        self.mean, self.cov = self.kf.predict(m, self.cov)

    def update(self, det_xyxy, score, frame_id):
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(det_xyxy))
        self.score, self.state, self.frame_id = float(score), TRACKED, frame_id
        self.hits += 1
        self.history_scores.append(float(score))

    def reactivate(self, det_xyxy, score, frame_id):
        self.update(det_xyxy, score, frame_id)
        self.activated = True


def _assign(cost: np.ndarray, thresh: float):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    r, c = linear_sum_assignment(cost)
    m = [(i, j) for i, j in zip(r, c) if cost[i, j] <= thresh]
    mi, mj = {i for i, _ in m}, {j for _, j in m}
    return m, [i for i in range(cost.shape[0]) if i not in mi], [j for j in range(cost.shape[1]) if j not in mj]


class ByteTracker:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.kf = KalmanFilterXYAH()
        self.frame_id = 0
        self.tracked: list[STrack] = []
        self.lost: list[STrack] = []
        self.finished: list[STrack] = []
        STrack._next_id = 1

    def update(self, dets: np.ndarray, affine: np.ndarray | None = None) -> np.ndarray:
        """dets (n,5) [x1,y1,x2,y2,score] → (k,6) [x1,y1,x2,y2,id,score] das trilhas ATIVAS neste quadro."""
        c = self.cfg
        self.frame_id += 1
        dets = np.asarray(dets, float).reshape(-1, 5)
        hi = dets[dets[:, 4] >= c.high_thresh]
        lo = dets[(dets[:, 4] >= c.low_thresh) & (dets[:, 4] < c.high_thresh)]

        confirmed = [t for t in self.tracked if t.activated]
        unconfirmed = [t for t in self.tracked if not t.activated]
        pool = confirmed + self.lost
        for t in pool + unconfirmed:
            t.predict()
            if affine is not None:  # move a previsão para o referencial do quadro atual
                nb = apply_affine(affine, t.box)[0]
                t.mean[:4] = xyxy_to_xyah(nb)

        def cost(tracks, d):
            return 1.0 - iou_matrix(np.array([t.box for t in tracks]).reshape(-1, 4), d[:, :4]) if len(tracks) and len(d) else np.zeros((len(tracks), len(d)))

        activated, refound, new_lost = [], [], []
        m, u_t, u_d = _assign(cost(pool, hi), c.match_thresh)
        for i, j in m:
            t = pool[i]
            if t.state == TRACKED:
                t.update(hi[j, :4], hi[j, 4], self.frame_id); activated.append(t)
            else:
                t.reactivate(hi[j, :4], hi[j, 4], self.frame_id); refound.append(t)

        rem = [pool[i] for i in u_t if pool[i].state == TRACKED]
        m2, u_t2, _ = _assign(cost(rem, lo), c.low_match_thresh)
        for i, j in m2:
            rem[i].update(lo[j, :4], lo[j, 4], self.frame_id); activated.append(rem[i])
        for i in u_t2:
            if rem[i].state != LOST:
                rem[i].state = LOST; new_lost.append(rem[i])

        hi_left = hi[u_d]
        m3, u_unc, u_d3 = _assign(cost(unconfirmed, hi_left), c.unconfirmed_thresh)
        for i, j in m3:
            unconfirmed[i].update(hi_left[j, :4], hi_left[j, 4], self.frame_id)
            if unconfirmed[i].hits >= max(2, c.min_hits):
                unconfirmed[i].activated = True
            activated.append(unconfirmed[i])
        for i in u_unc:
            unconfirmed[i].state = REMOVED
            self.finished.append(unconfirmed[i])

        for j in u_d3:
            if hi_left[j, 4] >= c.new_track_thresh:
                t = STrack(hi_left[j, :4], hi_left[j, 4], self.kf); t.activate(self.frame_id)
                if c.min_hits > 1:
                    t.activated = False
                activated.append(t)

        keep_lost = []
        for t in self.lost:
            if t.state == LOST:
                if self.frame_id - t.frame_id > c.track_buffer:
                    t.state = REMOVED; self.finished.append(t)
                else:
                    keep_lost.append(t)
        ids = {t.id for t in activated + refound}
        self.lost = [t for t in keep_lost if t.id not in ids] + new_lost
        # trilhas rastreadas que ficaram sem atualização já viraram 'lost'; mantém só as atualizadas neste quadro
        self.tracked = [t for t in {t.id: t for t in activated + refound}.values()]
        return np.array([[*t.box, t.id, t.score] for t in self.tracked if t.activated and t.frame_id == self.frame_id]).reshape(-1, 6)

    def all_tracks(self) -> list[STrack]:
        return self.tracked + self.lost + self.finished
