import numpy as np

from cattlecount.tracking import ByteTracker, TrackerConfig
from cattlecount.tracking.kalman import KalmanFilterXYAH, xyah_to_xyxy, xyxy_to_xyah


def _scene(n_frames=60, n_obj=3, seed=0):
    rng = np.random.default_rng(seed)
    start = np.array([[100 + 150 * i, 100 + 60 * i] for i in range(n_obj)], float)
    vel = rng.uniform(-3, 3, (n_obj, 2)) + np.array([4, 0])
    return [np.array([[*(s + v * t), *(s + v * t + 50)] for s, v in zip(start, vel)]) for t in range(n_frames)]


def test_kalman_roundtrip_e_convergencia():
    b = np.array([10, 20, 50, 80.])
    assert np.allclose(xyah_to_xyxy(xyxy_to_xyah(b)), b)
    kf = KalmanFilterXYAH(); mean, cov = kf.initiate(xyxy_to_xyah(b))
    for t in range(1, 15):
        mean, cov = kf.predict(mean, cov); mean, cov = kf.update(mean, cov, xyxy_to_xyah(b + [5 * t, 0, 5 * t, 0]))
    assert abs(mean[4] - 5) < 1.0  # velocidade em x ≈ 5 px/quadro


def test_ids_persistentes_sem_ruido():
    tr = ByteTracker(); seen = {}
    for t, boxes in enumerate(_scene()):
        out = tr.update(np.c_[boxes, np.full(len(boxes), 0.9)])
        for row in out:
            k = int(np.argmin(np.abs(boxes[:, 0] - row[0]) + np.abs(boxes[:, 1] - row[1])))
            seen.setdefault(k, set()).add(int(row[4]))
    assert all(len(v) == 1 for v in seen.values()) and len(seen) == 3


def test_segunda_etapa_mantem_id_com_confianca_baixa():
    """Na oclusão a confiança cai (0,3): sem a 2ª etapa a trilha seria perdida/ID trocado; ByteTrack mantém."""
    tr = ByteTracker(TrackerConfig(high_thresh=0.5, new_track_thresh=0.6)); ids = []
    for t, boxes in enumerate(_scene(40, 1)):
        s = 0.3 if 15 <= t < 25 else 0.9
        out = tr.update(np.c_[boxes, [s]]); ids += [int(r[4]) for r in out]
    assert len(set(ids)) == 1 and len(ids) >= 30   # ausentes só durante a oclusão (saída = apenas trilhas atualizadas c/ conf. baixa também contam)


def test_oclusao_total_curta_recupera_mesmo_id_e_longa_gera_novo():
    def run(gap):
        tr = ByteTracker(TrackerConfig(track_buffer=30)); ids = []
        for t, boxes in enumerate(_scene(80, 1)):
            d = np.zeros((0, 5)) if 20 <= t < 20 + gap else np.c_[boxes, [0.9]]
            ids += [int(r[4]) for r in tr.update(d)]
        return set(ids)
    assert len(run(8)) == 1          # dentro do buffer: mesmo ID
    assert len(run(50)) == 2         # além do buffer: ID novo (fragmentação!)


def test_falsos_positivos_isolados_nao_abrem_trilha_com_confianca_baixa():
    tr = ByteTracker(); n = 0
    rng = np.random.default_rng(1)
    for t in range(30):
        fp = np.array([[*rng.uniform(0, 500, 2), 0, 0, 0.3]]); fp[0, 2:4] = fp[0, :2] + 40
        n += len(tr.update(fp))
    assert n == 0
