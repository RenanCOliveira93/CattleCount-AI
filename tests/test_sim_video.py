import numpy as np

from cattlecount import counting as C
from cattlecount.mot_metrics import evaluate_mot, tracks_to_rows
from cattlecount.sim import noisy_detections, pan_windows, render_pan
from cattlecount.video import annotate_video, events_csv, load_tracks, save_tracks, track_frames, iter_frames, video_info

import cv2


def _scene():
    img = np.random.default_rng(0).integers(0, 255, (400, 1200, 3), dtype=np.uint8)
    boxes = np.array([[150 + 100 * i, 150 + 20 * (i % 3), 190 + 100 * i, 190 + 20 * (i % 3)] for i in range(10)], float)
    offs = pan_windows((400, 1200), (400, 300), 60, (0, 50), (800, 50))
    return render_pan(img, boxes, (400, 300), offs)


def test_pan_gt_visibilidade_e_ids_estaveis():
    frames, gt = _scene()
    assert len(frames) == 60 and frames[0].shape == (300, 400, 3)
    assert set(gt[:, 1]) <= set(range(1, 11)) and (gt[:, 2:] >= 0).all() and (gt[:, [4]] <= 400).all()


def test_pipeline_com_deteccoes_simuladas_e_contagem_de_linha(tmp_path):
    frames, gt = _scene()
    dets = noisy_detections(gt, 60, seed=1, recall=0.95, fp_per_frame=0.0, win_wh=(400, 300))
    it = iter(range(1, 61))
    res = track_frames(((i, f) for i, f in enumerate(frames)), lambda rgb, it=it: dets[next(it)])
    m = evaluate_mot(np.column_stack([gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3], gt[:, 4] - gt[:, 2], gt[:, 5] - gt[:, 3]]), tracks_to_rows(res["tracks"]))
    assert m["IDF1"] > 0.6
    # persistência: salvar/carregar e recontar sem detecção
    save_tracks(tmp_path / "t.csv", res)
    back = load_tracks(tmp_path / "t.csv", res["n_det"])
    cfg = C.CountConfig([C.LineSpec((200, 0), (200, 300), "L")])
    a, b = C.run_counting(res["tracks"], cfg, 10), C.run_counting(back["tracks"], cfg, 10)
    assert a["totals"] == b["totals"]
    events_csv(a["events"], tmp_path / "e.csv")
    assert (tmp_path / "e.csv").read_text().splitlines()[0].startswith("event_id,type")


def test_video_roundtrip_e_anotado(tmp_path):
    frames, gt = _scene()
    p = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (400, 300))
    for f in frames:
        w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    w.release()
    info = video_info(str(p))
    assert info["width"] == 400 and info["n_frames"] == 60
    dets = noisy_detections(gt, 60, seed=2, recall=1.0, fp_per_frame=0.0, win_wh=(400, 300))
    it = iter(range(1, 61))
    res = track_frames(iter_frames(str(p)), lambda rgb: dets[next(it)])
    cfg = C.CountConfig([C.LineSpec((200, 0), (200, 300), "L")])
    cnt = C.run_counting(res["tracks"], cfg, info["fps"])
    annotate_video(str(p), str(tmp_path / "a.mp4"), res, cfg, cnt)
    assert video_info(str(tmp_path / "a.mp4"))["n_frames"] == 60


def test_n_det_respeita_count_conf():
    frames = [np.zeros((50, 50, 3), np.uint8)] * 3
    dets = np.array([[0, 0, 10, 10, 0.9], [20, 20, 30, 30, 0.15]])
    todos = track_frames(((i, f) for i, f in enumerate(frames)), lambda rgb: dets)
    acima = track_frames(((i, f) for i, f in enumerate(frames)), lambda rgb: dets, count_conf=0.25)
    assert list(todos["n_det"].values()) == [2, 2, 2] and list(acima["n_det"].values()) == [1, 1, 1]
