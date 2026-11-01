import numpy as np
import pytest

from cattlecount import geometry as G, tiling as T


def test_segmentos_e_poligono():
    assert G.segments_intersect((0, 0), (10, 10), (0, 10), (10, 0))
    assert not G.segments_intersect((0, 0), (1, 1), (5, 0), (5, 10))
    assert not G.segments_intersect((0, 0), (10, 0), (0, 5), (10, 5))          # paralelos
    assert not G.segments_intersect((0, 0), (4, 4), (5, 0), (5, 10))           # reta infinita cruzaria; segmento não
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert G.point_in_polygon((5, 5), sq) and G.point_in_polygon((0, 5), sq) and not G.point_in_polygon((11, 5), sq)
    assert G.iou_matrix([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0] == pytest.approx(1 / 3)


@pytest.mark.parametrize("h,w", [(3956, 5280), (3000, 4000), (800, 900), (1024, 1024), (500, 3000)])
def test_tile_grid_cobre_a_imagem_sem_buracos(h, w):
    g = T.tile_grid(h, w, 1024, 0.2)
    cov = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in g:
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and (x1 - x0 == 1024 or x1 - x0 == w) and (y1 - y0 == 1024 or y1 - y0 == h)
        cov[y0:y1, x0:x1] = True
    assert cov.all()


def test_sliced_predict_conta_cada_objeto_uma_vez_inclusive_na_emenda():
    """Detector 'perfeito' que devolve os objetos de GT visíveis no tile (incluindo cortados). Objetos perto da emenda não podem duplicar nem sumir."""
    h, w = 2000, 3000
    gt = np.array([[900, 100, 950, 150], [1000, 500, 1050, 560], [1010, 1300, 1060, 1350], [2900, 1900, 2950, 1960], [2000, 1000, 2050, 1050]], float)
    img = np.zeros((h, w, 3), np.uint8)

    def detect(tiles_):
        out = []
        # reconstrói a posição de cada tile procurando-a na ordem da grade
        return out
    grid = T.tile_grid(h, w, 1024, 0.2)
    it = iter(grid)

    def detect_tiles(tiles_):
        res = []
        for _ in tiles_:
            x0, y0, x1, y1 = next(it)
            d = []
            for b in gt:
                bb = np.array([max(b[0], x0), max(b[1], y0), min(b[2], x1), min(b[3], y1)])
                if bb[2] - bb[0] > 5 and bb[3] - bb[1] > 5:
                    d.append([*(bb - [x0, y0, x0, y0]), 0.9])
            res.append(np.array(d).reshape(-1, 5))
        return res
    out = T.sliced_predict(detect_tiles, img, 1024, 0.2)
    assert len(out) == len(gt)
    ious = G.iou_matrix(out[:, :4], gt)
    assert (ious.max(1) > 0.95).all() and (ious.max(0) > 0.95).all()
