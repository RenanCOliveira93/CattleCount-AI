import numpy as np
import pytest

from cattlecount import counting as C
from cattlecount.count_metrics import bootstrap_ci, count_errors, density_bins
from cattlecount.tracking.stitch import StitchConfig, apply_mapping, stitch


def _walk(tid, frames, p0, vel, size=20, score=0.9):
    return {f: np.array([[*(np.add(p0, np.multiply(vel, f - frames[0])) - size / 2), *(np.add(p0, np.multiply(vel, f - frames[0])) + size / 2), tid, score]]) for f in frames}


def _merge(*ds):
    out = {}
    for d in ds:
        for f, r in d.items():
            out[f] = np.vstack([out[f], r]) if f in out else r
    return out


LINE = C.LineSpec((100, 0), (100, 200), "L")


def test_cruzamento_unico_e_direcao():
    cfg = C.CountConfig([LINE])
    res = C.run_counting(_walk(1, range(30), (50, 100), (4, 0)), cfg, 10)            # esquerda → direita
    res2 = C.run_counting(_walk(1, range(30), (150, 100), (-4, 0)), cfg, 10)         # direita → esquerda
    t1, t2 = res["totals"]["L"], res2["totals"]["L"]
    assert (t1["in"] + t1["out"], t2["in"] + t2["out"]) == (1, 1) and t1["net"] == -t2["net"] != 0
    assert res["events"][0]["event_id"] == 1 and res["events"][0]["config"] == cfg.digest()


def test_jitter_na_linha_nao_conta_varias_vezes():
    rng = np.random.default_rng(0)
    pf = {f: np.array([[*(100 + rng.normal(0, 2) - 10, 100 - 10), *(100 + rng.normal(0, 2) + 10, 110), 1, 0.9]]) for f in range(40)}
    assert C.run_counting(pf, C.CountConfig([LINE]), 10)["totals"]["L"]["total_passagens"] == 0   # oscila dentro da zona morta


def test_passar_pela_ponta_do_segmento_nao_conta():
    cfg = C.CountConfig([C.LineSpec((100, 0), (100, 200), "L")])
    res = C.run_counting(_walk(1, range(30), (50, 300), (4, 0)), cfg, 10)            # cruza x=100 mas em y=300 (fora do segmento)
    assert res["totals"]["L"]["total_passagens"] == 0


def test_trilha_curta_filtrada_e_ida_e_volta_conta_duas():
    cfg = C.CountConfig([LINE], min_track_len=3)
    assert C.run_counting(_walk(1, range(2), (95, 100), (10, 0)), cfg, 10)["totals"]["L"]["total_passagens"] == 0
    f = {k: np.array([[*(np.array([50 + 6 * min(k, 40 - k) * 2, 100]) - 10), *(np.array([50 + 6 * min(k, 40 - k) * 2, 100]) + 10), 1, 0.9]]) for k in range(41)}
    t = C.run_counting(f, cfg, 10)["totals"]["L"]
    assert t["in"] == 1 and t["out"] == 1 and t["net"] == 0


def test_regiao_entrada_saida_com_debounce():
    reg = C.RegionSpec([(100, 0), (200, 0), (200, 200), (100, 200)], "R", min_frames=3)
    res = C.run_counting(_walk(1, range(60), (50, 100), (5, 0)), C.CountConfig(regions=[reg]), 10)
    assert res["totals"]["R"]["enter"] == 1 and res["totals"]["R"]["exit"] == 1 and res["totals"]["R"]["ocupacao_max"] == 1


def test_tres_contagens_sao_separadas_e_ids_nao_sao_populacao():
    pf = _merge(_walk(1, range(0, 30), (50, 60), (4, 0)), _walk(2, range(0, 30), (50, 140), (4, 0)), _walk(3, range(10, 40), (30, 100), (4, 0)))
    res = C.run_counting(pf, C.CountConfig([LINE]), 10)
    hb = res["herd_bounds"]
    assert max(res["instantaneous"]["count"]) == 3 and hb["piso_observado_simultaneo"] <= 3 and "aviso" in hb and "tamanho do rebanho" in hb["aviso"]
    assert res["totals"]["L"]["in"] == 3 and "population" not in res and "populacao" not in res


def test_costura_une_apos_oclusao_mas_nao_une_animais_distantes():
    a = _walk(1, range(0, 20), (50, 100), (4, 0))
    b = _walk(7, range(28, 50), (50 + 4 * 28, 100), (4, 0))                              # mesmo animal reaparece após 8 quadros
    c = _walk(9, range(28, 50), (400, 300), (4, 0))                                       # outro animal, longe
    m = stitch({**{k: C.build_trajectories(_merge(a, b, c))[k] for k in (1, 7, 9)}}, StitchConfig())
    assert m[7] == m[1] and m[9] == 9
    out = apply_mapping(_merge(a, b), m)
    assert len({int(r[0, 4]) for r in out.values()}) == 1


def test_costura_respeita_gap_e_tamanho():
    a = _walk(1, range(0, 10), (50, 100), (0, 0), size=20)
    far_in_time = _walk(2, range(80, 90), (50, 100), (0, 0), size=20)
    other_size = _walk(3, range(15, 25), (50, 100), (0, 0), size=60)
    tr = C.build_trajectories(_merge(a, far_in_time, other_size))
    m = stitch(tr, StitchConfig(max_gap=45))
    assert m == {1: 1, 2: 2, 3: 3}


def test_metricas_de_contagem():
    r = count_errors([10, 12, 0, 5], [10, 10, 0, 8])
    assert r["mae"] == pytest.approx(1.25) and r["bias"] == pytest.approx(-0.25) and r["rmse"] == pytest.approx(np.sqrt((0 + 4 + 0 + 9) / 4))
    ci = bootstrap_ci([10, 12, 0, 5, 7, 7], [10, 10, 0, 8, 7, 9], n_boot=200, groups=[1, 1, 2, 2, 3, 3])
    assert ci["unidade_bootstrap"] == "grupo" and ci["mae"][0] <= ci["mae"][1]
    assert density_bins([0, 3, 8, 40]) == ["0", "1-5", "6-15", ">=16"]
