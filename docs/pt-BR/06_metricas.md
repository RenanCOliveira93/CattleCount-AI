# 06 · Métricas

## Detecção (`cattlecount/det_metrics.py`)
- **AP** com interpolação de 101 pontos (estilo COCO); **AP50** (IoU ≥ 0,5), **AP50‑95** (média de 0,50:0,05:0,95), **AP75**. Casamento guloso por score; ≤ 100 detecções/imagem.
- **Precisão/Revocação/F1** a um limiar de confiança fixo, IoU 0,5.
- Implementação própria, conferida contra `pycocotools` (projeto 04) e por testes.
- **Objetos minúsculos**: AP50‑95 é severa (um erro de 3 px já derruba o IoU de uma caixa de 43 px). Por isso AP50 e AP50‑95 divergem bastante; olhe os dois.

## Contagem (`cattlecount/count_metrics.py`)
Por imagem: **MAE**, **RMSE**, **viés** (pred − GT; positivo = superconta), **MAPE** (só imagens com GT > 0), erro % do total. **IC 95 % por bootstrap reamostrando VOOS inteiros** (imagens do mesmo voo são correlacionadas; reamostrar imagens subestimaria a incerteza). Estratos: por voo, altitude, densidade (0 | 1‑5 | 6‑15 | ≥ 16), fazenda.

## Rastreamento (`cattlecount/mot_metrics.py`, via TrackEval oficial)
- **HOTA** (DetA × AssA), **IDF1**, **MOTA**, **IDSW** (trocas de ID), FP, FN, MT/PT/ML, fragmentações.
- Conferidas contra `motmetrics` em testes (IDF1/MOTA/IDSW).
- Para objetos minúsculos reporta‑se também **IDF1 a IoU 0,3**.

## Contagem de passagens
Erro absoluto de passagens por linha contra a **passagem calculada sobre o GT** (as trilhas verdadeiras passadas pelo **mesmo** contador). Isto mede o erro do detector + rastreador, não da regra de contagem.
