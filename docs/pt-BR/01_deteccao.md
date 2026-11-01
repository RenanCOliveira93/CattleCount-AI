# 01 · Detecção de bovinos

## Objetos minúsculos
No dataset ICAERUS a **mediana da área de uma caixa é 0,015 % da imagem** (≈ 46 px de lado (mediana medida, todas as imagens rotuladas); ≈ 7 px se a imagem for reduzida a 640 px). Redimensionar a foto inteira para 640/1024 px apaga o animal. Por isso:

- **Treino**: tiles de **1024 px na resolução nativa**, 20 % de sobreposição (`training/build_tiles.py`). Caixas entram no tile se ≥ 50 % da sua área estiver nele (o resto é cortado). Tiles sem gado são mantidos com probabilidade 0,15 (negativos).
- **Inferência (imagens grandes)**: **inferência fatiada** (`cattlecount/tiling.py::sliced_predict`): cada tile é detectado; detecções **encostadas numa borda interna do tile são descartadas** (o animal cortado é visto inteiro no tile vizinho, graças à sobreposição); a união é filtrada por NMS com IoU e IoS (interseção sobre a menor área), para não duplicar animais na faixa de sobreposição.
- Imagens/quadros com lado > 1280 px são fatiados; quadros de vídeo até 1280 px são processados inteiros (reduzidos a 1024 px). Testei fatiar também o quadro 1280×720 (escala 1,0× como no treino): **não melhorou** (3,9 contra 4,8 detecções/quadro no vídeo sintético) e dobra o custo, então a regra ficou em 1280 px.

## Modelo
YOLO11 (Ultralytics) ajustado a partir de pesos COCO (`yolo11s`), exportado a **ONNX** (1024×1024) para a API, que roda com ONNX Runtime **sem torch e sem ultralytics**. A paridade PyTorch × ONNX foi verificada (IoU 1,0 e diferença de score ~1e‑6 num tile real).

> **Licença**: pesos e código Ultralytics são **AGPL‑3.0**. O rastreador (ByteTrack) e todo o código de contagem são próprios (sem AGPL). Uso comercial dos pesos treinados exige avaliar a AGPL (ou licença comercial da Ultralytics).

## Escolha do limiar
O limiar operacional é escolhido **no conjunto de validação** (melhor F1) e aplicado **sem alteração** ao teste. Para vídeo, os limiares do ByteTrack são derivados dele (alto = limiar; baixo = 0,1; abertura de trilha = limiar + 0,1), não ajustados em cada vídeo.
