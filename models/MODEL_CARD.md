# Model card — `cattle_yolo11s_1024`

- **Arquitetura**: YOLO11s (Ultralytics), 1 classe (`cow`), ajustado a partir de pesos COCO; exportado a ONNX 1024×1024 (opset 17). Paridade PyTorch×ONNX verificada (IoU 0,999999; diferença de score ~1e‑6).
- **Treino**: tiles nativos 1024 px (20 % de sobreposição) de 6 voos (Mauron ×5, Derval ×1) do ICAERUS (Zenodo 10.5281/zenodo.11048412, CC BY 4.0); 346 tiles (164 com gado; 1 437 caixas). 40 épocas, MPS (Apple M4 Max), semente 0, `imgsz=1024`, `batch=8`, aumentos flipud/fliplr/rotação 90°/mosaic/escala 0,3.
- **Seleção**: melhor F1 no val (voo Mauron 028) entre dois checkpoints; limiar de confiança 0,25 escolhido no val.
- **Uso pretendido**: contar animais **visíveis** em fotos aéreas nadir de pastagem a ~60–100 m, com resolução de ~46 px por animal; apoio a inspeção manual.
- **Fora do escopo**: censo de rebanho; outras altitudes/ângulos; terreno com mata densa; decisões sanitárias, comerciais ou regulatórias.
- **Desempenho medido** (ver `docs/GUIA_TECNICO.md (seção 8)`, `experiments/registry.json`): val (Mauron) mAP50 0,589, MAE 1,80; **teste em fazenda inédita (Jalogny) mAP50 0,102, precisão 0,126, superestima em +113 %**. Falsos positivos em mata (galhos secos) foram observados.
- **Limitações**: animais escuros/ocultos perdidos; subconjunto de 199 imagens; teste de uma fazenda; sem avaliação por iluminação ou oclusão.
- **Licença**: **AGPL‑3.0** (pesos/código Ultralytics). Os arquivos `.pt`/`.onnx` não são versionados; reproduza com `training/` ou obtenha com o autor.
- **Reprodução**: `training/build_tiles.py` → `training/train_yolo.py yolo11s yolo11s_cattle_full 40 100` → `training/predict_split.py` → `training/eval_predictions.py`.

## Variante adaptada (`cattle_yolo11s_1024_jalogny_adapt`, não é o padrão)
Modelo acima ajustado com tiles de 2 voos de Jalogny (031, 058), 30 épocas, lr0 0,002. Nos 3 voos retidos de Jalogny: MAE de contagem 1,70 (contra 11,14). **Piora e subconta em Mauron** (mAP50 0,589 → 0,368). Usar somente na fazenda de adaptação; números em `experiments/registry.json`.
