# 12 · Resultados (gerado automaticamente de `experiments/*.json`)

> Este arquivo é produzido por `scripts/gen_results_doc.py`. Nenhum número foi digitado à mão. Interpretação e limites: ver o fim do capítulo.

## 1 · Em uma frase

- **Na fazenda de validação (Mauron, voo não usado no treino)** o modelo escolhido conta razoavelmente: MAE 1,80 animais/imagem, mAP50 0,589.
- **Em fazenda inédita (Jalogny)** ele **superconta muito**: 1724 animais previstos contra 809 reais (**+113 %**), precisão 0,126, mAP50 0,102. **Não use este modelo, como está, numa fazenda nova sem calibrar.**
- O detector COCO pré-treinado (zero-shot) quase não detecta nada nessa escala (revocação 0,033 no teste).

## 2 · Detecção e contagem por imagem — modelos comparados

Limiar de confiança escolhido no VAL (melhor F1) e aplicado ao TESTE sem alteração. Teste = 92 imagens de 5 voos de Jalogny; val = 41 imagens de 1 voo de Mauron. IC95 % por bootstrap **de voos** (só 5 voos no teste: IC largo).

| modelo | conf | split | mAP50 | mAP50-95 | precisão | revocação | MAE contagem | viés | IC95 MAE (voos) |
|---|---|---|---|---|---|---|---|---|---|
| COCO zero-shot (yolo11s) | 0.05 | val | 0,000 | 0,000 | — | 0,000 | 8,71 | -8.71 | — (1 voo) |
| COCO zero-shot (yolo11s) | 0.05 | test | 0,030 | 0,010 | 0,587 | 0,033 | 8,29 | -8.29 | [5,3; 11,7] |
| YOLO11s, parada antecipada (época 16) | 0.3 | val | 0,542 | 0,113 | 0,584 | 0,630 | 2,68 | +0.68 | — (1 voo) |
| YOLO11s, parada antecipada (época 16) | 0.3 | test | 0,103 | 0,022 | 0,108 | 0,337 | 19,11 | +18.65 | [11,8; 27,4] |
| **YOLO11s, 40 épocas (selecionado)** | 0.25 | val | 0,589 | 0,127 | 0,701 | 0,630 | 1,80 | -0.88 | — (1 voo) |
| **YOLO11s, 40 épocas (selecionado)** | 0.25 | test | 0,102 | 0,020 | 0,126 | 0,269 | 11,16 | +9.95 | [7,4; 15,5] |

**Seleção do modelo:** regra aplicada: melhor F1 no val. **Transparência:** o primeiro modelo (parada antecipada) já tinha sido avaliado no teste quando decidi treinar as 40 épocas completas; a decisão de treinar mais não foi cega ao teste, mas a escolha entre os dois modelos usou só o val. F1 no val: 40 épocas 0,664 contra parada antecipada 0,606. O val também escolheu o checkpoint dentro de cada treino (métrica interna do Ultralytics), então os números do val são **um pouco otimistas**.

## 3 · Onde o modelo selecionado erra no teste (fazenda inédita)

| estrato | imagens | GT | previsto | MAE | viés |
|---|---|---|---|---|---|
| voo: DJI_202306231103_031 | 17 | 77 | 162 | 6,76 | +5.00 |
| voo: DJI_202306271323_032 | 23 | 94 | 202 | 6,17 | +4.70 |
| voo: DJI_202309151045_056 | 19 | 214 | 413 | 11,63 | +10.47 |
| voo: DJI_202309261339_058 | 18 | 209 | 486 | 15,39 | +15.39 |
| voo: DJI_202309261339_059 | 15 | 215 | 461 | 18,13 | +16.40 |
| altitude (m): 60 | 23 | 94 | 202 | 6,17 | +4.70 |
| altitude (m): 100 | 69 | 715 | 1522 | 12,83 | +11.70 |
| densidade (animais/imagem): 0 | 20 | 0 | 276 | 13,80 | +13.80 |
| densidade (animais/imagem): 1-5 | 21 | 51 | 311 | 12,67 | +12.38 |
| densidade (animais/imagem): 6-15 | 29 | 245 | 455 | 9,59 | +7.24 |
| densidade (animais/imagem): >=16 | 22 | 513 | 682 | 9,41 | +7.68 |

- **Imagens sem nenhum animal (20):** o modelo previu 276 animais (≈ 13,8 por imagem). São falsos positivos puros; inspeção visual mostra galhos secos e troncos claros em áreas de mata (cenário quase ausente no treino).
- **Altitude:** 60 m tem erro menor que 100 m, mas só **um** voo de teste é de 60 m. Altitude e voo estão **confundidos**: não dá para concluir efeito da altitude.
- **Densidade:** o viés relativo é menor nas cenas densas; o erro absoluto não cresce com a densidade, ou seja, a superestimativa vem de falsos positivos, não de contar demais animais reais.

## 4 · Adaptação a uma fazenda nova (experimento declarado antes de rodar)

Pergunta prática: *quanto ajuda ajustar o modelo com poucas fotos da fazenda do cliente?* Voos de **adaptação**: DJI_202306231103_031 e DJI_202309261339_058 (Jalogny) entram no treino. **Teste da comparação**: os 3 voos restantes (032, 056, 059), nunca vistos. Para a adaptação usa-se o checkpoint final (30 épocas, sem seleção) e conf fixo 0,25, porque o val de Mauron não representa a fazenda nova.

| modelo | imagens | GT | previsto | mAP50 | precisão | revocação | MAE | viés |
|---|---|---|---|---|---|---|---|---|
| parada antecipada | 57 | 523 | 1639 | 0,089 | 0,101 | 0,317 | 19,86 | +19.58 |
| **40 épocas (selecionado)** | 57 | 523 | 1076 | 0,084 | 0,117 | 0,241 | 11,14 | +9.70 |
| **40 épocas + adaptação (2 voos de Jalogny)** | 57 | 523 | 462 | 0,512 | 0,645 | 0,570 | 1,70 | -1.07 |

Por voo retido (40 épocas → adaptado):

| voo | imagens | GT | previsto antes | MAE antes | previsto depois | MAE depois | viés depois |
|---|---|---|---|---|---|---|---|
| DJI_202306271323_032 | 23 | 94 | 202 | 6,17 | 76 | 1,65 | -0.78 |
| DJI_202309151045_056 | 19 | 214 | 413 | 11,63 | 192 | 1,37 | -1.16 |
| DJI_202309261339_059 | 15 | 215 | 461 | 18,13 | 194 | 2,20 | -1.40 |

IC95 % do MAE adaptado (bootstrap de 3 voos): [1,4; 2,2] — **muito estreito para 3 voos; não o interprete como garantia**.

- **Efeito:** nos voos retidos de Jalogny o erro de contagem cai de 11,14 para 1,70 animais/imagem e o viés de +9.70 para -1.07. A melhora aparece nos três voos, inclusive em voos de outro dia (032, a 60 m; 056).
- **Falsos positivos em imagens sem animais:** 7 animais previstos em 12 imagens vazias (antes: 221).
- **Custo (esquecimento):** no val de Mauron o modelo adaptado cai de mAP50 0,589 para 0,368, a revocação cai para 0,300 e ele passa a **subcontar** (viés -4.85 animais/imagem; MAE 4,90). Adaptar a uma fazenda pode **piorar** as outras: o caminho seguro é um modelo por fazenda/domínio, ou re-treinar com todas.
- **Ressalvas:** o voo 059 foi feito no mesmo dia e em sequência ao 058 (usado na adaptação): é quase a mesma cena. Mesmo sem ele, 032 e 056 melhoram. Conf fixo 0,25 e checkpoint final (sem seleção). É **uma** fazenda; o ganho prático depende de haver imagens anotadas da fazenda-alvo (aqui, 2 voos, 35 imagens).

## 5 · Rastreamento

Parâmetros do tracker **não ajustados** nestes vídeos (cenário A: limiares derivados do conf operacional do detector; cenário B: padrões do ByteTrack original).

### A · Vídeo aéreo **sintético** (janela deslizando sobre foto real; 60 quadros, 11 animais; GT exato)

Animais estáticos e câmera sintética: valida a cadeia detector→tracker→contagem com GT exato, **não** o comportamento de bovinos reais em vídeo aéreo.

| método | HOTA | IDF1 | IDF1@IoU0,3 | trocas de ID | IDs previstos | passagens na linha (GT: 3) |
|---|---|---|---|---|---|---|
| ByteTrack próprio | 0,346 | 0,402 | 0,575 | 2 | 7 | 0 |
| ByteTrack próprio + compensação de câmera | 0,346 | 0,402 | 0,575 | 2 | 7 | 0 |
| ByteTrack Ultralytics | 0,314 | 0,382 | 0,466 | 0 | 4 | 0 |
| Baseline IoU | 0,266 | 0,289 | 0,333 | 14 | 38 | 0 |

**Com detecções perfeitas (as caixas do GT como entrada)**, isolando o rastreador do detector:

| método | HOTA | IDF1 | trocas de ID | IDs previstos |
|---|---|---|---|---|
| ByteTrack próprio | 0,976 | 0,999 | 0 | 11 |
| Baseline IoU | 1,000 | 1,000 | 0 | 11 |

O detector aéreo viu em média **4,8 animais por quadro** contra **9,8** reais: a revocação do detector (animais escuros, caixas frouxas) limita tudo a jusante. Com detecções perfeitas o rastreador é quase perfeito; com as reais, as trilhas fragmentam e a contagem de passagens fica abaixo da verdade. (Nota: nesse cenário simples e suave, o baseline IoU também é perfeito com detecção perfeita: a vantagem do ByteTrack aparece quando as detecções são ruidosas.)

### B · Vídeo **terrestre real** (Mendeley, clipe com GT de 9 animais, 479 quadros; detector COCO)

O GT foi gerado com YOLOv8-COCO + DeepSORT (corrigido): **favorece detectores COCO**. É cenário terrestre, não aéreo.

| método | HOTA | IDF1 | MOTA | trocas de ID | IDs previstos |
|---|---|---|---|---|---|
| ByteTrack próprio | 0,727 | 0,880 | 0,841 | 3 | 8 |
| ByteTrack Ultralytics | 0,731 | 0,892 | 0,837 | 2 | 8 |
| Baseline IoU | 0,691 | 0,739 | 0,853 | 22 | 34 |

O ByteTrack próprio e o `BYTETracker` do Ultralytics (mesmas detecções e limiares) ficam praticamente empatados; ambos superam o baseline IoU em IDF1 e em trocas de ID.

## 6 · Latência (CPU, ONNX Runtime, máquina ociosa)

Máquina: macOS-15.8.2-arm64-arm-64bit · CPUExecutionProvider (onnxruntime)

| operação | mediana (ms) | p95 (ms) | n |
|---|---|---|---|
| detecção de 1 tile 1024×1024 | 74.3 | 79.9 | 20 |
| quadro 1280×720 (inteiro) | 67.7 | 84.8 | 10 |
| foto 4000×3000 (fatiada) | 1437.5 | 1501.5 | 5 |
| ByteTrack, 1 quadro, 30 detecções | 0.5 | 0.7 | 300 |

Medido nesta máquina; **não** extrapole para outros hardwares ou para a VM do Podman.

## 7 · O que estes resultados NÃO provam

- O teste tem **5 voos de uma única fazenda**: o IC por voo é largo e não representa outras fazendas.
- Subconjunto de 199 imagens do dataset (download limitado): resultados valem para esse subconjunto.
- **Não há GT de rastreamento em vídeo aéreo real**; o cenário A é sintético e o B é terrestre.
- Anotações do ICAERUS têm lacunas conhecidas; voos inconsistentes foram excluídos, mas ainda pode haver animais sem caixa nos voos usados (isso **subestima** a precisão).
- Um único modelo/tamanho (YOLO11s) e uma semente: sem variância entre execuções.
