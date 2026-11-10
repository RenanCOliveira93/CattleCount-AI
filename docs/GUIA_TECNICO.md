# Guia técnico — CattleCount AI

Documentação de engenharia do projeto: arquitetura, dados, instalação, execução, metodologia, API, validação, limitações e licenças.

## Sumário

1. [Visão geral](#1-visão-geral)
2. [Arquitetura](#2-arquitetura)
3. [Instalação e execução](#3-instalação-e-execução)
4. [Dados e proveniência](#4-dados-e-proveniência)
5. [Metodologia](#5-metodologia)
6. [API](#6-api)
7. [Métricas e validação](#7-métricas-e-validação)
8. [Resultados](#8-resultados)
9. [Mudança de domínio](#9-mudança-de-domínio)
10. [Fontes de erro de contagem](#10-fontes-de-erro-de-contagem)
11. [Limitações conhecidas](#11-limitações-conhecidas)
12. [Licenças e atribuições](#12-licenças-e-atribuições)
13. [Solução de problemas](#13-solução-de-problemas)

---

## 1. Visão geral

O CattleCount AI detecta bovinos em imagens aéreas e conta animais em vídeo. O sistema distingue três grandezas, que não devem ser confundidas:

| Grandeza | Pergunta | Como é obtida | O que não é |
|---|---|---|---|
| **Visíveis agora** (contagem instantânea) | Quantos animais aparecem neste quadro/imagem? | Número de detecções com score >= limiar escolhido | Não inclui animais ocultos (árvores, sombra, oclusão) nem fora do enquadramento |
| **Passagens** (contagem acumulada) | Quantas vezes animais cruzaram a linha ou entraram/saíram da região? | Eventos sobre trilhas ByteTrack (cruzamento com zona morta, direção, debounce) | Não é número de animais: o mesmo animal pode cruzar várias vezes e uma troca de ID pode gerar passagem a mais ou a menos |
| **Referência de rebanho** | Qual é o tamanho do rebanho? | Não é calculada como número único. Apenas faixas: piso observado simultâneo e IDs únicos filtrados/brutos | Nunca é população. IDs únicos inflam com trocas/fragmentação e deflacionam com animais nunca vistos |

Modos de operação:

1. **Imagem** (`POST /infer/image`): conta animais visíveis. Imagens com lado maior que 1280 px são analisadas por fatias de 1024 px na resolução nativa.
2. **Vídeo** (`POST /videos`, assíncrono): detecção por quadro, ByteTrack, trilhas persistidas e contagem de passagens por linha ou região. As trilhas ficam salvas; é possível alterar a linha e recontar sem refazer a detecção (`POST /analyses/{id}/count`). Cada recontagem é registrada para auditoria.

O detector foi avaliado em fazenda inédita, com queda acentuada de desempenho (seção 8). O rastreador foi validado em vídeo terrestre real e em vídeo aéreo sintético; não existe verdade de rastreamento em vídeo aéreo real neste projeto.

## 2. Arquitetura

```
 navegador --> nginx (frontend estático + proxy /api) --> FastAPI --> PostgreSQL (análises, recontagens, eventos)
                                                            |
                                                            +- ONNX Runtime (detector, 1024 px, fatias)
                                                            +- ByteTrack próprio + contagem (cattlecount/)
                                                            +- volume de jobs (vídeo, tracks.csv, vídeo anotado)
```

| Diretório | Conteúdo |
|---|---|
| `backend/` | API FastAPI, SQLAlchemy, serviço de análise, detector ONNX |
| `frontend/` | React/TypeScript (upload, linha/região, trilhas, gráficos, CSV, auditoria) |
| `cattlecount/` | Núcleo: tiling, ByteTrack, contagem, métricas, ingestão e validação de entrada |
| `training/` | Dados, tiles, treino e avaliação |
| `scripts/` | Benchmark, registro de modelos, geração das tabelas de resultados |
| `experiments/` | JSONs reais de avaliação e `registry.json` agregado |
| `models/` | Model card e registro (pesos não versionados) |
| `notebooks/` | Notebooks executados de dados, resultados e rastreamento |
| `infra/`, `docker-compose.yml` | Dockerfiles, nginx e orquestração |

Características do backend:

- **Assíncrono**: `POST /videos` valida e devolve `202` com `analysis_id`; um worker (1 thread) processa; o progresso é obtido por `GET /analyses/{id}` (polling) e por WebSocket em `/ws`.
- **Validação de entrada** (`cattlecount/ingest.py`): formato, corrupção e limites de tamanho; avisos explícitos (altitude ausente, FPS baixo, resolução baixa). Nada é corrigido silenciosamente.
- **Inferência**: ONNX Runtime, sem torch e sem ultralytics em tempo de execução.
- **Autenticação**: opcional por chave (`API_KEY` exige o cabeçalho `X-API-Key` nas rotas que gravam). Não há usuários nem papéis.

## 3. Instalação e execução

### 3.1 Contêineres (Docker ou Podman)

```bash
cp .env.example .env            # defina POSTGRES_PASSWORD
docker compose up --build       # ou: podman compose up --build
# http://localhost:8080
```

O `docker-compose.yml` usa imagens com nome totalmente qualificado (`docker.io/...`) e nenhum recurso exclusivo do Docker. Os alvos `make up` / `make down` usam `podman compose`.

Os pesos ONNX (`models/onnx/cattle_yolo11s_1024.onnx`) não são versionados (ver seção 12 e `models/MODEL_CARD.md`). Sem eles a API sobe, mas a inferência exige gerá-los (seção 3.4) ou obtê-los conforme o model card.

Variáveis de ambiente (`.env.example`): `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `API_KEY` (opcional), `DETECT_MODE` (`sliced` ou `full`), `DEFAULT_CONF`, `MAX_UPLOAD_MB`, `CORS_ORIGINS`, `HTTP_PORT`. O arquivo `.env` real não é versionado.

### 3.2 Execução local (sem contêiner)

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
.venv/bin/python -m uvicorn backend.app.asgi:app --port 8000      # SQLite por padrão
cd frontend && npm ci && npm run dev                                # http://localhost:5173 (proxy /api -> :8000)
```

### 3.3 Testes

```bash
.venv/bin/python -m pytest                       # unidade + API + PostgreSQL (pgserver) + modelo ONNX real
.venv/bin/python -m pytest tests/e2e -m e2e      # navegador contra a pilha em http://localhost:8080
cd frontend && npx vitest run && npx tsc --noEmit
```

Equivalente: `make test` e `make e2e` (este requer a pilha no ar via `make up`). `tests/test_real_model.py` requer o modelo ONNX presente localmente.

### 3.4 Reprodução do treino e dos experimentos

`make setup` cria o ambiente com as dependências de treino. Ordem dos scripts em `training/`:

`zenodo_plan.py` -> `zenodo_download_images.py` -> `build_tiles.py` -> `train_yolo.py` -> `predict_split.py` -> `eval_predictions.py` -> `eval_tracking.py`

| Alvo `make` | Ação |
|---|---|
| `data` | Plano de download, imagens e tiles (requer rede; cerca de 2,3 GB, lento) |
| `train` | `train_yolo.py yolo11s yolo11s_cattle_full 40 100` |
| `eval` | Predições fatiadas em val e teste, mais métricas |
| `track` | `training/eval_tracking.py` |
| `bench` | `scripts/benchmark.py` (executar com a máquina ociosa) |
| `registry` | `scripts/build_registry.py` |
| `results` | `scripts/gen_results_doc.py` (tabelas da seção 8, escritas em `runs/results.md`) |
| `notebooks` | Reconstrói os três notebooks a partir de `notebooks/build_*.py` |

Os resultados ficam em `experiments/*.json` e são agregados em `experiments/registry.json`.

## 4. Dados e proveniência

### 4.1 Critérios de validação dos datasets

Licença, tipo de anotação, diversidade de cenários, resolução, tamanho dos objetos, condições de aquisição e consistência das anotações com o que o autor declara.

### 4.2 Dataset principal: ICAERUS "aerial cows" (Zenodo)

- **Registro**: Zenodo 11048412 (doi:10.5281/zenodo.11048412), licença CC BY 4.0, 16,6 GB. Fotos de drone (DJI Mavic 3E/3T), voo nadir a 60 e 100 m, três fazendas (Mauron, Derval, Jalogny) e outras sem rótulo; caixas em formato YOLO.
- **Inconsistências na versão baixada**: o resumo oficial declara 4 941 caixas, mas os arquivos de rótulo contêm 2 666; 306 imagens não têm arquivo de rótulo; 13 voos divergem do resumo. Arquivo de rótulo ausente não significa ausência de gado.
- **Regra adotada**: um voo é considerado confiável apenas se o número de caixas nos rótulos coincide com o do resumo oficial (±1). Voos não confiáveis foram excluídos de avaliação e treino (não foram usados como negativos).
- **Unidade de amostragem**: imagens do mesmo voo são quase duplicadas. A divisão é por voo (e por fazenda no teste), nunca por imagem.
- **Divisão definida antes de qualquer resultado de modelo**:
  - Treino: voos DJI_202306051814_022, 202306051839_025, 202306071703_029, 202306071745_034, 202306071745_035 (Mauron) e DJI_202308091442_012 (Derval).
  - Validação: DJI_202306071624_028 (Mauron).
  - Teste (fazenda inédita): voos confiáveis de Jalogny (outro local, outro drone/câmera, cor e vegetação diferentes).
- **Subconjunto baixado**: o servidor entrega 0,2–0,6 MB/s, então foram baixadas 199 imagens (2,26 GB) escolhidas por plano reproduzível (`training/zenodo_plan.py`, semente 0): imagens com gado primeiro e até 4 imagens sem gado por voo. Os resultados valem para esse subconjunto.

### 4.3 Dataset de rastreamento: Mendeley Data dk54zg67dd

Vídeos de celular ao nível do solo, CC BY 4.0, com verdade MOT para um único clipe (cerca de 16 s, 479 quadros; o arquivo de GT contém 9 identidades, enquanto a descrição do autor cita 8 animais). O autor descreve que o GT partiu de YOLOv8-COCO + DeepSORT e foi corrigido, o que favorece detectores tipo COCO. Uso restrito à validação do rastreador em movimento real de bovinos; não é cenário aéreo e não valida o detector aéreo.

### 4.4 Fontes não utilizadas

- Roboflow Universe: nenhum dataset foi usado; licença e qualidade de anotação variam por projeto e exigiriam validação individual.
- Não existe verdade de rastreamento em vídeo aéreo real.

### 4.5 Amostras

`sample_data/` contém uma foto aérea real recortada (`icaerus_mauron_crop.jpg`) e um vídeo sintético (`demo_pan_synthetic.mp4`, com `demo_pan_synthetic_gt.txt`). Atribuição em `sample_data/ATTRIBUTION.md`.

## 5. Metodologia

### 5.1 Detecção

Na ICAERUS, a mediana da área de uma caixa é 0,015 % da imagem (cerca de 46 px de lado; cerca de 7 px se a imagem for reduzida a 640 px). Redimensionar a foto inteira apaga o animal. Por isso:

- **Treino**: tiles de 1024 px na resolução nativa com 20 % de sobreposição (`training/build_tiles.py`). Caixas entram no tile se pelo menos 50 % da área estiver nele. Tiles sem gado são mantidos com probabilidade 0,15.
- **Inferência em imagens grandes** (`cattlecount/tiling.py::sliced_predict`): cada tile é detectado; detecções encostadas numa borda interna do tile são descartadas (o animal cortado é visto inteiro no tile vizinho); a união é filtrada por NMS com IoU e IoS (interseção sobre a menor área).
- Imagens/quadros com lado maior que 1280 px são fatiados; quadros de vídeo até 1280 px são processados inteiros (reduzidos a 1024 px). Fatiar também o quadro 1280x720 não melhorou (3,9 contra 4,8 detecções/quadro no vídeo sintético) e dobra o custo.
- **Modelo**: YOLO11 (Ultralytics) ajustado a partir de pesos COCO (`yolo11s`), exportado para ONNX (1024x1024). A paridade PyTorch x ONNX foi verificada num tile real (IoU 1,0; diferença de score da ordem de 1e-6).
- **Limiar**: escolhido no conjunto de validação (melhor F1) e aplicado sem alteração ao teste. Para vídeo, os limiares do ByteTrack derivam dele (alto = limiar; baixo = 0,1; abertura de trilha = limiar + 0,1).

### 5.2 Rastreamento (ByteTrack)

Implementação própria em `cattlecount/tracking/` do ByteTrack (Zhang et al., ECCV 2022):

1. Filtro de Kalman (estado: centro, razão de aspecto, altura e velocidades) prevê a posição de cada trilha.
2. Associação em duas etapas (Húngaro sobre custo 1 − IoU): detecções de confiança alta contra trilhas ativas e perdidas; depois detecções de confiança baixa contra as trilhas restantes, o que recupera animais parcialmente ocultos.
3. Trilhas novas só nascem de detecção com score >= `new_track_thresh`; trilhas sem associação permanecem perdidas por `track_buffer` quadros.
4. Compensação opcional de movimento de câmera: fluxo óptico Lucas-Kanade + RANSAC (afim parcial) aplicado às previsões do Kalman. Não é necessária com câmera fixa.

A implementação própria evita a dependência `boxmot` (conflito de versões) e mantém o rastreador fora da licença AGPL. Foi comparada ao `BYTETracker` do Ultralytics nas mesmas detecções (seção 8). As trilhas de um vídeo são salvas em `tracks.csv`; linha, região, zona morta e costura podem ser reconfiguradas e recontadas sem nova detecção.

### 5.3 Associação, duplicatas e oclusão

| Problema | Tratamento | O que não resolve |
|---|---|---|
| Detecção duplicada na sobreposição de tiles | Descarte de caixas coladas à borda interna + NMS (IoU e IoS) | Duas caixas legítimas muito próximas podem ser fundidas |
| Mesmo animal com dois IDs | Segunda etapa do ByteTrack; `track_buffer`; costura limitada de tracklets (`tracking/stitch.py`): une fim e início de trilhas com intervalo curto e posição/tamanho compatíveis | Reaparecimento após oclusão longa ou fora do quadro: não há reidentificação por aparência |
| Troca de ID | Kalman + IoU reduz, não elimina | Animais próximos e parecidos continuam sendo o pior caso |
| Oclusão | Confiança baixa aproveitada na segunda etapa; trilhas perdidas sobrevivem ao buffer | Animais ocultos durante todo o vídeo não existem para o sistema |
| Jitter na linha | Zona morta (histerese) e cooldown por ID e linha | Linha mal posicionada continua gerando contagem errada |

A costura é opcional e sempre sinalizada no resultado (`stitching.ids_unidos`), pois pode unir animais diferentes ou deixar de unir o mesmo animal.

### 5.4 Contagem

- **Linha virtual**: para cada trilha, calcula-se a distância assinada do ponto âncora (centro ou base da caixa) à linha. Há cruzamento quando o sinal muda, ignorando pontos dentro da zona morta (`deadband_px`). O ponto de cruzamento é interpolado e deve cair dentro do segmento. `cooldown_frames` impede nova contagem do mesmo ID logo após um evento. Direção: `in` é do lado negativo para o positivo da distância assinada `(x−ax)·dy − (y−ay)·dx`.
- **Região (ROI)**: evento `enter`/`exit` quando o ponto âncora muda de dentro/fora por `min_frames` quadros consecutivos. Calcula também a ocupação por quadro.
- **Auditoria**: cada evento registra número, tipo, alvo, direção, ID da trilha, quadro, tempo, posição, comprimento da trilha, score médio, flags (por exemplo `trilha_curta`) e o digest da configuração. Cada recontagem fica em `count_runs` (`GET /analyses/{id}/audit`).
- **Câmera em movimento**: linha e região são referenciadas ao quadro, não ao terreno. Para contagem de passagens, prefira câmera fixa ou voo planejado.
- **Referência de rebanho**: `herd_bounds` devolve `piso_observado_simultaneo` (maior contagem instantânea suavizada por mediana móvel), `ids_unicos_filtrados` (IDs com pelo menos 5 observações), `ids_unicos_brutos` e um aviso. Não existe campo "população" na API; um teste automatizado garante isso.

## 6. API

Documentação OpenAPI interativa em `/docs` do backend. Rotas (`backend/app/main.py`):

| Rota | Função |
|---|---|
| `GET /health`, `/limits`, `/models`, `/evaluations` | Estado, limites, modelos e avaliações registradas |
| `POST /infer/image` | Contagem de animais visíveis em imagem |
| `POST /videos` | Envia vídeo (202 + `analysis_id`) |
| `GET /analyses`, `/analyses/{id}`, `DELETE /analyses/{id}` | Listagem, estado/progresso e exclusão |
| `GET /analyses/{id}/preview.jpg`, `/analyses/{id}/tracks` | Pré-visualização e trilhas |
| `POST /analyses/{id}/count` | Recontagem (linha/região/zona morta/costura) sobre trilhas persistidas |
| `GET /analyses/{id}/runs/{rid}/events`, `events.csv`, `series.csv` | Eventos e séries de uma contagem |
| `GET /analyses/{id}/tracks.csv`, `runs/{rid}/annotated.mp4` | Trilhas em CSV e vídeo anotado (H.264, sob demanda) |
| `GET /analyses/{id}/report`, `report.html`, `audit` | Relatório (JSON/HTML) e trilha de auditoria |
| `WS /ws` | Progresso em tempo real |

As rotas que gravam (`POST`/`DELETE`) exigem `X-API-Key` quando `API_KEY` está definida. Em rede não confiável, coloque o serviço atrás de um proxy autenticado.

## 7. Métricas e validação

### 7.1 Detecção (`cattlecount/det_metrics.py`)

- AP com interpolação de 101 pontos (estilo COCO): AP50, AP50-95 (média de 0,50:0,05:0,95) e AP75. Casamento guloso por score; até 100 detecções por imagem.
- Precisão, revocação e F1 num limiar de confiança fixo, IoU 0,5.
- Implementação própria, conferida contra `pycocotools` e por testes.
- Em objetos minúsculos AP50-95 é severa (um erro de 3 px já derruba o IoU de uma caixa de 43 px); AP50 e AP50-95 divergem bastante e ambos são reportados.

### 7.2 Contagem (`cattlecount/count_metrics.py`)

Por imagem: MAE, RMSE, viés (previsto − GT; positivo indica supercontagem), MAPE (somente imagens com GT > 0) e erro percentual do total. IC 95 % por bootstrap reamostrando voos inteiros (imagens do mesmo voo são correlacionadas). Estratos: voo, altitude, densidade (0 | 1-5 | 6-15 | >=16) e fazenda. Uma contagem global pequena pode esconder falsos positivos e negativos que se cancelam; por isso precisão e revocação também são reportadas.

### 7.3 Rastreamento (`cattlecount/mot_metrics.py`, via TrackEval)

HOTA (DetA x AssA), IDF1, MOTA, IDSW, FP, FN, MT/PT/ML e fragmentações, conferidas contra `motmetrics` em testes (IDF1/MOTA/IDSW). Para objetos minúsculos reporta-se também IDF1 a IoU 0,3.

### 7.4 Passagens

Erro absoluto de passagens por linha contra a passagem calculada sobre o GT (trilhas verdadeiras passadas pelo mesmo contador). Mede o erro de detector e rastreador, não da regra de contagem.

### 7.5 Testes automatizados

`tests/`: `test_bytetrack`, `test_counting`, `test_det_metrics`, `test_geometry_tiling`, `test_ingest`, `test_mot_metrics`, `test_sim_video`, `test_api`, `test_postgres` (PostgreSQL via pgserver), `test_real_model` (modelo ONNX real) e `tests/e2e/test_ui.py` (Playwright). No frontend, `frontend/src/lib.test.ts` (vitest).

## 8. Resultados

As tabelas desta seção são geradas a partir de `experiments/*.json` por `scripts/gen_results_doc.py` (`make results`); nenhum número foi digitado à mão.

### 8.1 Em uma frase

- **Na fazenda de validação (Mauron, voo não usado no treino)** o modelo escolhido conta razoavelmente: MAE 1,80 animais/imagem, mAP50 0,589.
- **Em fazenda inédita (Jalogny)** ele **superconta muito**: 1724 animais previstos contra 809 reais (**+113 %**), precisão 0,126, mAP50 0,102. **Não use este modelo, como está, numa fazenda nova sem calibrar.**
- O detector COCO pré-treinado (zero-shot) quase não detecta nada nessa escala (revocação 0,033 no teste).

### 8.2 Detecção e contagem por imagem — modelos comparados

Limiar de confiança escolhido no VAL (melhor F1) e aplicado ao TESTE sem alteração. Teste = 92 imagens de 5 voos de Jalogny; val = 41 imagens de 1 voo de Mauron. IC95 % por bootstrap **de voos** (só 5 voos no teste: IC largo).

| modelo | conf | split | mAP50 | mAP50-95 | precisão | revocação | MAE contagem | viés | IC95 MAE (voos) |
|---|---|---|---|---|---|---|---|---|---|
| COCO zero-shot (yolo11s) | 0.05 | val | 0,000 | 0,000 | — | 0,000 | 8,71 | -8.71 | — (1 voo) |
| COCO zero-shot (yolo11s) | 0.05 | test | 0,030 | 0,010 | 0,587 | 0,033 | 8,29 | -8.29 | [5,3; 11,7] |
| YOLO11s, parada antecipada (época 16) | 0.3 | val | 0,542 | 0,113 | 0,584 | 0,630 | 2,68 | +0.68 | — (1 voo) |
| YOLO11s, parada antecipada (época 16) | 0.3 | test | 0,103 | 0,022 | 0,108 | 0,337 | 19,11 | +18.65 | [11,8; 27,4] |
| **YOLO11s, 40 épocas (selecionado)** | 0.25 | val | 0,589 | 0,127 | 0,701 | 0,630 | 1,80 | -0.88 | — (1 voo) |
| **YOLO11s, 40 épocas (selecionado)** | 0.25 | test | 0,102 | 0,020 | 0,126 | 0,269 | 11,16 | +9.95 | [7,4; 15,5] |

**Seleção do modelo:** regra aplicada: melhor F1 no val. **Transparência:** o primeiro modelo (parada antecipada) já tinha sido avaliado no teste quando se decidiu treinar as 40 épocas completas; a decisão de treinar mais não foi cega ao teste, mas a escolha entre os dois modelos usou só o val. F1 no val: 40 épocas 0,664 contra parada antecipada 0,606. O val também escolheu o checkpoint dentro de cada treino (métrica interna do Ultralytics), então os números do val são **um pouco otimistas**.

### 8.3 Onde o modelo selecionado erra no teste (fazenda inédita)

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

### 8.4 Adaptação a uma fazenda nova (experimento declarado antes de rodar)

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

### 8.5 Rastreamento

Parâmetros do tracker **não ajustados** nestes vídeos (cenário A: limiares derivados do conf operacional do detector; cenário B: padrões do ByteTrack original).

#### A · Vídeo aéreo **sintético** (janela deslizando sobre foto real; 60 quadros, 11 animais; GT exato)

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

#### B · Vídeo **terrestre real** (Mendeley, clipe com GT de 9 animais, 479 quadros; detector COCO)

O GT foi gerado com YOLOv8-COCO + DeepSORT (corrigido): **favorece detectores COCO**. É cenário terrestre, não aéreo.

| método | HOTA | IDF1 | MOTA | trocas de ID | IDs previstos |
|---|---|---|---|---|---|
| ByteTrack próprio | 0,727 | 0,880 | 0,841 | 3 | 8 |
| ByteTrack Ultralytics | 0,731 | 0,892 | 0,837 | 2 | 8 |
| Baseline IoU | 0,691 | 0,739 | 0,853 | 22 | 34 |

O ByteTrack próprio e o `BYTETracker` do Ultralytics (mesmas detecções e limiares) ficam praticamente empatados; ambos superam o baseline IoU em IDF1 e em trocas de ID.

### 8.6 Latência (CPU, ONNX Runtime, máquina ociosa)

Máquina: macOS-15.8.2-arm64-arm-64bit · CPUExecutionProvider (onnxruntime)

| operação | mediana (ms) | p95 (ms) | n |
|---|---|---|---|
| detecção de 1 tile 1024×1024 | 74.3 | 79.9 | 20 |
| quadro 1280×720 (inteiro) | 67.7 | 84.8 | 10 |
| foto 4000×3000 (fatiada) | 1437.5 | 1501.5 | 5 |
| ByteTrack, 1 quadro, 30 detecções | 0.5 | 0.7 | 300 |

Medido nesta máquina; **não** extrapole para outros hardwares ou para a VM do Podman.

### 8.7 O que estes resultados NÃO provam

- O teste tem **5 voos de uma única fazenda**: o IC por voo é largo e não representa outras fazendas.
- Subconjunto de 199 imagens do dataset (download limitado): resultados valem para esse subconjunto.
- **Não há GT de rastreamento em vídeo aéreo real**; o cenário A é sintético e o B é terrestre.
- Anotações do ICAERUS têm lacunas conhecidas; voos inconsistentes foram excluídos, mas ainda pode haver animais sem caixa nos voos usados (isso **subestima** a precisão).
- Um único modelo/tamanho (YOLO11s) e uma semente: sem variância entre execuções.

## 9. Mudança de domínio

O detector que funciona na fazenda de validação (Mauron) falha em fazenda inédita (Jalogny).

| Fator | Treino/val (Mauron, Derval) | Teste (Jalogny) |
|---|---|---|
| Drone/câmera | DJI Mavic 3T (4000x3000) | DJI Mavic 3E (5280x3956) |
| Paisagem | Pastagem aberta | Pastagem com mata, galhos secos, sombras e cursos d'água |
| Cor/iluminação | Uma estação | Várias datas (jun–set) |
| Animais | Mistura de pelagens do treino | Pelagens e tamanhos aparentes diferentes |

Observações:

1. Falsos positivos em mata: numa imagem de teste inspecionada, os animais reais foram detectados, mas dezenas de caixas surgiram em galhos secos e troncos claros. Em imagens de teste sem animais, o modelo previu animais em todos os limiares usados.
2. Superestimativa em todos os voos de teste.
3. Revocação baixa a IoU 0,5.
4. Mais épocas melhoraram val e contagem de teste, mas não resolveram o problema.

Não está demonstrado que a causa seja a mata (não houve ablação por cenário), nem que o resultado generalize para outras fazendas (o teste é uma fazenda, 5 voos).

Avaliação por condição: o mecanismo está implementado (estratos por voo, altitude, densidade e fazenda, IC por bootstrap de voos), mas os dados limitam as conclusões. Altitude: só há 60 m e 100 m, e o único voo de teste a 60 m está confundido com o próprio voo. Iluminação: não avaliada (poucos voos nublados entre os confiáveis). Densidade: avaliada. Oclusão: não há anotação, não avaliada.

Recomendações operacionais:

1. Calibrar por fazenda: o experimento de adaptação (seção 8.4) ajustou o modelo com 2 voos (35 imagens) de Jalogny e, nos 3 voos retidos, o erro de contagem caiu de cerca de 11 para cerca de 1,7 animais/imagem. Custo medido: o mesmo modelo piorou e passou a subcontar na fazenda original. Calibrar por fazenda exige imagens anotadas da fazenda-alvo e modelo separado ou novo treino.
2. Validar contra contagem manual de algumas imagens da fazenda antes de confiar nos números (o relatório guarda a configuração para repetir).
3. Tratar a contagem como estimativa com viés conhecido, não como censo.

## 10. Fontes de erro de contagem

| Fonte | Efeito em visíveis | Efeito em passagens | Indicador |
|---|---|---|---|
| Falso positivo (galho seco, sombra, rocha clara) | + | + (trilhas espúrias) | Viés positivo, precisão baixa |
| Animal não detectado (cor escura, oclusão, borda) | − | − | Viés negativo, revocação baixa |
| Duplicata (tiles, animais encostados) | + | + | Depende de NMS/IoS |
| Troca de ID | 0 | ± | IDSW |
| Fragmentação de trilha | 0 | + ou − (trilha curta filtrada) | Frag, IDs únicos > GT |
| Linha mal posicionada / zona morta | 0 | ± | Passagem perdida ou duplicada |
| Câmera móvel | 0 | + (cruzamento causado pela câmera) | Ver seção 5.4 |
| Animais ocultos ou fora do quadro | − | − | Não corrigível por software |

## 11. Limitações conhecidas

**Modelo**
- Treinado apenas na ICAERUS (3 fazendas europeias, 60–100 m, nadir). Não validado para outras raças, relevos, altitudes, ângulos oblíquos ou iluminação diferente; em fazenda inédita superconta.
- Animais escuros e ocultos são perdidos com frequência; animais encostados podem ser fundidos.
- Escala: treinado com animais de cerca de 46 px (mediana); altitudes muito diferentes alteram a escala.

**Rastreamento e contagem**
- Sem verdade de rastreamento em vídeo aéreo real; validação em vídeo terrestre real e aéreo sintético.
- Com detector fraco as trilhas fragmentam e a contagem de passagens fica abaixo da verdade; com troca de ID pode ficar acima.
- Linha/região são do quadro, não do terreno.
- Sem reidentificação por aparência; a costura limitada só atua em intervalos curtos.
- IDs únicos não são população; o sistema não estima tamanho de rebanho.

**Sistema**
- Autenticação limitada a chave opcional; sem usuários nem papéis. Não exponha à internet sem proxy autenticado.
- Um worker de vídeo com fila em memória: jobs em andamento se perdem se o processo cair; sem reprocessamento automático.
- Vídeos e trilhas em volume local, sem política de retenção.
- O vídeo anotado só é reencodado para H.264 com o ffmpeg do pacote `imageio-ffmpeg`; sem ele o MP4 pode não tocar em alguns navegadores.
- A pilha em contêineres foi executada com Podman; Docker Desktop não foi testado.
- O frontend não tem testes de acessibilidade além de rótulos ARIA básicos.

**Uso responsável**: estimativas de modelo não substituem contagem física para decisões de manejo, comerciais, sanitárias ou regulatórias.

## 12. Licenças e atribuições

- **Código do repositório**: GPL-3.0 (`LICENSE`).
- **Ultralytics (YOLO11, treino e pesos)**: AGPL-3.0. Os pesos (`.pt`, `.onnx`) não são versionados. O uso comercial ou como serviço em rede exige avaliar a AGPL ou obter licença comercial da Ultralytics. O rastreador (ByteTrack), a contagem e a API não têm componente AGPL no caminho de inferência (ONNX Runtime).
- **Dados**: ICAERUS (Zenodo 10.5281/zenodo.11048412) e Mendeley Data dk54zg67dd, ambos CC BY 4.0; atribuição obrigatória em `sample_data/ATTRIBUTION.md`.
- **Model card**: `models/MODEL_CARD.md`.

## 13. Solução de problemas

- **Build do frontend falha com `EAI_AGAIN` em `npm ci`** (Podman/macOS): o Dockerfile do frontend usa `node:22-slim` (glibc); com `node:22-alpine` o resolvedor musl falhou na VM do Podman.
- **`apt-get` com pacotes de hash inválido durante o build**: observado na rede da VM do Podman; por isso o ffmpeg vem do PyPI (`imageio-ffmpeg`) e não do `apt`.
- **Upload grande via `curl` falha com "Invalid multipart data"** no nginx: `curl` envia `Expect: 100-continue`; use `curl -H "Expect:"`. Navegadores não são afetados.
- **Memória**: a VM do Podman tem cerca de 3,7 GB de RAM; a pilha cabe, mas não execute treino dentro dela.
- **Inferência indisponível**: os pesos ONNX não estão em `models/onnx/`; ver seção 3.1.
- **MP4 anotado não toca no navegador**: ffmpeg do `imageio-ffmpeg` ausente (seção 11).
