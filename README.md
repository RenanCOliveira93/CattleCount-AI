# CattleCount AI

Detecção, contagem e rastreamento de bovinos em **imagens aéreas** e **vídeos**, com API, painel web e trilha de auditoria.
*(English: [README.en.md](README.en.md))*

> **Leia isto primeiro.** O modelo foi treinado e avaliado no dataset ICAERUS (drones, 60–100 m). Em **fazenda inédita ele superconta muito** (+113 % no teste). Os números são **estimativas** e **não são um censo do rebanho**. Detalhes honestos em [docs/pt-BR/12_resultados.md](docs/pt-BR/12_resultados.md) e [10_limitacoes_operacionais.md](docs/pt-BR/10_limitacoes_operacionais.md).

## As três contagens (nunca confundir)
| | O que é | O que **não** é |
|---|---|---|
| **Visíveis agora** | animais detectados num quadro/imagem | não inclui ocultos nem fora do quadro |
| **Passagens** | cruzamentos de linha / entradas e saídas de região, em vídeo | não é nº de animais (o mesmo animal pode cruzar várias vezes; troca de ID distorce) |
| **Referência de rebanho** | só *faixas*: piso observado e IDs únicos | **nunca é população**; não existe campo "população" na API |

## Rodar (Docker **ou** Podman)
```bash
cp .env.example .env              # defina POSTGRES_PASSWORD
docker compose up --build         # ou: podman compose up --build
# http://localhost:8080
```
Os pesos ONNX (`models/onnx/cattle_yolo11s_1024.onnx`, ~36 MB, **AGPL‑3.0**) não vão no git: veja [models/MODEL_CARD.md](models/MODEL_CARD.md). Exemplos em `sample_data/` (foto aérea real recortada + vídeo **sintético**, CC BY 4.0).
Execução local sem contêiner e demais detalhes: [docs/pt-BR/11_como_rodar.md](docs/pt-BR/11_como_rodar.md).

## O que tem aqui
`backend/` FastAPI + SQLAlchemy/PostgreSQL · `frontend/` React/TypeScript (upload, linha/região, trilhas, gráficos, CSV, auditoria) · `cattlecount/` núcleo (tiling, ByteTrack próprio, contagem, métricas) · `training/` dados→treino→avaliação · `models/` · `experiments/` registro **gerado dos JSONs reais** · `notebooks/` (executados) · `tests/` · `docs/pt-BR/` · `infra/` · `docker-compose.yml`.

## Resultados em uma olhada (detalhes e IC em docs/pt-BR/12_resultados.md)
| | Resultado real |
|---|---|
| Detector, fazenda de validação (Mauron) | mAP50 0,589 · MAE de contagem 1,80 animais/imagem |
| Detector, **fazenda inédita** (Jalogny, 92 imagens, 5 voos) | mAP50 0,102 · precisão 0,126 · **1 724 previstos para 809 reais** |
| **Adaptação** com 2 voos da própria fazenda (voos retidos de Jalogny) | MAE 11,14 → **1,70**; viés +9,7 → −1,1. **Custo:** piora na fazenda original (mAP50 0,589 → 0,368). Ver seção 4 de docs/pt-BR/12_resultados.md |
| Detector COCO sem ajuste | quase nada detectado (revocação 0,03) |
| ByteTrack em vídeo terrestre real (Mendeley) | IDF1 0,880 (Ultralytics: 0,892; baseline IoU: 0,739) |
| ByteTrack com detecções perfeitas (sintético) | IDF1 0,999: o rastreador funciona; o gargalo é o detector |
| ByteTrack com detector real (sintético) | IDF1 ≈ 0,40; passagens 0 contra 3 reais |

## Testes
`pytest` (unidade, API, **PostgreSQL real**, modelo ONNX real) · `pytest tests/e2e -m e2e` (navegador real contra a pilha em contêineres) · `vitest`. Estado e limites do que foi executado: ver [docs/pt-BR/11_como_rodar.md](docs/pt-BR/11_como_rodar.md).

## Licenças e atribuição
Código: ver `docs/pt-BR/10_limitacoes_operacionais.md` (pesos/código Ultralytics são **AGPL‑3.0**). Dados: ICAERUS (Zenodo 10.5281/zenodo.11048412) e Mendeley dk54zg67dd, **CC BY 4.0**; atribuição em `sample_data/ATTRIBUTION.md`.

## Repositório, versionamento e execução em outra máquina

Código publicado a partir da máquina de desenvolvimento original (macOS, Podman). Isto **não** foi executado em GitHub Codespaces nem em Docker Desktop; se rodar lá, trate como primeira verificação e compare com a seção de testes.

- **Versionado:** `cattlecount/`, `training/`, `backend/`, `frontend/`, `infra/`, `sample_data/`, notebooks executados, testes, `docs/pt-BR/` e `experiments/` (JSONs reais de onde os números são gerados).
- **Não versionado:** `data/` (≈ 2,3 GB, via `make data`, precisa de rede), `runs/` e **todos os pesos** (`models/*.pt`, `models/onnx/`; AGPL-3.0, veja `models/MODEL_CARD.md`). Sem os pesos a API sobe, mas a inferência exige gerá-los com `make data && make train` ou obtê-los conforme o MODEL_CARD.
- Rodado em Podman (compose + Postgres reais) e E2E Playwright; **Docker Desktop não testado**.
- **Licença do código:** GPL-3.0 (arquivo `LICENSE`). Licenças e atribuições de dados/modelos de terceiros: veja as seções e docs indicados acima.
