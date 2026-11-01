# 11 · Como rodar

## Contêineres (Docker **ou** Podman)
```bash
cp .env.example .env            # defina POSTGRES_PASSWORD
docker compose up --build       # ou: podman compose up --build
# abre http://localhost:8080
```
O `docker-compose.yml` usa imagens com nome totalmente qualificado (`docker.io/...`) e nenhum recurso exclusivo do Docker. Foi **executado e testado com Podman 6.1.4** (provider docker-compose 5.5.1). **Docker Desktop não foi testado.**

Notas aprendidas ao rodar no Podman (macOS):
- A etapa de build do frontend usa `node:22-slim`; com `node:22-alpine` o `npm ci` falhou na VM do Podman com `EAI_AGAIN` (resolvedor musl).
- `apt-get` dentro do build mostrou pacotes com hash inválido (rede da VM); por isso o ffmpeg vem do PyPI (`imageio-ffmpeg`), não do `apt`.
- `curl` com arquivo grande envia `Expect: 100-continue` e o upload via nginx falhou com "Invalid multipart data"; use `curl -H "Expect:"`. Navegadores não são afetados (E2E passa).
- A VM do Podman tem ~3,7 GB de RAM; a pilha cabe, mas não rode treino dentro dela.

## Local (sem contêiner)
```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
.venv/bin/python -m uvicorn backend.app.asgi:app --port 8000      # SQLite por padrão
cd frontend && npm ci && npm run dev                                # http://localhost:5173 (proxy /api → :8000)
```
## Testes
```bash
.venv/bin/python -m pytest                       # unidade + API + PostgreSQL real (pgserver) + modelo real
.venv/bin/python -m pytest tests/e2e -m e2e      # navegador real contra a pilha em http://localhost:8080
cd frontend && npx vitest run && npx tsc --noEmit
```
## Reproduzir o treino e os experimentos
`training/` tem os scripts na ordem: `zenodo_plan.py` → `zenodo_download_images.py` → `build_tiles.py` → `train_yolo.py` → `predict_split.py` → `eval_predictions.py` → `eval_tracking.py`. Os resultados ficam em `experiments/*.json` e são agregados em `experiments/registry.json`.
