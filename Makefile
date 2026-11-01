PY=.venv/bin/python
.PHONY: setup data train eval track bench registry results notebooks api web test e2e up down
setup:        ## venv + dependências de execução e de treino
	python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest ultralytics pycocotools TrackEval motmetrics pgserver playwright nbconvert ipykernel matplotlib pandas openpyxl remotezip
data:         ## plano de download -> imagens -> tiles (exige rede; ~2,3 GB, lento)
	$(PY) training/zenodo_plan.py && $(PY) training/zenodo_download_images.py && $(PY) training/build_tiles.py
train:
	$(PY) training/train_yolo.py yolo11s yolo11s_cattle_full 40 100
eval:         ## predições fatiadas (val e teste) + métricas
	$(PY) training/predict_split.py models/yolo11s_cattle_full_best.pt val runs/preds/full_val.json && $(PY) training/predict_split.py models/yolo11s_cattle_full_best.pt test runs/preds/full_test.json && $(PY) training/eval_predictions.py yolo11s_cattle_full runs/preds/full_val.json runs/preds/full_test.json
track:
	$(PY) training/eval_tracking.py
bench:        ## rode com a máquina ociosa
	$(PY) scripts/benchmark.py
registry:
	$(PY) scripts/build_registry.py
results:
	$(PY) scripts/gen_results_doc.py
notebooks:
	cd notebooks && ../$(PY) build_01.py && ../$(PY) build_02.py && ../$(PY) build_03.py
api:
	$(PY) -m uvicorn backend.app.asgi:app --port 8000
web:
	cd frontend && npm ci && npm run dev
test:
	$(PY) -m pytest -q && cd frontend && npx vitest run && npx tsc --noEmit
e2e:          ## requer a pilha no ar (make up)
	$(PY) -m pytest tests/e2e -m e2e -q
up:
	podman compose up -d --build     # ou: docker compose up -d --build
down:
	podman compose down
