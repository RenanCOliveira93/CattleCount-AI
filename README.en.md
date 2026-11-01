# CattleCount AI

Cattle detection, counting and tracking in **aerial images** and **videos**, with an API, a web dashboard and an audit trail.
*(Português: [README.md](README.md); the in-depth docs are in Portuguese under `docs/pt-BR/`.)*

> **Read this first.** The model was trained and evaluated on ICAERUS (drone imagery, 60–100 m). On an **unseen farm it heavily over-counts** (+113 % on the test set). Outputs are **estimates**, **not a herd census**. Honest details: [docs/pt-BR/12_resultados.md](docs/pt-BR/12_resultados.md), [10_limitacoes_operacionais.md](docs/pt-BR/10_limitacoes_operacionais.md).

## Three different counts (never conflated)
| | What it is | What it is **not** |
|---|---|---|
| **Visible now** | animals detected in one frame/image | excludes occluded and out-of-frame animals |
| **Passages** | line crossings / region enter-exit events in video | not a head count (one animal can cross repeatedly; ID switches distort it) |
| **Herd reference** | ranges only: observed simultaneous floor and unique IDs | **never a population**; the API has no "population" field |

## Run (Docker **or** Podman)
```bash
cp .env.example .env              # set POSTGRES_PASSWORD
docker compose up --build         # or: podman compose up --build
# http://localhost:8080
```
Model weights (`models/onnx/cattle_yolo11s_1024.onnx`, ~36 MB, **AGPL-3.0**) are not committed: see [models/MODEL_CARD.md](models/MODEL_CARD.md). Samples in `sample_data/` (a cropped real aerial photo + a **synthetic** video; CC BY 4.0).
The compose stack was **executed and tested with Podman 6.1.4**; **Docker Desktop was not tested**.

## Results at a glance (details and CIs in docs/pt-BR/12_resultados.md)
| | Measured |
|---|---|
| Detector, validation farm (Mauron) | mAP50 0.589 · count MAE 1.80 animals/image |
| Detector, **unseen farm** (Jalogny, 92 images, 5 flights) | mAP50 0.102 · precision 0.126 · **1,724 predicted vs 809 real** |
| **Adaptation** with 2 flights from the target farm (held-out Jalogny flights) | count MAE 11.14 → **1.70**; bias +9.7 → −1.1. **Cost:** worse on the original farm (mAP50 0.589 → 0.368). See section 4 of docs/pt-BR/12_resultados.md |
| Off-the-shelf COCO detector | almost nothing detected (recall 0.03) |
| ByteTrack on real ground-level video (Mendeley) | IDF1 0.880 (Ultralytics: 0.892; IoU baseline: 0.739) |
| ByteTrack with perfect detections (synthetic) | IDF1 0.999: the tracker works; the detector is the bottleneck |
| ByteTrack with the real detector (synthetic) | IDF1 ≈ 0.40; 0 passages vs 3 real |

## Caveats
No tracking ground truth exists for real aerial video here; the aerial tracking scenario is **synthetic** (a window sliding over a real photo). The test set is one farm (5 flights). Dataset subset: 199 images (throttled download). One model size, one seed.

## Licences
Ultralytics code/weights are **AGPL-3.0**. Data: ICAERUS and Mendeley dk54zg67dd, **CC BY 4.0** (attribution in `sample_data/ATTRIBUTION.md`).

## Repository, versioning and running on another machine

Published from the original development machine (macOS, Podman). This has **not** been run on GitHub Codespaces or Docker Desktop; if you run it there, treat it as a first verification and compare with the tests section.

- **Versioned:** `cattlecount/`, `training/`, `backend/`, `frontend/`, `infra/`, `sample_data/`, executed notebooks, tests, `docs/pt-BR/` and `experiments/` (real JSONs the numbers are generated from).
- **Not versioned:** `data/` (≈ 2.3 GB via `make data`, network required), `runs/` and **all weights** (`models/*.pt`, `models/onnx/`; AGPL-3.0, see `models/MODEL_CARD.md`). Without weights the API starts but inference needs `make data && make train` or obtaining them per the model card.
- Run on Podman (real compose + Postgres) and Playwright E2E; **Docker Desktop untested**.
- **Code licence:** GPL-3.0 (`LICENSE`). Third-party data/model licences and attributions: see the sections and docs above.
