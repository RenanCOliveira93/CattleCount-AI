"""Gera experiments/registry.json e models/registry.json a partir dos artefatos REAIS (experiments/*.json). Nenhuma métrica é digitada à mão."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "experiments"
det = []
for p in sorted(EX.glob("*.json")):
    if p.stem in ("registry", "tracking_eval", "benchmark"):
        continue
    e = json.loads(p.read_text())
    row = {"name": e["name"], "arquivo": f"experiments/{p.name}", "pesos": e["weights"], "coco_zero_shot": e["coco_zero_shot"], "conf_operacional": e["conf_operacional"],
           "regra_conf": e["regra_conf"], "voos_de_teste": e.get("test_flights")}
    for s in ("val", "test"):
        o = e[s]
        row[s] = {"n_imagens": o["n_images"], "caixas_gt": o["n_gt_boxes"], "map50": o["map50"], "map50_95": o["map50_95"], "precisao": o["prf"]["precision"], "revocacao": o["prf"]["recall"],
                  "contagem_mae": o["contagem"]["mae"], "contagem_rmse": o["contagem"]["rmse"], "contagem_vies": o["contagem"]["bias"], "contagem_mae_ic95_por_voo": o["contagem_ic95_bootstrap_por_voo"]["mae"]}
    det.append(row)
reg = {"deteccao_e_contagem": det}
for k in ("tracking_eval", "benchmark"):
    if (EX / f"{k}.json").exists():
        reg[k] = f"experiments/{k}.json"
(EX / "registry.json").write_text(json.dumps(reg, indent=1, ensure_ascii=False))

best = next((r for r in det if r["name"] == "yolo11s_cattle_full"), None)
models = [{"id": "cattle_yolo11s_1024", "nome": "YOLO11s ajustado em tiles ICAERUS (40 épocas)", "formato": "onnx 1024×1024", "arquivo": "models/onnx/cattle_yolo11s_1024.onnx",
           "classes": ["cow"], "licenca": "AGPL-3.0 (pesos e código Ultralytics); dados de treino CC BY 4.0",
           "treinado_em": "ICAERUS (Zenodo 11048412): voos de Mauron e Derval, tiles nativos 1024 px", "regra_de_selecao": "melhor F1 no val (voo Mauron 028) entre dois checkpoints; conf escolhido no val",
           "metricas": best, "limite": "Não validado fora das fazendas/drones/altitudes do ICAERUS; desempenho em fazenda inédita (Jalogny) é fraco: ver docs/pt-BR/12_resultados.md"}]
ad = next((r for r in det if r["name"] == "yolo11s_cattle_adapt__jalogny_holdout"), None)
if ad:
    models.append({"id": "cattle_yolo11s_1024_jalogny_adapt", "nome": "YOLO11s adaptado a Jalogny (2 voos) — NÃO é o padrão", "formato": "onnx 1024×1024", "arquivo": "models/onnx/cattle_yolo11s_1024_jalogny_adapt.onnx",
                   "classes": ["cow"], "licenca": "AGPL-3.0", "treinado_em": "modelo de 40 épocas + tiles de 2 voos de Jalogny (031, 058), 30 épocas, lr0 0,002",
                   "metricas": ad, "limite": "Melhora muito em Jalogny (voos retidos), mas PIORA e subconta em Mauron (esquecimento). Usar só na fazenda de adaptação."})
(ROOT / "models/registry.json").write_text(json.dumps(models, indent=1, ensure_ascii=False))
print([r["name"] for r in det])
