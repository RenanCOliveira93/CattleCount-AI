"""Plano de download do dataset ICAERUS (Zenodo 11048412): decide QUAIS imagens baixar (o ZIP tem 16,7 GB; o servidor entrega ~0,2–0,6 MB/s).
Voo 'confiável' = nº de caixas nos rótulos baixados == nº do resumo oficial (±1). Voos sem rótulo ou divergentes ficam FORA (sem rótulo ≠ sem gado)."""
from __future__ import annotations

import collections
import glob
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "data/raw/zenodo/ex"
SPLIT_BY_FLIGHT = {  # decisão tomada ANTES de olhar qualquer resultado de modelo
    "train": ["DJI_202306051814_022", "DJI_202306051839_025", "DJI_202306071703_029", "DJI_202306071745_034", "DJI_202306071745_035", "DJI_202308091442_012"],
    "val": ["DJI_202306071624_028"],
}  # tudo de Jalogny confiável vira teste (fazenda inédita)


def stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def load():
    inf = pickle.load(open(ROOT / "data/raw/zenodo/infos.pkl", "rb"))
    imgs = {stem(n): (n, s) for n, s, *_ in inf if n.lower().endswith(".jpg")}
    flight = {s: v[0].split("/")[3] for s, v in imgs.items()}
    farm = {s: v[0].split("/")[1] for s, v in imgs.items()}
    lab = {}
    for f in glob.glob(str(EX / "**/obj_train_data/*.txt"), recursive=True):
        L = [l.split() for l in open(f).read().splitlines() if l.strip()]
        lab[stem(f)] = np.array(L, float).reshape(-1, 5) if L else np.zeros((0, 5))
    rows = list(openpyxl.load_workbook(EX / "Cattle_drone_images/conditions_summary.xlsx", data_only=True).active.iter_rows(values_only=True))[1:]
    return imgs, flight, farm, lab, rows


def reliable_flights(flight, lab, rows):
    nb = collections.Counter()
    nl = collections.Counter()
    for s, v in lab.items():
        if s in flight:
            nb[flight[s]] += len(v); nl[flight[s]] += 1
    out = {}
    for r in rows:
        f = r[2]
        # chaves de voo repetem entre fazendas? usa fazenda+pasta quando necessário
        out.setdefault(f, []).append(r)
    ok = {}
    for f, rs in out.items():
        r = rs[0]
        ok[f] = {"farm": r[1], "alt": r[3], "weather": r[8].strip(), "drone": r[10], "size": r[12], "summary_boxes": r[6], "label_boxes": nb[f],
                 "reliable": nl[f] > 0 and abs(nb[f] - r[6]) <= 1 and len(rs) == 1}
    return ok


def split_of(flight_name: str, farm: str) -> str | None:
    for sp, fl in SPLIT_BY_FLIGHT.items():
        if flight_name in fl:
            return sp
    return "test" if farm == "Jalogny" else None


def plan(seed: int = 0, n_neg_per_flight: int = 4):
    imgs, flight, farm, lab, rows = load()
    info = reliable_flights(flight, lab, rows)
    rng = random.Random(seed)
    per = collections.defaultdict(lambda: {"cow": [], "neg": []})
    for s, v in lab.items():
        f = flight.get(s)
        if f is None or not info.get(f, {}).get("reliable"):
            continue
        sp = split_of(f, farm[s])
        if sp:
            per[(sp, f)]["cow" if len(v) else "neg"].append(s)
    order = []
    for k, d in per.items():
        rng.shuffle(d["neg"])
        d["neg"] = d["neg"][:n_neg_per_flight]
    # intercala splits: round-robin entre voos, cow primeiro
    queues = {k: d["cow"] + d["neg"] for k, d in per.items()}
    keys = sorted(queues)
    while any(queues.values()):
        for k in keys:
            if queues[k]:
                s = queues[k].pop(0); order.append({"stem": s, "split": k[0], "flight": k[1], "farm": farm[s], "member": imgs[s][0], "size": imgs[s][1], "boxes": int(len(lab[s]))})
    return order, info


if __name__ == "__main__":
    order, info = plan()
    (ROOT / "data/raw/zenodo/download_plan.json").write_text(json.dumps({"order": order, "flights": info}, indent=1))
    print("voos confiáveis:", [f for f, v in info.items() if v["reliable"]])
    print("voos NÃO confiáveis:", {f: (v["label_boxes"], v["summary_boxes"]) for f, v in info.items() if not v["reliable"]})
    c = collections.Counter((o["split"], o["boxes"] > 0) for o in order); print(dict(c), "GB:", sum(o["size"] for o in order) / 1e9)
