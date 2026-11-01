"""Baixa, em ordem de prioridade e de forma resumível, as imagens do plano (uma conexão; respeita 429). Log em data/raw/zenodo/dl_images.log"""
import json, pickle, sys, time, zipfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
import zenodo_fetch as Z

ROOT = Path(__file__).resolve().parents[1]
U = "https://zenodo.org/api/records/11048412/files/Cattle_drone_images_042024.zip/content"
plan = json.load(open(ROOT / "data/raw/zenodo/download_plan.json"))["order"]
inf = {n: SimpleNamespace(filename=n, file_size=s, compress_size=c, compress_type=t, header_offset=o) for n, s, c, t, o in pickle.load(open(ROOT / "data/raw/zenodo/infos.pkl", "rb"))}
out = ROOT / "data/raw/zenodo/ex"
t0 = time.time(); done = 0
for k, o in reversed(list(enumerate(plan))):
    dst = out / o["member"]
    if dst.exists() and dst.stat().st_size == o["size"]:
        done += 1; continue
    try:
        Z.fetch_members(U, [inf[o["member"]]], out)
        done += 1
        print(f"{k + 1}/{len(plan)} {o['split']} {o['stem']} {o['size'] / 1e6:.1f}MB t={time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        print("ERRO", o["stem"], e, flush=True); time.sleep(30)
print("FIM", done, "de", len(plan), flush=True)
