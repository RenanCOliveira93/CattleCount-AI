"""Baixa partes de um ZIP remoto (Zenodo) por HTTP Range, em poucas requisições, com retry/backoff para 429.
Uso: fetch_members(url, infos, out_dir). Respeita o servidor: nunca abre conexões em paralelo."""
from __future__ import annotations

import struct
import time
import zlib
from pathlib import Path

import requests
from remotezip import RemoteZip


def _get(url: str, rng: str, tries: int = 8) -> bytes:
    for k in range(tries):
        r = requests.get(url, headers={"Range": f"bytes={rng}"}, timeout=300)
        if r.status_code == 206:
            return r.content
        if r.status_code in (429, 503):
            time.sleep(min(120, 10 * (k + 1)))
            continue
        r.raise_for_status()
    raise RuntimeError(f"falhou após {tries} tentativas: {rng}")


def infolist(url: str, tries: int = 6):
    for k in range(tries):
        try:
            with RemoteZip(url) as z:
                return z.infolist()
        except Exception:
            time.sleep(15 * (k + 1))
    raise RuntimeError("não foi possível ler o diretório do ZIP")


def fetch_members(url: str, infos, out_dir: Path, max_gap: int = 2_000_000, max_chunk: int = 64_000_000) -> int:
    """Extrai os `infos` (ZipInfo) agrupando faixas próximas. Retorna nº de arquivos gravados."""
    infos = sorted(infos, key=lambda i: i.header_offset)
    groups, cur = [], []
    for i in infos:
        end = i.header_offset + 30 + 1024 + i.compress_size
        if cur and (i.header_offset - cur[-1][1] > max_gap or end - cur[0][0].header_offset > max_chunk):
            groups.append(cur); cur = []
        cur.append((i, end))
    if cur:
        groups.append(cur)
    n = 0
    for g in groups:
        start, stop = g[0][0].header_offset, g[-1][1]
        buf = _get(url, f"{start}-{stop}")
        for i, _ in g:
            o = i.header_offset - start
            sig, = struct.unpack("<I", buf[o:o + 4])
            assert sig == 0x04034B50, "cabeçalho local inválido"
            nlen, elen = struct.unpack("<HH", buf[o + 26:o + 30])
            data = buf[o + 30 + nlen + elen:o + 30 + nlen + elen + i.compress_size]
            raw = data if i.compress_type == 0 else zlib.decompress(data, -15)
            assert len(raw) == i.file_size, f"tamanho incorreto: {i.filename}"
            p = out_dir / i.filename
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw); n += 1
    return n
