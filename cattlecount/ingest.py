"""Ingestão e validação de imagens e vídeos. Não "conserta" nada em silêncio: devolve erros (rejeita) e avisos (aceita, mas informa)."""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png"}
VID_EXT = {".mp4", ".mov", ".avi", ".mkv"}
LIMITS = {"max_image_pixels": 120_000_000, "max_video_frames": 60_000, "max_video_side": 4096, "min_side": 64}


class IngestError(ValueError):
    """Entrada rejeitada (mensagem em português, segura para devolver ao usuário)."""


@dataclass
class IngestReport:
    kind: str
    sha256: str
    width: int
    height: int
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def dict(self):
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _xmp_altitude(raw: bytes) -> dict:
    """Procura RelativeAltitude/AbsoluteAltitude no XMP embutido (DJI grava aqui). Retorna {} se ausente — nunca um valor presumido."""
    out = {}
    head = raw[:262_144]
    for key in ("RelativeAltitude", "AbsoluteAltitude"):
        m = re.search(rb'drone-dji:' + key.encode() + rb'="([+-]?\d+(?:\.\d+)?)"', head)
        if m:
            out[key] = float(m.group(1))
    return out


def validate_image_bytes(data: bytes, filename: str = "") -> tuple[np.ndarray, IngestReport]:
    ext = Path(filename).suffix.lower()
    if filename and ext not in IMG_EXT:
        raise IngestError(f"formato de imagem não suportado: '{ext}' (use JPG ou PNG)")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:  # noqa: BLE001
        raise IngestError("arquivo de imagem inválido ou corrompido") from e
    if im.width * im.height > LIMITS["max_image_pixels"]:
        raise IngestError(f"imagem grande demais ({im.width}x{im.height})")
    if min(im.size) < LIMITS["min_side"]:
        raise IngestError("imagem pequena demais")
    warns, meta = [], {}
    exif = im.getexif()
    if exif.get(274, 1) not in (1, None):
        warns.append("orientação EXIF aplicada (a imagem foi rotacionada para a orientação correta antes da detecção)")
    im = ImageOps.exif_transpose(im).convert("RGB")
    meta |= _xmp_altitude(data)
    if "RelativeAltitude" not in meta:
        warns.append("altitude de voo ausente nos metadados: a escala (cm/pixel) é desconhecida; contagem não pode ser comparada entre altitudes")
    if max(im.size) < 1500:
        warns.append("resolução baixa para imagens aéreas: animais muito pequenos podem não ser detectados")
    arr = np.asarray(im)
    if arr.std() < 3:
        warns.append("imagem quase uniforme (pouco contraste): verifique se o arquivo está correto")
    return arr, IngestReport("image", hashlib.sha256(data).hexdigest(), arr.shape[1], arr.shape[0], warns, meta)


def validate_video_file(path: str | Path) -> IngestReport:
    p = Path(path)
    if p.suffix.lower() not in VID_EXT:
        raise IngestError(f"formato de vídeo não suportado: '{p.suffix}'")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise IngestError("não foi possível abrir o vídeo (corrompido ou codec não suportado)")
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, first = cap.read()
    cap.release()
    if not ok or first is None:
        raise IngestError("o vídeo não contém quadros legíveis")
    if max(w, h) > LIMITS["max_video_side"]:
        raise IngestError(f"resolução do vídeo acima do limite ({w}x{h})")
    if n > LIMITS["max_video_frames"]:
        raise IngestError(f"vídeo longo demais ({n} quadros; limite {LIMITS['max_video_frames']})")
    warns = []
    if not (1 <= fps <= 240):
        warns.append(f"FPS inválido ou ausente ({fps:.2f}): tempos em segundos não são confiáveis; usando 30 fps como referência de exibição")
        fps = 30.0
    if fps < 10:
        warns.append("FPS baixo: associação entre quadros fica mais difícil (deslocamento grande entre quadros)")
    if n <= 0:
        warns.append("contagem de quadros não informada pelo contêiner")
    if max(w, h) < 640:
        warns.append("resolução baixa: animais pequenos podem não ser detectados")
    return IngestReport("video", sha256_file(p), w, h, warns, {"fps": fps, "n_frames": n, "duration_s": n / fps if n > 0 else None})
