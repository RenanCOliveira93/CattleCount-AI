import io

import cv2
import numpy as np
import pytest
from PIL import Image

from cattlecount.ingest import IngestError, validate_image_bytes, validate_video_file


def _jpg(w=800, h=600, xmp=b""):
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(buf, "JPEG")
    b = buf.getvalue()
    if xmp:  # como nos arquivos DJI: XMP em segmento APP1 logo após o SOI
        payload = b"http://ns.adobe.com/xap/1.0/\x00" + xmp
        b = b[:2] + b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload + b[2:]
    return b


def test_imagem_valida_com_avisos_e_sem_altitude_inventada():
    arr, rep = validate_image_bytes(_jpg(), "a.jpg")
    assert arr.shape == (600, 800, 3) and "RelativeAltitude" not in rep.meta
    assert any("altitude" in w for w in rep.warnings) and any("resolução baixa" in w for w in rep.warnings)


def test_altitude_xmp_lida_quando_existe():
    _, rep = validate_image_bytes(_jpg(xmp=b' drone-dji:RelativeAltitude="+99.80" '), "a.jpg")
    assert rep.meta["RelativeAltitude"] == pytest.approx(99.8) and not any("altitude de voo ausente" in w for w in rep.warnings)


@pytest.mark.parametrize("data,name", [(b"nao sou imagem", "a.jpg"), (_jpg(), "a.gif"), (_jpg(32, 32), "a.jpg")])
def test_imagem_rejeitada(data, name):
    with pytest.raises(IngestError):
        validate_image_bytes(data, name)


def test_video_valido_e_invalido(tmp_path):
    p = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 5, (320, 240))
    for _ in range(10):
        w.write(np.zeros((240, 320, 3), np.uint8))
    w.release()
    rep = validate_video_file(p)
    assert rep.meta["n_frames"] == 10 and any("FPS baixo" in x for x in rep.warnings) and any("resolução baixa" in x for x in rep.warnings)
    bad = tmp_path / "x.mp4"; bad.write_bytes(b"lixo")
    with pytest.raises(IngestError):
        validate_video_file(bad)
    with pytest.raises(IngestError):
        validate_video_file(tmp_path / "x.txt")
