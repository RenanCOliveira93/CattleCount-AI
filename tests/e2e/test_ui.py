"""E2E em navegador real (Chromium/Playwright) contra a pilha completa (nginx + API + PostgreSQL), p.ex. subida com `podman compose up -d`.
Executar: pytest tests/e2e -m e2e   (BASE_URL padrão http://localhost:8080). Usa o vídeo SINTÉTICO de sample_data e a foto aérea real recortada."""
import os
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[2]
IMG, VID = ROOT / "sample_data/icaerus_mauron_crop.jpg", ROOT / "sample_data/demo_pan_synthetic.mp4"
SHOTS = ROOT / "docs/img"
BASE = os.environ.get("BASE_URL", "http://localhost:8080")
pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def page():
    import urllib.request
    try:
        urllib.request.urlopen(f"{BASE}/api/health", timeout=3)
    except Exception:
        pytest.skip(f"pilha não está no ar em {BASE}")
    with pw.sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_context(viewport={"width": 1280, "height": 1000}).new_page()
        yield pg
        b.close()


def test_a_imagem_visiveis_e_avisos(page):
    page.goto(BASE)
    assert "modelo" in page.inner_text("[data-testid=health]")
    page.set_input_files("input[aria-label='arquivo de imagem']", str(IMG))
    page.click("text=Contar")
    page.wait_for_selector("[data-testid=visible]", timeout=120000)
    n = int(page.inner_text("[data-testid=visible]"))
    assert n > 0
    body = page.inner_text("[data-testid=image-result]")
    assert "Não é contagem de rebanho" in body or "VISÍVEIS" in body.upper() or "Visíveis" in body
    assert page.locator("img[alt='imagem anotada']").is_visible()
    page.screenshot(path=str(SHOTS / "ui_imagem.png"), full_page=True)


def test_b_video_linha_contagem_e_tres_numeros_separados(page):
    page.goto(BASE)
    page.click("role=tab[name='Vídeo']")
    page.set_input_files("input[aria-label='arquivo de vídeo']", str(VID))
    page.click("text=Enviar e processar")
    page.wait_for_selector("[data-testid=canvas]", timeout=300000)
    assert page.inner_text("[data-testid=status]") == "done"
    page.click("text=Traçar linha")
    box = page.locator("[data-testid=canvas]").bounding_box()
    page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.02)
    page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.98)
    page.click("text=Contar passagens")
    page.wait_for_selector("[data-testid=three-counts]", timeout=60000)
    txt = page.inner_text("[data-testid=three-counts]")
    assert "Visíveis agora" in txt and "Passagens (acumulado)" in txt and "não é população" in txt
    assert "NÃO é o tamanho do rebanho" in txt
    assert page.locator("[data-testid=tot-linha1]").count() == 1
    assert page.locator("table tbody tr").count() >= 1 or "Nenhum evento" in page.inner_text("[data-testid=count-result]")
    page.screenshot(path=str(SHOTS / "ui_video_contagem.png"), full_page=True)
    page.click("text=Gerar vídeo anotado")
    page.wait_for_selector("[data-testid=annotated]", state="attached", timeout=120000)
    with page.expect_download() as d:
        page.click("text=Eventos (CSV)")
    assert Path(d.value.path()).read_text().startswith("event_id,type")


def test_c_historico_e_limites(page):
    page.goto(BASE)
    page.click("role=tab[name='Histórico']")
    page.wait_for_selector("tbody tr")
    assert page.locator("tbody tr").count() >= 2
    page.screenshot(path=str(SHOTS / "ui_historico.png"), full_page=True)
    page.click("role=tab[name='Sobre e limites']")
    page.wait_for_selector("text=Câmera móvel")
    t = page.inner_text("main")
    assert "não validado" in t and "Câmera móvel" in t
    page.screenshot(path=str(SHOTS / "ui_limites.png"), full_page=True)
