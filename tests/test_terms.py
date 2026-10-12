import pymupdf
from fastapi.testclient import TestClient

from app.ir import Document, Page, Region
from app.main import app
from app.terms import _match_expansion, extract_terms

from .make_sample_pdf import make


def _doc(*texts: str) -> Document:
    regions = [
        Region(page_index=0, bbox=pymupdf.Rect(0, 0, 1, 1), text=t, kind="text", font_size=10,
               bold=False, italic=False, serif=True, color=0, align="left")
        for t in texts
    ]
    return Document(source="x.pdf", pages=[Page(index=0, number=1, width=595, height=842, regions=regions)])


def test_match_expansion():
    assert _match_expansion("Reconstruction from diffusion MRI", "dMRI") == "diffusion MRI"
    assert _match_expansion("we generate pseudo-ground-truth", "pGT") == "pseudo-ground-truth"
    assert _match_expansion("compared with T1-weighted", "T1w") == "T1-weighted"
    assert _match_expansion("the signed distance function", "SDF") == "signed distance function"
    assert _match_expansion("the Desikan–Killiany", "DK") == "Desikan–Killiany"
    assert _match_expansion("this is unrelated text", "SDF") is None


def test_extract_terms_kinds():
    body = (
        "Cortical surface reconstruction from diffusion MRI (dMRI) is hard. We train DDSurfer on dMRI data. "
        "The white matter (WM) and pial surfaces are reconstructed. DDSurfer improves cortical surface "
        "reconstruction. The pial surface and the pial surfaces of WM. Cortical surface reconstruction again. "
        "FIGURE 1 shows the figure."
    )
    terms = {t.term: t for t in extract_terms(_doc(body))}
    assert terms["diffusion MRI"].kind == "definition" and terms["diffusion MRI"].expansion == "dMRI"
    assert terms["white matter"].expansion == "WM"
    assert terms["DDSurfer"].kind == "acronym" and terms["DDSurfer"].suggestion == "DDSurfer"
    assert terms["cortical surface reconstruction"].kind == "phrase"
    assert terms["pial surface"].count == 3  # gộp số nhiều
    assert "FIGURE" not in terms


def test_terms_api(tmp_path):
    p = tmp_path / "s.pdf"
    make(str(p))
    with TestClient(app) as client:
        r = client.post("/api/terms", files={"file": ("s.pdf", p.read_bytes(), "application/pdf")})
        assert r.status_code == 200, r.text
        terms = r.json()["terms"]
        assert any(t["term"] == "Transformer" or t["kind"] == "phrase" for t in terms)
        assert client.post("/api/terms", files={"file": ("x.pdf", b"nope", "application/pdf")}).status_code == 400
