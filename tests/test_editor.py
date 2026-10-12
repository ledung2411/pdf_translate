import time

import pymupdf
from fastapi.testclient import TestClient

from app.config import settings
from app.jobs import manager
from app.main import app
from app.tm import get_tm

from .make_sample_pdf import make


def _run_job(client, pdf_bytes):
    r = client.post("/api/jobs", files={"file": ("paper.pdf", pdf_bytes, "application/pdf")},
                    data={"engine": "mock", "post_edit": "false"})
    job_id = r.json()["id"]
    for _ in range(200):
        st = client.get(f"/api/jobs/{job_id}").json()
        if st["status"] in ("done", "error"):
            return st
        time.sleep(0.05)
    raise AssertionError("job quá lâu")


def test_edit_rerender_tm_and_restore(tmp_path):
    src = tmp_path / "paper.pdf"
    make(str(src))
    with TestClient(app) as client:
        st = _run_job(client, src.read_bytes())
        assert st["status"] == "done" and st["summary"]["doubtful"] > 0
        job_id = st["id"]

        doc = client.get(f"/api/jobs/{job_id}/document").json()
        target = next(r for r in doc["regions"] if r["doubtful"])  # engine giả lập → có đoạn điểm thấp
        names = next(r for r in doc["regions"] if r["source"].startswith("Jane Doe"))
        assert "Jane Doe¹" in names["translation"]  # tên riêng giữ hoa, chỉ số trên giữ nguyên

        fixed = "Chúng tôi trình bày cơ chế chú ý thưa cho văn bản dài^{2}."
        r = client.put(f"/api/jobs/{job_id}/document", json={"edits": [{"id": target["id"], "translation": fixed}]})
        assert r.status_code == 200 and r.json()["saved"] == 1
        assert r.json()["job"]["summary"]["edited"] == 1
        assert r.json()["job"]["summary"]["doubtful"] == st["summary"]["doubtful"] - 1

        pdf = client.get(f"/api/jobs/{job_id}/download/translated").content
        page = pymupdf.open("pdf", pdf)[0]
        assert "Chúng tôi trình bày cơ chế chú ý thưa" in page.get_text()
        titles = [a.info["title"] for a in page.annots()]
        assert len(titles) == st["summary"]["doubtful"] - 1  # đoạn đã sửa không còn tô vàng

        # Bộ nhớ dịch: lần dịch sau dùng ngay bản đã sửa
        assert get_tm().lookup([target["source"]])
        st2 = _run_job(client, src.read_bytes())
        doc2 = client.get(f"/api/jobs/{st2['id']}/document").json()
        again = next(r for r in doc2["regions"] if r["source"] == target["source"])
        assert again["translation"] == fixed and "tm" in again["flags"]

        # Khởi động lại server: job đã xong vẫn mở lại được từ đĩa
        manager.jobs.pop(job_id)
        restored = client.get(f"/api/jobs/{job_id}/document")
        assert restored.status_code == 200 and restored.json()["job"]["status"] == "done"
        assert client.get(f"/api/jobs/{job_id}/download/translated").status_code == 200

        assert client.put(f"/api/jobs/{job_id}/document", json={"edits": [{"id": "x", "translation": "y"}]}).status_code == 400
        assert client.get("/api/jobs/ffffffffffff/document").status_code == 404
    assert (settings.data_dir / "tm.sqlite").exists()


def test_superscript_roundtrip():
    from app.main import _from_editable, _to_editable
    from app.pipeline import SUP_END, SUP_START

    internal = f"Diện tích 5 cm{SUP_START}2{SUP_END} [3]"
    assert _to_editable(internal) == "Diện tích 5 cm^{2} [3]"
    assert _from_editable("Diện tích 5 cm^{2} [3]") == internal
