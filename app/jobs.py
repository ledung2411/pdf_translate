"""Hàng đợi công việc dịch chạy nền (một worker để không tranh tài nguyên mô hình)."""

from __future__ import annotations

import json
import logging
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import settings
from .ir import Document
from .pdf_processor import IR_FILENAME, PdfError, TranslateOptions, is_doubtful, translate_pdf
from .translators import TranslatorError

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    filename: str
    options: TranslateOptions
    status: str = "queued"  # queued | running | done | error
    progress: float = 0.0
    message: str = "Đang chờ trong hàng đợi..."
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)  # thống kê chất lượng sau khi dịch

    @property
    def dir(self) -> Path:
        return settings.data_dir / "jobs" / self.id

    @property
    def input_path(self) -> Path:
        return self.dir / "input.pdf"

    def public(self) -> dict:
        d = asdict(self)
        d.pop("outputs")
        d["options"] = {k: v for k, v in d["options"].items() if k != "custom_glossary"}
        d["downloads"] = sorted(self.outputs)
        if self.started:
            d["elapsed"] = round((self.finished or time.time()) - self.started, 1)
        return d


def quality_summary(ir_path: Path) -> dict:
    try:
        ir = Document.load(ir_path)
    except Exception:  # noqa: BLE001
        return {}
    texts = [r for r in ir.regions if r.translation is not None]
    scores = [r.score for r in texts if r.score is not None]
    return {
        "translated": len(texts),
        "doubtful": sum(1 for r in texts if is_doubtful(r)),
        "post_edited": sum(1 for r in texts if "post_edited" in r.flags),
        "edited": sum(1 for r in texts if "edited" in r.flags),
        "avg_score": round(sum(scores) / len(scores), 3) if scores else None,
    }


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._q: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, name="translate-worker", daemon=True)
            self._worker.start()

    def submit(self, filename: str, data: bytes, options: TranslateOptions) -> Job:
        self._cleanup()
        job = Job(id=uuid.uuid4().hex[:12], filename=filename, options=options)
        job.dir.mkdir(parents=True, exist_ok=True)
        job.input_path.write_bytes(data)
        meta = {"filename": filename, "created": job.created, "options": asdict(options)}
        (job.dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        with self._lock:
            self.jobs[job.id] = job
        self._q.put(job.id)
        self.start()
        return job

    def get(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job is None and re.fullmatch(r"[0-9a-f]{12}", job_id):
            job = self._restore(job_id)
        return job

    def _restore(self, job_id: str) -> Job | None:
        """Nạp lại job đã hoàn tất từ đĩa (sau khi khởi động lại server)."""
        d = settings.data_dir / "jobs" / job_id
        meta_path, ir_path = d / "meta.json", d / IR_FILENAME
        if not (meta_path.exists() and ir_path.exists() and (d / "input.pdf").exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            opts = TranslateOptions(**meta.get("options", {}))
        except Exception:  # noqa: BLE001
            return None
        job = Job(id=job_id, filename=meta.get("filename", "document.pdf"), options=opts,
                  status="done", progress=1.0, message="Hoàn tất", created=meta.get("created", time.time()))
        stem = re.sub(r"[^\w\-]+", "_", Path("input.pdf").stem)
        for kind, suffix in (("translated", "_vi.pdf"), ("bilingual", "_song_ngu.pdf"), ("markdown", "_vi.md")):
            f = d / f"{stem}{suffix}"
            if f.exists():
                job.outputs[kind] = str(f)
        job.summary = quality_summary(ir_path)
        job.finished = ir_path.stat().st_mtime
        with self._lock:
            self.jobs[job_id] = job
        return job

    def queue_position(self, job: Job) -> int:
        waiting = sorted((j for j in self.jobs.values() if j.status == "queued"), key=lambda j: j.created)
        return next((i + 1 for i, j in enumerate(waiting) if j.id == job.id), 0)

    def _run(self) -> None:
        while True:
            job_id = self._q.get()
            job = self.jobs.get(job_id)
            if job is None:
                continue
            job.status, job.started = "running", time.time()
            job.message = "Đang chuẩn bị..."

            def progress(frac: float, msg: str, job: Job = job) -> None:
                job.progress, job.message = round(frac, 4), msg

            try:
                outputs = translate_pdf(job.input_path, job.dir, job.options, progress)
                job.outputs = {k: str(v) for k, v in outputs.items()}
                job.summary = quality_summary(job.dir / IR_FILENAME)
                job.status, job.progress, job.message = "done", 1.0, "Hoàn tất"
            except (PdfError, TranslatorError) as e:
                job.status, job.error, job.message = "error", str(e), "Lỗi"
            except Exception as e:  # noqa: BLE001
                log.error("Job %s lỗi:\n%s", job.id, traceback.format_exc())
                job.status, job.error, job.message = "error", f"Lỗi không mong muốn: {e}", "Lỗi"
            finally:
                job.finished = time.time()

    def _cleanup(self) -> None:
        cutoff = time.time() - settings.job_ttl_hours * 3600
        with self._lock:
            for jid, job in list(self.jobs.items()):
                if job.status in ("done", "error") and (job.finished or job.created) < cutoff:
                    shutil.rmtree(job.dir, ignore_errors=True)
                    del self.jobs[jid]


manager = JobManager()
