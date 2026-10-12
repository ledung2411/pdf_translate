"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const jobId = new URLSearchParams(location.search).get("job");
const LABELS = { translated: "PDF tiếng Việt", bilingual: "PDF song ngữ", docx: "Word (.docx)", markdown: "Văn bản (.md)" };

let regions = [];
let filter = "doubtful";
const dirty = new Map(); // id → bản dịch mới

function scoreBadge(r) {
  if (r.flags.includes("edited")) return '<span class="pill ok">đã sửa</span>';
  if (r.flags.includes("tm")) return '<span class="pill ok">từ bộ nhớ dịch</span>';
  if (r.score == null) return "";
  const pct = Math.round(r.score * 100);
  const cls = r.doubtful ? "warn" : "ok";
  return `<span class="pill ${cls}">điểm ${pct}</span>` +
    (r.flags.includes("post_edited") ? '<span class="pill">LLM đã sửa</span>' : "");
}

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function autosize(t) {
  t.style.height = "auto";
  t.style.height = t.scrollHeight + 2 + "px";
}

function visible(r) {
  if (filter === "doubtful" && !(r.doubtful && !dirty.has(r.id))) return false;
  if (filter === "edited" && !(r.flags.includes("edited") || dirty.has(r.id))) return false;
  const q = $("#search").value.trim().toLowerCase();
  if (q && !(r.source.toLowerCase().includes(q) || (r.translation || "").toLowerCase().includes(q))) return false;
  return true;
}

function render() {
  const list = $("#list");
  list.innerHTML = "";
  const shown = regions.filter(visible);
  for (const r of shown) {
    const card = document.createElement("div");
    card.className = "card seg-card" + (r.doubtful ? " doubtful" : "");
    card.innerHTML = `
      <div class="seg-head"><span class="muted">Trang ${r.page}</span>${scoreBadge(r)}
        ${r.notes.length ? `<span class="muted notes">${escapeHtml(r.notes.join(" · "))}</span>` : ""}</div>
      <div class="grid two">
        <div class="src">${escapeHtml(r.source)}</div>
        <div><textarea rows="3"></textarea>
          <div class="row end actions">
            ${r.doubtful ? '<button class="btn small accept" type="button">Bản dịch đúng, bỏ tô vàng</button>' : ""}
          </div></div>
      </div>`;
    const ta = card.querySelector("textarea");
    ta.value = dirty.has(r.id) ? dirty.get(r.id) : r.translation || "";
    ta.addEventListener("input", () => {
      autosize(ta);
      if (ta.value !== (r.translation || "")) dirty.set(r.id, ta.value);
      else dirty.delete(r.id);
      updateState();
    });
    const accept = card.querySelector(".accept");
    if (accept) accept.addEventListener("click", () => {
      dirty.set(r.id, ta.value);
      accept.disabled = true;
      accept.textContent = "Đã đánh dấu";
      updateState();
    });
    list.appendChild(card);
    autosize(ta);
  }
  $("#empty").classList.toggle("hidden", shown.length > 0);
}

function counts() {
  $("#n-all").textContent = `(${regions.length})`;
  $("#n-doubtful").textContent = `(${regions.filter((r) => r.doubtful).length})`;
  $("#n-edited").textContent = `(${regions.filter((r) => r.flags.includes("edited")).length})`;
}

function updateState() {
  $("#save").disabled = dirty.size === 0;
  $("#save-state").textContent = dirty.size ? `${dirty.size} đoạn chưa lưu` : "Chưa có thay đổi";
}

function showDownloads(job) {
  const box = $("#downloads");
  box.innerHTML = "";
  for (const k of job.downloads || []) {
    const a = document.createElement("a");
    a.className = "btn small";
    a.href = `api/jobs/${job.id}/download/${k}?t=${Date.now()}`;
    a.textContent = LABELS[k] || k;
    box.appendChild(a);
  }
}

async function load() {
  if (!jobId) { $("#doc-info").textContent = "Thiếu mã công việc (?job=...)."; return; }
  const res = await fetch(`api/jobs/${jobId}/document`);
  const data = await res.json();
  if (!res.ok) { $("#doc-info").textContent = data.detail || "Không tải được bản dịch."; return; }
  regions = data.regions;
  const sm = data.job.summary || {};
  $("#doc-info").textContent = `${data.job.filename} · ${regions.length} đoạn · mô hình ${data.engine}` +
    (sm.avg_score != null ? ` · điểm trung bình ${Math.round(sm.avg_score * 100)}/100` : "");
  if (!regions.some((r) => r.doubtful)) setFilter("all");
  counts();
  render();
  showDownloads(data.job);
}

function setFilter(f) {
  filter = f;
  $$(".seg-btn").forEach((b) => b.classList.toggle("active", b.dataset.filter === f));
  render();
}
$$(".seg-btn").forEach((b) => b.addEventListener("click", () => setFilter(b.dataset.filter)));
$("#search").addEventListener("input", render);

$("#save").addEventListener("click", async () => {
  const btn = $("#save");
  btn.disabled = true;
  $("#save-state").textContent = "Đang lưu và dựng lại PDF...";
  const edits = [...dirty].map(([id, translation]) => ({ id, translation }));
  try {
    const res = await fetch(`api/jobs/${jobId}/document`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ edits, save_tm: true }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Lưu thất bại");
    dirty.clear();
    await load();
    $("#save-state").textContent = `Đã lưu ${data.saved} đoạn, PDF đã được dựng lại.`;
  } catch (err) {
    $("#save-state").textContent = "Lỗi: " + err.message;
    btn.disabled = false;
  }
});

window.addEventListener("beforeunload", (e) => { if (dirty.size) { e.preventDefault(); e.returnValue = ""; } });
load();
