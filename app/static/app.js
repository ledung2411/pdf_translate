"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const store = {
  get(key, def) {
    try { const v = localStorage.getItem(key); return v === null ? def : v; } catch { return def; }
  },
  set(key, val) {
    try { localStorage.setItem(key, val); } catch { /* bỏ qua */ }
  },
};

// ---------------- Tabs ----------------
$$(".tab").forEach((btn) =>
  btn.addEventListener("click", () => {
    $$(".tab").forEach((b) => b.classList.toggle("active", b === btn));
    $$(".panel").forEach((p) => p.classList.toggle("hidden", p.id !== `tab-${btn.dataset.tab}`));
    if (btn.dataset.tab === "glossary") loadGlossary();
  })
);

// ---------------- Engines ----------------
let engines = [];
async function loadEngines() {
  const sel = $("#engine");
  try {
    const res = await fetch("api/engines");
    const data = await res.json();
    engines = data.engines;
    const saved = store.get("engine", data.default);
    sel.innerHTML = "";
    for (const e of engines) {
      const opt = document.createElement("option");
      opt.value = e.key;
      opt.textContent = e.label + (e.available ? "" : " — chưa sẵn sàng");
      sel.appendChild(opt);
    }
    sel.value = engines.some((e) => e.key === saved) ? saved : data.default;
    showEngineDesc();
  } catch (err) {
    sel.innerHTML = '<option value="">Không tải được danh sách mô hình</option>';
  }
}
function showEngineDesc() {
  const e = engines.find((x) => x.key === $("#engine").value);
  const el = $("#engine-desc");
  if (!e) { el.textContent = ""; return; }
  el.textContent = e.available ? e.description : `${e.description} ⚠ ${e.reason}`;
  el.style.color = e.available ? "" : "var(--error)";
}
$("#engine").addEventListener("change", () => {
  store.set("engine", $("#engine").value);
  showEngineDesc();
});

// ---------------- File chọn / kéo thả ----------------
let selectedFile = null;
const drop = $("#drop");
function setFile(f) {
  if (!f) return;
  if (!/\.pdf$/i.test(f.name) && f.type !== "application/pdf") {
    $("#file-name").textContent = "File không phải PDF";
    $("#file-hint").textContent = "Vui lòng chọn file .pdf";
    selectedFile = null;
    $("#submit").disabled = true;
    return;
  }
  selectedFile = f;
  resetTerms();
  $("#file-name").textContent = f.name;
  $("#file-hint").textContent = `${(f.size / 1024 / 1024).toFixed(2)} MB — bấm để chọn file khác`;
  $("#submit").disabled = false;
}
$("#file").addEventListener("change", (e) => setFile(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); })
);
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); })
);
drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

// Ghi nhớ tuỳ chọn
for (const id of ["skip-refs", "use-gloss", "bilingual"]) {
  const el = $("#" + id);
  el.checked = store.get("opt-" + id, el.checked ? "1" : "0") === "1";
  el.addEventListener("change", () => store.set("opt-" + id, el.checked ? "1" : "0"));
}

// ---------------- Thuật ngữ của bài ----------------
const KIND_LABEL = { definition: "viết tắt", acronym: "tên riêng", phrase: "cụm từ" };

function resetTerms() {
  $("#terms-table tbody").innerHTML = "";
  $("#terms-wrap").classList.add("hidden");
  $("#save-terms").classList.add("hidden");
  $("#terms-status").textContent = "";
  $("#analyze-terms").disabled = !selectedFile;
}

function termGlossaryLines() {
  return $$("#terms-table tbody tr")
    .map((tr) => [tr.dataset.term, tr.querySelector("input").value.trim()])
    .filter(([en, vi]) => en && vi)
    .map(([en, vi]) => `${en} = ${vi}`);
}

$("#analyze-terms").addEventListener("click", async () => {
  if (!selectedFile) return;
  const btn = $("#analyze-terms");
  btn.disabled = true;
  $("#terms-status").textContent = "Đang phân tích...";
  const fd = new FormData();
  fd.append("file", selectedFile);
  if ($("#page-from").value) fd.append("page_from", $("#page-from").value);
  if ($("#page-to").value) fd.append("page_to", $("#page-to").value);
  try {
    const res = await fetch("api/terms", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Lỗi phân tích");
    const tbody = $("#terms-table tbody");
    tbody.innerHTML = "";
    for (const t of data.terms) {
      const tr = document.createElement("tr");
      tr.dataset.term = t.term;
      const a = document.createElement("td");
      a.textContent = t.term + (t.expansion ? ` (${t.expansion})` : "");
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = KIND_LABEL[t.kind] || t.kind;
      a.appendChild(badge);
      const n = document.createElement("td");
      n.className = "num";
      n.textContent = t.count;
      const c = document.createElement("td");
      const input = document.createElement("input");
      input.value = t.suggestion || "";
      input.placeholder = t.kind === "phrase" || t.kind === "definition" ? "để trống = mô hình tự dịch" : "";
      c.appendChild(input);
      tr.append(a, n, c);
      tbody.appendChild(tr);
    }
    $("#terms-wrap").classList.toggle("hidden", !data.terms.length);
    $("#save-terms").classList.toggle("hidden", !data.terms.length);
    $("#terms-status").textContent = data.terms.length
      ? `Tìm thấy ${data.terms.length} thuật ngữ.`
      : "Không tìm thấy thuật ngữ lặp lại.";
  } catch (err) {
    $("#terms-status").textContent = "Lỗi: " + err.message;
  } finally {
    btn.disabled = false;
  }
});

$("#save-terms").addEventListener("click", () => {
  const lines = termGlossaryLines().filter((l) => {
    const [en, vi] = l.split(" = ");
    return en !== vi; // bỏ các mục giữ nguyên (tên riêng)
  });
  if (!lines.length) return;
  const g = $("#custom-gloss");
  g.value = (g.value.trim() ? g.value.trim() + "\n" : "") + lines.join("\n");
  store.set("custom-glossary", g.value);
  $("#terms-status").textContent = `Đã lưu ${lines.length} mục vào "Thuật ngữ của tôi".`;
});

// ---------------- Gửi job ----------------
let pollTimer = null;
let currentJob = null;

$("#pdf-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!selectedFile) return;
  const fd = new FormData();
  fd.append("file", selectedFile);
  fd.append("engine", $("#engine").value);
  if ($("#page-from").value) fd.append("page_from", $("#page-from").value);
  if ($("#page-to").value) fd.append("page_to", $("#page-to").value);
  fd.append("skip_references", $("#skip-refs").checked);
  fd.append("use_glossary", $("#use-gloss").checked);
  fd.append("make_bilingual", $("#bilingual").checked);
  fd.append("custom_glossary", $("#custom-gloss").value + "\n" + termGlossaryLines().join("\n"));

  $("#submit").disabled = true;
  showJob({ filename: selectedFile.name, status: "queued", progress: 0, message: "Đang tải file lên..." });
  try {
    const res = await fetch("api/jobs", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Tải lên thất bại");
    currentJob = data.id;
    store.set("last-job", data.id);
    poll();
  } catch (err) {
    showJob({ filename: selectedFile.name, status: "error", progress: 0, error: err.message });
    $("#submit").disabled = false;
  }
});

async function poll() {
  clearTimeout(pollTimer);
  if (!currentJob) return;
  try {
    const res = await fetch(`api/jobs/${currentJob}`);
    if (res.status === 404) { currentJob = null; $("#job").classList.add("hidden"); return; }
    const job = await res.json();
    showJob(job);
    if (job.status === "queued" || job.status === "running") {
      pollTimer = setTimeout(poll, 1000);
    } else {
      $("#submit").disabled = !selectedFile;
    }
  } catch {
    pollTimer = setTimeout(poll, 3000);
  }
}

function fmtTime(s) {
  if (s == null) return "";
  const m = Math.floor(s / 60), r = Math.round(s % 60);
  return m ? `${m} phút ${r} giây` : `${r} giây`;
}

function showJob(job) {
  $("#job").classList.remove("hidden");
  $("#job-file").textContent = job.filename;
  const pct = Math.round((job.progress || 0) * 100);
  $("#job-pct").textContent = `${pct}%`;
  $("#job-bar").style.width = `${pct}%`;
  let msg = job.message || "";
  if (job.status === "queued" && job.queue_position > 1) msg = `Đang chờ — vị trí ${job.queue_position} trong hàng đợi`;
  if (job.elapsed != null) msg += ` · ${fmtTime(job.elapsed)}`;
  $("#job-msg").textContent = msg;

  const err = $("#job-error");
  err.classList.toggle("hidden", job.status !== "error");
  if (job.status === "error") {
    err.textContent = (job.error || "Có lỗi xảy ra.") +
      "\n\nĐể chẩn đoán, chạy trong thư mục dự án: python -m app.doctor --try vinai";
  }

  const dl = $("#downloads");
  const done = job.status === "done";
  dl.classList.toggle("hidden", !done);
  $("#preview").classList.toggle("hidden", !done);
  if (done && job.id) {
    const kinds = job.downloads || [];
    for (const k of ["translated", "bilingual", "markdown"]) {
      const a = $(`#dl-${k}`);
      a.classList.toggle("hidden", !kinds.includes(k));
      a.href = `api/jobs/${job.id}/download/${k}`;
    }
    $$(".seg-btn").forEach((b) => b.classList.toggle("hidden", !kinds.includes(b.dataset.kind)));
    setPreview(store.get("preview-kind", "translated"), job);
  }
}

function setPreview(kind, job) {
  const id = (job && job.id) || currentJob;
  if (!id) return;
  const kinds = (job && job.downloads) || [];
  if (kinds.length && !kinds.includes(kind)) kind = "translated";
  $$(".seg-btn").forEach((b) => b.classList.toggle("active", b.dataset.kind === kind));
  const src = `api/jobs/${id}/download/${kind}?inline=true`;
  const frame = $("#preview-frame");
  if (frame.dataset.src !== src) { frame.src = src; frame.dataset.src = src; }
}
$$(".seg-btn").forEach((b) =>
  b.addEventListener("click", () => { store.set("preview-kind", b.dataset.kind); setPreview(b.dataset.kind); })
);

// ---------------- Dịch đoạn văn ----------------
$("#translate-text").addEventListener("click", async () => {
  const text = $("#src-text").value.trim();
  if (!text) return;
  const btn = $("#translate-text");
  btn.disabled = true;
  $("#text-status").textContent = "Đang dịch... (lần đầu có thể lâu do phải nạp mô hình)";
  const t0 = performance.now();
  try {
    const res = await fetch("api/translate-text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        engine: $("#engine").value,
        use_glossary: $("#use-gloss").checked,
        custom_glossary: $("#custom-gloss").value,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Lỗi dịch");
    $("#dst-text").value = data.translation;
    $("#text-status").textContent = `Xong trong ${((performance.now() - t0) / 1000).toFixed(1)} giây`;
  } catch (err) {
    $("#text-status").textContent = "Lỗi: " + err.message;
  } finally {
    btn.disabled = false;
  }
});
$("#copy-text").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("#dst-text").value); $("#text-status").textContent = "Đã sao chép"; }
  catch { $("#dst-text").select(); }
});

// ---------------- Thuật ngữ ----------------
const gloss = $("#custom-gloss");
gloss.value = store.get("custom-glossary", "");
gloss.addEventListener("input", () => store.set("custom-glossary", gloss.value));

let glossLoaded = false;
async function loadGlossary() {
  if (glossLoaded) return;
  glossLoaded = true;
  const res = await fetch("api/glossary");
  const data = await res.json();
  const tbody = $("#gloss-table tbody");
  const frag = document.createDocumentFragment();
  for (const [en, vi] of Object.entries(data.entries)) {
    const tr = document.createElement("tr");
    const a = document.createElement("td"); a.textContent = en;
    const b = document.createElement("td"); b.textContent = vi;
    tr.append(a, b);
    frag.appendChild(tr);
  }
  tbody.appendChild(frag);
}
$("#gloss-search").addEventListener("input", (e) => {
  const q = e.target.value.trim().toLowerCase();
  $$("#gloss-table tbody tr").forEach((tr) => {
    tr.classList.toggle("hidden", q && !tr.textContent.toLowerCase().includes(q));
  });
});

// ---------------- Khởi động ----------------
loadEngines();
const last = store.get("last-job", null);
if (last) { currentJob = last; poll(); }
