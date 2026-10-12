# Kiến trúc

```
PDF ─▶ ① Tiếp nhận ─▶ ② Phân tích bố cục ─▶ ③ DocIR (document.json)
                                                │
   ┌────────────────────────────────────────────┘
   ▼
④ Thuật ngữ ─▶ ⑤ Tách câu ─▶ ⑥ Dịch ─▶ ⑦ Chấm điểm ─▶ ⑧ Dựng (PDF / song ngữ / MD) ─▶ ⑨ Sửa tay
                                     ▲          │ câu điểm thấp            tô vàng đoạn nghi ngờ      │
                                     │          └─▶ LLM hiệu đính                                    │
                                     └──────────── bộ nhớ dịch (TM) ◀──────────────────────────────┘
  mặc định       nối dòng      TM/cache   lặp, rác,       xóa chữ gốc,
  + của bài      hoa/thường    NMT ct2    độ dài          ghi bản dịch,
  + của bạn      span công     LLM        → dịch lại      giữ hình & nét vẽ
                 thức                     → giữ tiếng Anh
```

## Nguyên tắc

1. **DocIR là nguồn dữ liệu duy nhất** (`app/ir.py`): phân tích tạo `Region`, tầng dịch điền `translation`,
   tầng dựng chỉ đọc IR. Sửa bản dịch → dựng lại PDF bằng `render_outputs()` mà không dịch lại.
2. **Không bao giờ in ra rác**: mọi câu qua `is_degenerate()`; hỏng → dịch lại từng vế → vẫn hỏng thì giữ
   tiếng Anh và gắn cờ `kept_english`.
3. **Chiến lược theo loại mô hình**: mô hình dịch máy (OPUS/VinAI/NLLB) không nhận placeholder `{n}` (chúng
   sinh chữ rác "Name"/"Comment"); LLM nhận placeholder + thuật ngữ trong prompt.
4. **Offline**: mô hình tải một lần, nạp từ thư mục trên máy; CTranslate2 int8 không cần torch khi chạy.
5. **Kiểm thử bằng PDF thật**: `tests/test_corpus.py` + `tests/corpus/local/`.

## Mã nguồn

| File | Vai trò |
|------|---------|
| `app/ir.py` | DocIR: `Document` / `Page` / `Region`, lưu/đọc JSON |
| `app/pdf_processor.py` | `analyze_pdf` (trích & phân loại vùng, tách cột), `translate_document`, `render_outputs` |
| `app/scientific.py` | Bảo vệ ký hiệu, glossary, tách câu, nhận diện tiêu đề, phân loại khối |
| `app/terms.py` | Tự trích thuật ngữ của bài |
| `app/quality.py` | Chấm điểm bản dịch từng câu (QE), phát hiện bản dịch hỏng |
| `app/tm.py` | Bộ nhớ dịch: đoạn đã sửa tay được dùng lại ở lần dịch sau |
| `app/static/editor.*` | Trang sửa bản dịch song ngữ |
| `app/pipeline.py` | `TextTranslator`: chuẩn hóa, cache, dịch theo lô, kiểm tra & sửa bản dịch hỏng |
| `app/translators.py` | Engine: `HFSeq2SeqTranslator` (backend ct2 / hf), `LLMTranslator`, cache SQLite |
| `app/main.py`, `app/jobs.py` | Web API + hàng đợi công việc |
| `app/doctor.py`, `app/inspect.py` | Chẩn đoán môi trường, xem phân loại khối của một PDF |

## Lộ trình

- [x] **P1** — DocIR; CTranslate2 int8; tự trích thuật ngữ; bộ PDF kiểm thử nhiều nhà xuất bản
- [x] **P2** — Chấm điểm chất lượng từng câu (luật + tùy chọn dịch ngược chrF) + LLM sửa có chọn lọc câu điểm
  thấp; tô vàng câu nghi ngờ; trang biên tập song ngữ + bộ nhớ dịch (TM)
- [ ] **P3** — Nhận diện bố cục bằng mô hình (DocLayout-YOLO); giữ công thức dạng vector; OCR; xuất DOCX/HTML;
  bản cài đặt 1 click cho Windows
