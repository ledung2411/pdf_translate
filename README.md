# Dịch PDF khoa học Anh → Việt

Web tool dịch tài liệu khoa học dạng PDF (bài báo, luận văn, giáo trình) từ tiếng Anh sang tiếng Việt,
**chạy hoàn toàn trên máy của bạn** bằng các mô hình mã nguồn mở. Không dùng API trả phí, không gửi tài liệu
lên dịch vụ bên ngoài.

## Tính năng

- **Giữ nguyên bố cục**: bản dịch được ghi đúng vị trí đoạn văn gốc; hình ảnh, hình vẽ, bảng số liệu, công thức
  dạng khối giữ nguyên. Tự thu nhỏ cỡ chữ khi bản tiếng Việt dài hơn.
- **Chuyên cho tài liệu khoa học**:
  - Bảo vệ công thức toán (nhận diện font LaTeX: CMMI, CMSY, MSBM…), ký hiệu Hy Lạp, biểu thức như
    `p < 0.05`, `y = Wx + b`, trích dẫn `[1, 3]`, URL, DOI, arXiv ID, email, tên hàm/biến trong mã nguồn.
  - Bảng thuật ngữ gần 200 mục (Toán, Thống kê, AI/ML, Vật lý, Hóa, Y sinh) để dịch nhất quán; tự thêm thuật ngữ riêng.
  - **Tự trích thuật ngữ của bài** trước khi dịch: định nghĩa viết tắt (`diffusion MRI (dMRI)`), tên phương
    pháp/phần mềm lặp lại, cụm chuyên ngành (`cortical surface reconstruction`) — duyệt và điền bản dịch ngay trên giao diện.
  - Dịch chuẩn tiêu đề mục (Abstract → Tóm tắt, Related Work → Các công trình liên quan…) và nhãn
    (Fig. 3 → Hình 3, Table 2 → Bảng 2, Eq. (1) → Phương trình (1), et al. → và cộng sự).
  - Giữ nguyên mục **Tài liệu tham khảo** (tùy chọn), bỏ qua bảng số liệu, mã nguồn, watermark xoay của arXiv.
  - Tách câu thông minh (không cắt sai ở `et al.`, `e.g.`, `Fig.`…), nối lại từ bị ngắt dòng bằng gạch nối.
- **4 dạng kết quả**: PDF tiếng Việt, PDF song ngữ (gốc | dịch cạnh nhau), **Word (.docx)** để chỉnh sửa tiếp
  (đúng thứ tự đọc 2 cột, công thức/bảng/hình là ảnh cắt từ bản gốc), văn bản Markdown.
- **PDF scan**: tự nhận dạng chữ (OCR, Tesseract tích hợp sẵn — dữ liệu tiếng Anh ~4 MB tự tải), dịch và ghi
  đè lên ảnh.
- **Nhận diện bố cục bằng mô hình** DocLayout-YOLO (ONNX, CPU): phân biệt đoạn văn / công thức / bảng / hình /
  đầu-chân trang chính xác hơn; tự quay về luật nếu không có mô hình.
- **Giữ công thức nguyên bản**: công thức đứng riêng không bị xóa/vẽ lại; công thức nội dòng in nghiêng.
- **Nhanh trên CPU**: mô hình chạy bằng CTranslate2 int8 (nhanh hơn, ít RAM, không cần torch khi dịch).
- **Không in ra rác**: câu dịch bị lặp/hỏng được dịch lại từng vế; vẫn hỏng thì giữ tiếng Anh và gắn cờ.
- **Chấm điểm từng câu & tô vàng đoạn nghi ngờ**: mất số liệu/trích dẫn, sót tiếng Anh, sai thuật ngữ, độ dài
  bất thường… Đoạn điểm thấp được tô vàng trong PDF (chú thích, ẩn được khi in) kèm lý do.
- **LLM cục bộ sửa có chọn lọc**: nếu Ollama/llama.cpp/LM Studio đang chạy, chỉ những câu điểm thấp nhất
  (tối đa 30%) được LLM hiệu đính — chất lượng tăng mà không chậm như dịch toàn bộ bằng LLM.
- **Trang sửa bản dịch**: xem song song Anh – Việt, lọc đoạn nghi ngờ, sửa rồi dựng lại PDF ngay; chỗ sửa được
  lưu vào **bộ nhớ dịch** và tự dùng lại ở những lần dịch sau.
- Có ô **dịch nhanh đoạn văn**, bộ nhớ đệm bản dịch (dịch lại tài liệu cũ rất nhanh), hàng đợi công việc,
  thanh tiến độ, xem trước PDF ngay trên trình duyệt.

## Mô hình dịch (đều miễn phí, chạy cục bộ)

| Engine  | Mô hình | Ghi chú |
|---------|---------|---------|
| `vinai` | [vinai/vinai-translate-en2vi-v2](https://huggingface.co/vinai/vinai-translate-en2vi-v2) | Chất lượng Anh→Việt tốt nhất trong nhóm mô hình nhỏ; ~1,5 GB; nên có GPU nhưng CPU vẫn chạy được |
| `opus`  | [Helsinki-NLP/opus-mt-en-vi](https://huggingface.co/Helsinki-NLP/opus-mt-en-vi) | Nhẹ (~300 MB), nhanh trên CPU nhưng dịch văn bản khoa học kém — chỉ để xem nhanh |
| `nllb`  | [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | Giấy phép CC-BY-NC (chỉ dùng phi thương mại) |
| `llm`   | LLM mã nguồn mở tự host (Qwen2.5, Llama 3, Gemma…) qua Ollama / llama.cpp / LM Studio | Dịch theo ngữ cảnh cả đoạn, hiểu thuật ngữ tốt nhất; cần máy đủ mạnh |

Mặc định (`auto`) dùng `vinai`. `opus` nhanh hơn nhưng dịch văn bản khoa học kém, chỉ nên dùng để xem nhanh.
Mô hình Hugging Face chỉ cần tải **một lần** (tự động ở lần dịch đầu tiên hoặc dùng `scripts/download_models.py`),
sau đó chạy offline hoàn toàn.

## Cài trên Windows (1 click)

1. Tải mã nguồn: nút **Code → Download ZIP** trên GitHub (hoặc file `pdf_translate-windows.zip` ở mục
   Releases / Actions), giải nén vào một thư mục, ví dụ `C:\pdf_translate`.
2. Bấm đúp **`windows\install.bat`**. Bộ cài tự: tìm/cài Python 3.12 → tạo môi trường riêng `.venv` →
   cài thư viện → tải và chuyển mô hình VinAI sang CTranslate2 int8 → tạo shortcut **"Dịch PDF khoa học"**
   trên Desktop → mở tool trên trình duyệt. Lần đầu cần Internet và mất khoảng 10–20 phút.
3. Những lần sau: bấm shortcut trên Desktop (hoặc `windows\start.bat`). Cập nhật: `windows\update.bat`.

Nhật ký cài đặt nằm ở `install.log` nếu có lỗi.

## Cài đặt (thủ công / Linux / macOS)

Yêu cầu Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt -r requirements-ml.txt
# torch chỉ cần một lần để chuyển mô hình sang CTranslate2 int8 (nhanh hơn ~3-4 lần trên CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/convert_ct2.py vinai

# (tuỳ chọn) tải trước mô hình để dùng offline
python scripts/download_models.py vinai opus
```

## Chạy

```bash
python -m app.launcher      # tự mở trình duyệt; hoặc ./run.sh / run.bat
```

Mở <http://localhost:8000>, kéo thả file PDF, chọn mô hình và bấm **Bắt đầu dịch**.

Dùng dòng lệnh:

```bash
python -m app.cli paper.pdf -o output/ --engine opus --pages 1-10
python -m app.cli paper.pdf --glossary my_terms.txt --translate-references
```

Docker:

```bash
docker compose up --build   # http://localhost:8000, mô hình được lưu trong ./data
```

### Dùng LLM cục bộ qua Ollama (tuỳ chọn)

```bash
ollama pull qwen2.5:7b
PDFTR_LLM_MODEL=qwen2.5:7b ./run.sh
```

Rồi chọn engine **LLM cục bộ** trên giao diện. Mọi server có API tương thích OpenAI chạy trên máy
(llama.cpp `llama-server`, LM Studio, vLLM) đều dùng được qua `PDFTR_LLM_BASE_URL`.

## Cấu hình (biến môi trường)

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `PDFTR_DEFAULT_ENGINE` | `auto` | Engine mặc định (`auto` / `vinai` / `opus` / `nllb` / `llm`) |
| `PDFTR_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `PDFTR_BACKEND` | `auto` | `auto` (ưu tiên CTranslate2) / `ct2` / `hf` (transformers + torch) |
| `PDFTR_THREADS` | `0` | Số luồng CPU cho CTranslate2 (0 = tự chọn) |
| `PDFTR_BATCH_SIZE` | `8` | Số câu dịch mỗi lô |
| `PDFTR_NUM_BEAMS` | `4` | Beam search (cao hơn = tốt hơn nhưng chậm hơn) |
| `PDFTR_DATA_DIR` | `./data` | Thư mục lưu file tải lên, kết quả, cache bản dịch |
| `PDFTR_MAX_UPLOAD_MB` / `PDFTR_MAX_PAGES` | `100` / `500` | Giới hạn kích thước |
| `PDFTR_JOB_TTL_HOURS` | `24` | Tự xoá kết quả cũ sau số giờ này |
| `PDFTR_FONT_SERIF` / `PDFTR_FONT_SANS` | tự tìm | Đường dẫn font TTF có dấu tiếng Việt |
| `PDFTR_QE_THRESHOLD` | `0.6` | Dưới ngưỡng này đoạn bị coi là nghi ngờ (tô vàng, gửi LLM sửa) |
| `PDFTR_POSTEDIT_MAX_RATIO` | `0.3` | Tỉ lệ câu tối đa được LLM sửa |
| `PDFTR_QE_BACKTRANS` | `0` | `1` = chấm điểm thêm bằng dịch ngược Việt→Anh (`PDFTR_BACKTRANS_MODEL`, mặc định opus-mt-vi-en) |
| `PDFTR_LAYOUT` | `auto` | `auto` = dùng mô hình bố cục DocLayout-YOLO nếu có onnxruntime; `off` = chỉ dùng luật |
| `PDFTR_LAYOUT_MODEL` | DocStructBench ONNX | `repo_id/file.onnx` trên Hugging Face hoặc đường dẫn file .onnx |
| `PDFTR_OCR_LANG` / `PDFTR_OCR_DPI` | `eng` / `300` | Ngôn ngữ và độ phân giải OCR cho PDF scan |
| `PDFTR_LLM_BASE_URL` | `http://localhost:11434/v1` | Địa chỉ LLM tự host |
| `PDFTR_LLM_MODEL` | `qwen2.5:7b` | Tên mô hình LLM |
| `PDFTR_VINAI_MODEL` / `PDFTR_OPUS_MODEL` / `PDFTR_NLLB_MODEL` | | Đổi sang mô hình/đường dẫn khác |

## Bảng thuật ngữ

Bảng mặc định ở `app/data/glossary_en_vi.txt`, mỗi dòng `thuật ngữ tiếng Anh = bản dịch`. Muốn giữ nguyên tiếng
Anh thì để hai vế giống nhau (`Transformer = Transformer`). Thuật ngữ riêng nhập ở tab **Thuật ngữ** trên giao
diện (được ưu tiên hơn bảng mặc định) hoặc truyền file qua `--glossary` khi dùng dòng lệnh.

## Cách hoạt động

Chi tiết: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

1. **Phân tích** (PyMuPDF) → **DocIR**: từng vùng chữ kèm font, cỡ chữ, vị trí, loại vùng (đoạn văn / công thức /
   bảng / mã nguồn / tài liệu tham khảo / chữ xoay); tách lại các cột bị gộp. DocIR được lưu thành `document.json`.
2. **Thuật ngữ**: bảng mặc định + thuật ngữ tự trích của bài + thuật ngữ của bạn.
3. **Tách câu & chuẩn hóa**: nối dòng, sửa gạch nối, đưa chữ IN HOA/Viết Hoa về dạng câu.
4. **Dịch** theo lô (cache SQLite). Mô hình dịch máy nhận câu thuần (công thức, trích dẫn giữ nguyên trong câu);
   LLM nhận placeholder `{n}` và thuật ngữ trong prompt.
5. **Kiểm tra**: phát hiện câu lặp/hỏng → dịch lại từng vế → vẫn hỏng thì giữ tiếng Anh.
6. **Dựng**: xóa chữ gốc (giữ hình và nét vẽ), ghi bản dịch vào đúng khung cũ bằng font có dấu tiếng Việt.

## Không dịch được? Tự chẩn đoán

```bash
python -m app.doctor --try vinai
```

Lệnh này kiểm tra thư viện, GPU, mô hình đã tải và thử dịch một câu thật, in ra lỗi cụ thể. Các lỗi hay gặp:

| Hiện tượng | Nguyên nhân / cách sửa |
|-----------|------------------------|
| Mô hình hiện "chưa sẵn sàng", lỗi `Chưa cài torch, transformers...` | Chưa cài thư viện mô hình: `pip install torch --index-url https://download.pytorch.org/whl/cpu` rồi `pip install -r requirements-ml.txt`, sau đó **khởi động lại server** |
| `Không nạp được mô hình ...` | Lần đầu cần Internet để tải mô hình từ huggingface.co (bị chặn/mạng chậm). Thử `python scripts/download_models.py opus` để xem lỗi tải |
| Dừng lâu ở "Đang tải mô hình ... MB" | Đang tải lần đầu (OPUS ~300 MB, VinAI ~1,5 GB) — chờ tải xong, các lần sau không cần tải lại |
| Dịch rất chậm | Máy chỉ có CPU: đặt `PDFTR_NUM_BEAMS=2` (nhanh gấp ~2 lần, chất lượng giảm nhẹ), hoặc dịch theo khoảng trang |
| `Lỗi khi nạp torch: ... DLL` (Windows) | Cài [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) rồi chạy lại |
| File kết quả vẫn tiếng Anh / `Không tìm thấy đoạn văn nào cần dịch` | Chạy `python -m app.inspect file.pdf` để xem tool phân loại từng khối chữ, gửi kết quả để được hỗ trợ |
| `Trang ... là ảnh scan, cần OCR nhưng: ...` | Không tải được dữ liệu OCR: tải `eng.traineddata` (tessdata_fast) vào `data/tessdata/` |
| `./run.sh` không chạy trên Windows | Dùng `run.bat` |

## Giới hạn

- OCR bản scan hai cột chất lượng thấp có thể nhận sai vài đoạn; bản scan rõ nét một cột cho kết quả tốt nhất.
- Chữ trong hình ảnh/biểu đồ dạng ảnh không được dịch.
- Công thức nội dòng được giữ nguyên nhưng hiển thị lại bằng font Unicode nên có thể khác nhẹ so với bản gốc.
- Dịch máy vẫn có thể sai thuật ngữ chuyên sâu — nên dùng bản song ngữ để đối chiếu.

## Phát triển

```bash
pip install -r requirements-dev.txt
pytest
python tests/make_sample_pdf.py sample.pdf
PDFTR_ENABLE_MOCK=1 python -m app.cli sample.pdf -e mock   # thử bố cục không cần mô hình
```

**Bộ PDF kiểm thử** (`tests/test_corpus.py`): 5 tài liệu mô phỏng LaTeX/arXiv, Wiley, IEEE, một cột, Elsevier
kèm đáp án phân tích. PDF thật của bạn đặt vào `tests/corpus/local/` (không đưa lên git) sẽ được kiểm tra tự
động; có thể kèm `<tên>.expect.json` (định dạng như `expect` trong `tests/corpus/docs.py`).

## Giấy phép các thành phần

PyMuPDF dùng giấy phép AGPL-3.0; NLLB-200 dùng CC-BY-NC 4.0 (phi thương mại). Giấy phép của từng mô hình
ghi trên trang Hugging Face tương ứng — hãy kiểm tra nếu dùng cho mục đích thương mại.
