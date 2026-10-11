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
  - Dịch chuẩn tiêu đề mục (Abstract → Tóm tắt, Related Work → Các công trình liên quan…) và nhãn
    (Fig. 3 → Hình 3, Table 2 → Bảng 2, Eq. (1) → Phương trình (1), et al. → và cộng sự).
  - Giữ nguyên mục **Tài liệu tham khảo** (tùy chọn), bỏ qua bảng số liệu, mã nguồn, watermark xoay của arXiv.
  - Tách câu thông minh (không cắt sai ở `et al.`, `e.g.`, `Fig.`…), nối lại từ bị ngắt dòng bằng gạch nối.
- **3 dạng kết quả**: PDF tiếng Việt, PDF song ngữ (gốc | dịch cạnh nhau), văn bản Markdown.
- Có ô **dịch nhanh đoạn văn**, bộ nhớ đệm bản dịch (dịch lại tài liệu cũ rất nhanh), hàng đợi công việc,
  thanh tiến độ, xem trước PDF ngay trên trình duyệt.

## Mô hình dịch (đều miễn phí, chạy cục bộ)

| Engine  | Mô hình | Ghi chú |
|---------|---------|---------|
| `vinai` | [vinai/vinai-translate-en2vi-v2](https://huggingface.co/vinai/vinai-translate-en2vi-v2) | Chất lượng Anh→Việt tốt nhất trong nhóm mô hình nhỏ; ~1,5 GB; nên có GPU nhưng CPU vẫn chạy được |
| `opus`  | [Helsinki-NLP/opus-mt-en-vi](https://huggingface.co/Helsinki-NLP/opus-mt-en-vi) | Nhẹ (~300 MB), nhanh trên CPU, chất lượng khá |
| `nllb`  | [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | Giấy phép CC-BY-NC (chỉ dùng phi thương mại) |
| `llm`   | LLM mã nguồn mở tự host (Qwen2.5, Llama 3, Gemma…) qua Ollama / llama.cpp / LM Studio | Dịch theo ngữ cảnh cả đoạn, hiểu thuật ngữ tốt nhất; cần máy đủ mạnh |

Mặc định (`auto`): máy có GPU NVIDIA dùng `vinai`, chỉ có CPU dùng `opus` cho nhanh.
Mô hình Hugging Face chỉ cần tải **một lần** (tự động ở lần dịch đầu tiên hoặc dùng `scripts/download_models.py`),
sau đó chạy offline hoàn toàn.

## Cài đặt

Yêu cầu Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# torch: chọn bản phù hợp máy (bản CPU nhẹ hơn nhiều)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-ml.txt

# (tuỳ chọn) tải trước mô hình để dùng offline
python scripts/download_models.py vinai opus
```

## Chạy

```bash
./run.sh                    # Windows: run.bat  (hoặc: python -m uvicorn app.main:app --port 8000)
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
| `PDFTR_BATCH_SIZE` | `8` | Số câu dịch mỗi lô |
| `PDFTR_NUM_BEAMS` | `4` | Beam search (cao hơn = tốt hơn nhưng chậm hơn) |
| `PDFTR_DATA_DIR` | `./data` | Thư mục lưu file tải lên, kết quả, cache bản dịch |
| `PDFTR_MAX_UPLOAD_MB` / `PDFTR_MAX_PAGES` | `100` / `500` | Giới hạn kích thước |
| `PDFTR_JOB_TTL_HOURS` | `24` | Tự xoá kết quả cũ sau số giờ này |
| `PDFTR_FONT_SERIF` / `PDFTR_FONT_SANS` | tự tìm | Đường dẫn font TTF có dấu tiếng Việt |
| `PDFTR_LLM_BASE_URL` | `http://localhost:11434/v1` | Địa chỉ LLM tự host |
| `PDFTR_LLM_MODEL` | `qwen2.5:7b` | Tên mô hình LLM |
| `PDFTR_VINAI_MODEL` / `PDFTR_OPUS_MODEL` / `PDFTR_NLLB_MODEL` | | Đổi sang mô hình/đường dẫn khác |

## Bảng thuật ngữ

Bảng mặc định ở `app/data/glossary_en_vi.txt`, mỗi dòng `thuật ngữ tiếng Anh = bản dịch`. Muốn giữ nguyên tiếng
Anh thì để hai vế giống nhau (`Transformer = Transformer`). Thuật ngữ riêng nhập ở tab **Thuật ngữ** trên giao
diện (được ưu tiên hơn bảng mặc định) hoặc truyền file qua `--glossary` khi dùng dòng lệnh.

## Cách hoạt động

1. **Phân tích PDF** (PyMuPDF): lấy từng khối văn bản cùng font, cỡ chữ, màu, vị trí; gộp các khối bị tách giữa câu.
2. **Phân loại khối**: đoạn văn / công thức / bảng / mã nguồn / tài liệu tham khảo / chữ xoay.
3. **Bảo vệ**: công thức, ký hiệu, trích dẫn, URL… và thuật ngữ được thay bằng placeholder `{n}`.
4. **Dịch** từng câu theo lô; nếu mô hình làm mất placeholder thì tự dịch lại từng phần giữa các placeholder.
5. **Khôi phục** placeholder, xóa chữ gốc (giữ hình và nét vẽ) và ghi bản dịch vào đúng khung cũ bằng font
   hỗ trợ tiếng Việt.

## Không dịch được? Tự chẩn đoán

```bash
python -m app.doctor --try opus
```

Lệnh này kiểm tra thư viện, GPU, mô hình đã tải và thử dịch một câu thật, in ra lỗi cụ thể. Các lỗi hay gặp:

| Hiện tượng | Nguyên nhân / cách sửa |
|-----------|------------------------|
| Mô hình hiện "chưa sẵn sàng", lỗi `Chưa cài torch, transformers...` | Chưa cài thư viện mô hình: `pip install torch --index-url https://download.pytorch.org/whl/cpu` rồi `pip install -r requirements-ml.txt`, sau đó **khởi động lại server** |
| `Không nạp được mô hình ...` | Lần đầu cần Internet để tải mô hình từ huggingface.co (bị chặn/mạng chậm). Thử `python scripts/download_models.py opus` để xem lỗi tải |
| Dừng lâu ở "Đang tải mô hình ... MB" | Đang tải lần đầu (OPUS ~300 MB, VinAI ~1,5 GB) — chờ tải xong, các lần sau không cần tải lại |
| Dịch rất chậm | Máy chỉ có CPU: chọn engine **OPUS-MT**, hoặc giảm `PDFTR_NUM_BEAMS=1` |
| `Lỗi khi nạp torch: ... DLL` (Windows) | Cài [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) rồi chạy lại |
| `PDF gần như không có lớp chữ` | PDF dạng ảnh scan — cần OCR trước (`ocrmypdf`) |
| `./run.sh` không chạy trên Windows | Dùng `run.bat` |

## Giới hạn

- PDF dạng ảnh scan cần OCR trước, ví dụ `ocrmypdf --language eng input.pdf output.pdf`.
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

## Giấy phép các thành phần

PyMuPDF dùng giấy phép AGPL-3.0; NLLB-200 dùng CC-BY-NC 4.0 (phi thương mại). Giấy phép của từng mô hình
ghi trên trang Hugging Face tương ứng — hãy kiểm tra nếu dùng cho mục đích thương mại.
