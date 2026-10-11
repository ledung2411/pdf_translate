FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PDFTR_DATA_DIR=/data HF_HOME=/data/hf

RUN apt-get update && apt-get install -y --no-install-recommends fonts-liberation fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
# torch bản CPU (nhẹ hơn nhiều so với bản CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt

COPY app ./app
COPY scripts ./scripts

VOLUME ["/data"]
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
