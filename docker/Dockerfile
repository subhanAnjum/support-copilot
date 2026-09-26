FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf-cache \
    TOKENIZERS_PARALLELISM=false \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# CPU-only torch, the default wheel pulls in CUDA
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# bake the base model into the image
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('distilbert-base-uncased', allow_patterns=['*.json', '*.txt', '*.safetensors'])"

COPY . .

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
