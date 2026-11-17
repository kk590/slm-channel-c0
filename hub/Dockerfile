FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY hub.py page.html router.html ./
ENV SHARD_ID=0 \
    NUM_SHARDS=1 \
    SSE_BATCH_SIZE=8 \
    SSE_BATCH_MS=200 \
    DEDUPE_WINDOW=3600 \
    MAX_NEW=192 \
    QUOTA_PER_DAY=20 \
    SECS_PER_GEN=12
# Render injects PORT; local/HF default 7860
CMD ["sh", "-c", "uvicorn hub:app --host 0.0.0.0 --port ${PORT:-7860}"]
