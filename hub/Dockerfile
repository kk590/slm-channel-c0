FROM python:3.10-slim
RUN pip install --no-cache-dir fastapi uvicorn requests
COPY hub.py page.html router.html /app/
WORKDIR /app
ENV SHARD_ID=0 \
    NUM_SHARDS=1 \
    SSE_BATCH_SIZE=8 \
    SSE_BATCH_MS=200 \
    DEDUPE_WINDOW=3600 \
    MAX_NEW=192 \
    QUOTA_PER_DAY=3 \
    SECS_PER_GEN=12
EXPOSE 7860
CMD ["uvicorn", "hub:app", "--host", "0.0.0.0", "--port", "7860"]
