FROM python:3.10-slim
RUN pip install --no-cache-dir fastapi uvicorn requests torch --extra-index-url https://download.pytorch.org/whl/cpu transformers
COPY hub.py page.html /app/
WORKDIR /app
CMD ["uvicorn", "hub:app", "--host", "0.0.0.0", "--port", "7860"]
