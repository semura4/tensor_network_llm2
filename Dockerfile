FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU (smaller image, use nvidia/cuda base for GPU)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

COPY . .

ENV MODEL_CONFIG=config/gpt_tn_85m.yaml
ENV MODEL_CHECKPOINT=""

EXPOSE 8000

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
