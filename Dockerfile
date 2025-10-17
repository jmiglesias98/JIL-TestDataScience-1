# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_DIR=/app/models
EXPOSE 80

# Ejecuta uvicorn en puerto 80
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]