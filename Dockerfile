FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY backend/requirements.txt backend/

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces default port is 7860
EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
