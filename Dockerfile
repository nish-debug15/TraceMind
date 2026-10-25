FROM python:3.11-slim

# Install system dependencies (build-essential needed for compiling hdbscan wheels on Linux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requires running as user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    LLM_MODEL=openai/gpt-oss-120b \
    LLM_MAX_RPM=20

WORKDIR /app

# Copy dependency specifications first to take advantage of layer caching
COPY --chown=user:user requirements.txt .
COPY --chown=user:user backend/requirements.txt backend/

RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application source code and prebuilt data/indices
COPY --chown=user:user . .

# Hugging Face Spaces binds to port 7860 by default
EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
