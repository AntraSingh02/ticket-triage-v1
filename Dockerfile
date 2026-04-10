# Lean image - no heavy openenv-core runtime needed for serving
FROM python:3.11-slim

# ── Runtime behaviour ─────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Inference env variables (override at `docker run -e VAR=value`) ───────────
# API_BASE_URL  : LLM API base URL (OpenAI-compatible endpoint)
# API_KEY       : LLM API key  (HF_TOKEN is also checked as a fallback)
# MODEL_NAME    : Model name to use for ticket classification
# ENV_URL       : URL of the running OpenEnv server
# DOCKER_IMAGE  : Image name tag used when building this image locally
ENV API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/" \
    API_KEY="dummy" \
    MODEL_NAME="gemini-2.5-flash" \
    ENV_URL="http://127.0.0.1:8000" \
    DOCKER_IMAGE="ticket-triage-v1:latest"

WORKDIR /app

# Install only what matters for the FastAPI server
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn==0.30.6 \
    pydantic==2.7.1 \
    requests==2.31.0 \
    openai==1.30.1

# Copy project files
COPY . /app

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
