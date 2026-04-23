# ── Stage 1: base image ───────────────────────────────────────────────────
FROM python:3.10-slim

# System dependencies (needed by TensorFlow & Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user ──────────────────────────────────────────────────
RUN useradd -m -u 1000 appuser
WORKDIR /app

# ── Install Python deps ───────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────
COPY app.py .

# ── Copy pre-trained models ───────────────────────────────────────────────
# Place the following files inside a  models/  folder next to this Dockerfile:
#   vibe_classifier_best.h5
#   quote_generator_best.h5
#   tokenizer.pkl
#   label_map.json
#   vibe_meta.json
COPY models/ ./models/

# ── Ownership ─────────────────────────────────────────────────────────────
RUN chown -R appuser:appuser /app
USER appuser

# ── Streamlit config ──────────────────────────────────────────────────────
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
