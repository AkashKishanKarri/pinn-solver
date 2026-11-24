# Production image: builds the React UI, installs CPU-only PyTorch, serves API + UI with uvicorn.
# Works on Hugging Face Spaces (Docker SDK), Render, Railway, Fly.io or any Docker host.

# ---- stage 1: frontend ----
FROM node:20-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- stage 2: runtime ----
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg PINN_RUNS_DIR=/tmp/pinn-runs PORT=7860
WORKDIR /app
COPY requirements.txt ./
# CPU wheel of the pinned torch version (the default wheel bundles CUDA, ~2.5 GB larger).
RUN pip install --index-url https://download.pytorch.org/whl/cpu "$(grep '^torch==' requirements.txt)" \
 && pip install -r requirements.txt
COPY pinnsolver/ ./pinnsolver/
COPY --from=ui /ui/dist ./frontend/dist
# Hugging Face Spaces run the container as uid 1000.
RUN useradd -m -u 1000 app && mkdir -p /tmp/pinn-runs && chown -R app /tmp/pinn-runs
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health')"
CMD ["sh", "-c", "uvicorn pinnsolver.api:app --host 0.0.0.0 --port ${PORT}"]
