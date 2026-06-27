# Multi-stage Docker image for doctrans (CPU default).
# Installs core + serving + report extras; the heavy ML libs (torch/transformers) are
# optional — the service answers in degraded mode (line/regex parsers + dictionary MT)
# without them.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DOCTRANS_ARTIFACTS_DIR=/data/artifacts \
    HF_HOME=/data/hf_cache \
    MPLBACKEND=Agg

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[api,report,parsers]"

COPY configs ./configs
COPY sample_data ./sample_data

RUN mkdir -p /data/artifacts /data/hf_cache
EXPOSE 8000 7860

# default: REST API (+ /ui demo). Override CMD for the standalone Gradio demo.
CMD ["uvicorn", "doctrans.api.app_combined:app", "--host", "0.0.0.0", "--port", "8000"]
