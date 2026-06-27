#!/usr/bin/env bash
# Start the FastAPI service + Gradio demo (/ui).
set -e
export PYTHONPATH=src
export DOCTRANS_ARTIFACTS_DIR="${DOCTRANS_ARTIFACTS_DIR:-./artifacts}"
python -m doctrans.cli --config configs/infer.yaml serve --ui --host 0.0.0.0 --port 8000
