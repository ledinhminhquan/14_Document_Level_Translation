#!/usr/bin/env bash
# Offline end-to-end demo: baseline -> evaluate -> agent on the seed structured docs.
set -e
export PYTHONPATH=src
export DOCTRANS_ARTIFACTS_DIR="${DOCTRANS_ARTIFACTS_DIR:-./artifacts}"
export HF_HUB_OFFLINE=1
python -m doctrans.cli train-baseline
python -m doctrans.cli evaluate --fast
echo "--- document agent on the seed docs ---"
python -m doctrans.cli demo-agent --fast
