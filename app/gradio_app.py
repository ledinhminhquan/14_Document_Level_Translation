"""Standalone Gradio launcher for the doctrans demo.

    python app/gradio_app.py        # then open http://localhost:7860
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from doctrans.config import AppConfig, load_config  # noqa: E402
from doctrans.api.ui import launch  # noqa: E402


def main():
    path = os.environ.get("DOCTRANS_INFER_CONFIG")
    cfg = load_config(path) if path else AppConfig()
    port = int(os.environ.get("PORT", "7860"))
    launch(cfg, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
