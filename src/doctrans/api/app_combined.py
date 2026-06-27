"""Combined ASGI app: the FastAPI service + the Gradio demo mounted at ``/ui``."""

from __future__ import annotations

from .main import app
from ..logging_utils import get_logger

logger = get_logger(__name__)

try:
    import gradio as gr  # noqa: F401
    from .dependencies import get_config
    from .ui import build_demo

    demo = build_demo(get_config())
    app = gr.mount_gradio_app(app, demo, path="/ui")
    logger.info("Gradio demo mounted at /ui")
except Exception as exc:  # API still works without the UI
    logger.info("Gradio UI not mounted (%s)", exc)

__all__ = ["app"]
