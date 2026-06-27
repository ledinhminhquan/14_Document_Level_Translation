"""Gradio demo for structured document translation (lazy-imported).

Paste a Markdown/HTML/JSON document -> the translated document (same format) + a
structure-preservation report + the decision trace. Gradio is imported lazily.
"""

from __future__ import annotations

from typing import Optional

from ..config import AppConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def build_demo(cfg: Optional[AppConfig] = None):
    import gradio as gr
    cfg = cfg or AppConfig()
    from ..agent.doctrans_agent import DocTransAgent
    agent = DocTransAgent(cfg, load_model=True)

    def do_translate(text: str, fmt: str):
        if not (text or "").strip():
            return "", {}, ""
        job = agent.run(text, fmt=("" if fmt == "auto" else fmt), save=False)
        sd = job.to_dict()
        report = {"fmt": sd["fmt"], "status": sd["status"], "needs_review": sd["needs_review"],
                  "placeholder_retention": sd["placeholder_retention"],
                  "structure": sd["structure"], "decisions": [(d["id"], d["branch"]) for d in sd["decisions"]]}
        return sd["output"], report, sd["rationale"]

    with gr.Blocks(title=cfg.serving.api_title) as demo:
        gr.Markdown(f"# {cfg.serving.api_title}\n"
                    f"Translate a structured document ({cfg.mt.src_lang} -> {cfg.mt.tgt_lang}) "
                    "preserving its structure. Structure-breaking output is flagged for review.")
        with gr.Row():
            inp = gr.Textbox(label="Source document (markdown / html / json / plain)", lines=16)
            out = gr.Textbox(label="Translated document", lines=16)
        fmt = gr.Dropdown(["auto", "markdown", "html", "json", "plain"], value="auto", label="Format")
        btn = gr.Button("Translate document", variant="primary")
        report = gr.JSON(label="Structure-preservation report + decisions")
        rat = gr.Textbox(label="Rationale")
        btn.click(do_translate, [inp, fmt], [out, report, rat])
    return demo


def launch(cfg: Optional[AppConfig] = None, host: str = "0.0.0.0", port: int = 7860):
    build_demo(cfg).launch(server_name=host, server_port=port)


__all__ = ["build_demo", "launch"]
