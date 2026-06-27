"""Matplotlib charts for the document-translation report/slides.

  * an **MT quality** bar chart — chrF + BLEU for identity vs dictionary vs model;
  * a **document fidelity** chart — document chrF (0-100), SPS x100, PHR x100;
  * a **chrF bucket** chart (good / medium / poor) from error analysis.

Returns saved PNG paths under ``run_dir()/report``; matplotlib lazy-imported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..logging_utils import get_logger

logger = get_logger(__name__)

_IDENTITY = "#cbd5e0"
_DICT = "#9aa7b4"
_MODEL = "#2b6cb0"
_GOOD = "#2f855a"
_MED = "#dd6b20"
_POOR = "#c53030"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _f(d: Dict[str, Any], key: str) -> Optional[float]:
    v = (d or {}).get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def quality_chart(eval_art: Dict[str, Any], out_path: Path) -> Optional[Path]:
    model = (eval_art or {}).get("model") or {}
    if not model:
        return None
    try:
        plt = _mpl()
        metrics = [("chrf", "chrF"), ("bleu", "BLEU")]
        series = [("identity", (eval_art or {}).get("identity") or {}, _IDENTITY),
                  ("dictionary", (eval_art or {}).get("dictionary") or {}, _DICT),
                  ("model", model, _MODEL)]
        x = list(range(len(metrics)))
        width = 0.8 / len(series)
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for si, (label, data, color) in enumerate(series):
            vals = [(_f(data, k) or 0.0) for k, _ in metrics]
            offs = [i + (si - (len(series) - 1) / 2) * width for i in x]
            bars = ax.bar(offs, vals, width=width, label=label, color=color)
            for rect, v in zip(bars, vals):
                if v > 0:
                    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.5, f"{v:.1f}",
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in metrics])
        ax.set_ylabel("score (0-100)"); ax.set_ylim(0, 105)
        ax.set_title("MT quality: model vs dictionary vs identity floor")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("quality_chart skipped (%s)", exc)
        return None


def fidelity_chart(eval_art: Dict[str, Any], out_path: Path) -> Optional[Path]:
    doc = (eval_art or {}).get("document") or {}
    if not doc:
        return None
    try:
        plt = _mpl()
        labels = ["doc chrF", "SPS x100", "PHR x100"]
        vals = [(_f(doc, "doc_chrf") or 0.0), (_f(doc, "doc_sps") or 0.0) * 100,
                (_f(doc, "doc_phr") or 0.0) * 100]
        fig, ax = plt.subplots(figsize=(5.8, 3.4))
        ax.bar(labels, vals, color=[_MODEL, _GOOD, "#553c9a"])
        for i, v in enumerate(vals):
            ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 105); ax.set_ylabel("score")
        ax.set_title("Document fidelity (chrF + structure-preservation + placeholders)")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("fidelity_chart skipped (%s)", exc)
        return None


def buckets_chart(err_art: Dict[str, Any], out_path: Path) -> Optional[Path]:
    if not err_art:
        return None
    vals = [err_art.get("good"), err_art.get("medium"), err_art.get("poor")]
    if not any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return None
    try:
        plt = _mpl()
        labels = ["good\n(chrF>=60)", "medium\n(35-60)", "poor\n(<35)"]
        nums = [float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0 for v in vals]
        fig, ax = plt.subplots(figsize=(5.6, 3.3))
        ax.bar(labels, nums, color=[_GOOD, _MED, _POOR])
        for i, v in enumerate(nums):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("# documents"); ax.set_title("Document quality buckets (by chrF)")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("buckets_chart skipped (%s)", exc)
        return None


def build_all(arts: Dict[str, Any], out_dir: Path) -> List[Tuple[str, Path]]:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return []
    charts: List[Tuple[str, Path]] = []
    jobs = [("quality", lambda p: quality_chart(arts.get("eval") or {}, p)),
            ("fidelity", lambda p: fidelity_chart(arts.get("eval") or {}, p)),
            ("buckets", lambda p: buckets_chart(arts.get("error_analysis") or {}, p))]
    for name, fn in jobs:
        try:
            p = fn(out_dir / f"{name}.png")
        except Exception as exc:
            logger.info("chart %s skipped (%s)", name, exc)
            p = None
        if p:
            charts.append((name, p))
    return charts


__all__ = ["quality_chart", "fidelity_chart", "buckets_chart", "build_all"]
