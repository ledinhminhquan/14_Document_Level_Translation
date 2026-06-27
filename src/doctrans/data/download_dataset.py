"""Prefetch + sanity-check the datasets (no large files committed).

Streaming probes confirm the fine-tune + context corpora are reachable and report their
schema WITHOUT downloading them in full (OPUS-100 is 1M pairs). Degrades gracefully: the
built-in seed always works.
"""

from __future__ import annotations

from typing import Any, Dict

from ..config import AppConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def _probe(loader) -> Dict[str, Any]:
    try:
        return {"ok": True, **loader()}
    except Exception as exc:  # pragma: no cover - network dependent
        return {"ok": False, "error": str(exc)}


def download_all(cfg: AppConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {"mt": {}, "context": {}, "seed": {}}
    dc = cfg.data

    def mt_probe():
        from datasets import load_dataset
        ds = load_dataset(dc.mt_dataset, dc.mt_config, split="train", streaming=True)
        first = next(iter(ds))
        return {"dataset": dc.mt_dataset, "config": dc.mt_config, "reachable": True,
                "columns": list(first.keys())}

    def ctx_probe():
        from datasets import load_dataset
        ds = load_dataset(dc.context_dataset, dc.context_config, split="train", streaming=True)
        first = next(iter(ds))
        return {"dataset": dc.context_dataset, "config": dc.context_config, "reachable": True,
                "columns": list(first.keys())}

    out["mt"] = _probe(mt_probe)
    out["context"] = _probe(ctx_probe)

    from . import samples
    out["seed"] = {"ok": True, "pairs": len(samples.pairs()), "docs": len(samples.docs()),
                   "dict_size": len(samples.dictionary())}
    logger.info("download_all: mt=%s context=%s seed=%d pairs / %d docs",
                out["mt"].get("ok"), out["context"].get("ok"), out["seed"]["pairs"], out["seed"]["docs"])
    return out


__all__ = ["download_all"]
