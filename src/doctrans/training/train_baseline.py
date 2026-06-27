"""Persist the dictionary MT baseline (the dependency-light floor / offline fallback)."""

from __future__ import annotations

import json
from typing import Dict, Optional

from ..config import AppConfig
from ..logging_utils import get_logger
from ..data.dataset import seed_split
from ..data import samples
from ..mt.translator import DictionaryTranslator
from . import metrics as M

logger = get_logger(__name__)


def train_baseline(cfg: AppConfig, limit: Optional[int] = None, save: bool = True) -> Dict:
    table = samples.dictionary()
    mt = DictionaryTranslator(table)
    _, eval_pairs = seed_split(cfg.data.seed)
    hyps = [mt.translate(p.src) for p in eval_pairs]
    refs = [p.tgt for p in eval_pairs]
    m = M.translation_metrics(hyps, refs)
    if save:
        path = cfg.mt.baseline_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"dictionary": table}, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("dictionary baseline saved -> %s (chrF=%s BLEU=%s)", path, m["chrf"], m["bleu"])
    return {"model": "dictionary", "n_eval": len(eval_pairs), "metrics": m, "path": str(cfg.mt.baseline_path)}


__all__ = ["train_baseline"]
