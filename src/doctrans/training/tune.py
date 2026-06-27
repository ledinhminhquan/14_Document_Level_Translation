"""Lightweight context-window search (k) for document-level translation.

Sweeps the inference context window ``k`` and reports document-level chrF + SPS on the
seed docs — a cheap, offline proxy for the context-vs-no-context ablation. (With the
dictionary baseline, context is a no-op, so a single trial is reported; the real gain
shows on a context-fine-tuned model on Colab.)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger
from ..data import samples
from ..training import metrics as M

logger = get_logger(__name__)


def tune(cfg: AppConfig, ks: Optional[List[int]] = None, save: bool = True, load_model: bool = True) -> Dict:
    ks = ks or [0, 1, 2]
    from ..agent.doctrans_agent import DocTransAgent
    from ..mt.translator import load_translator
    translator = load_translator(cfg.mt, prefer="transformer" if load_model else "dictionary")
    is_transformer = getattr(translator, "name", "") == "transformer"

    trials: List[Dict[str, Any]] = []
    for k in ks:
        cfg.mt.context_k_infer = k
        cfg.mt.use_context = bool(k > 0 and is_transformer)
        agent = DocTransAgent(cfg, load_model=load_model, translator=translator)
        chrfs, sps = [], []
        for d in samples.docs():
            job = agent.run(d["src"], fmt=d.get("fmt", ""), save=False).to_dict()
            chrfs.append(M.chrf([job["output"]], [d["tgt"]]))
            sps.append(job["structure"].get("sps") or 0.0)
        trials.append({"k": k, "use_context": cfg.mt.use_context,
                       "doc_chrf": round(sum(chrfs) / len(chrfs), 2),
                       "doc_sps": round(sum(sps) / len(sps), 4)})
        if not is_transformer:
            break   # dictionary baseline is context-invariant

    best = max(trials, key=lambda t: t["doc_chrf"]) if trials else {}
    result = {"trials": trials, "best": best}
    if save:
        out = run_dir() / "tune"
        out.mkdir(parents=True, exist_ok=True)
        (out / "tune.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("tune: best k=%s doc_chrf=%s", best.get("k"), best.get("doc_chrf"))
    return result


__all__ = ["tune"]
