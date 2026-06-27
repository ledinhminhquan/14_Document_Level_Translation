"""Document translation error analysis (offline): chrF buckets + structure / placeholder issues.

Runs the agent on the seed structured docs, scores each doc's chrF vs gold + its SPS, and
buckets good/medium/poor; records docs that needed review (placeholder loss / structure
mismatch / low chrF). Short keys ``good``/``medium``/``poor`` feed the charts.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp
from ..training.metrics import chrf

logger = get_logger(__name__)


def error_analysis(cfg: AppConfig = None, limit: Optional[int] = None, save: bool = True) -> Dict:
    cfg = cfg or AppConfig()
    try:
        from ..agent.doctrans_agent import DocTransAgent
        from ..data import samples
    except Exception as exc:
        return _stub(str(exc), save)
    try:
        agent = DocTransAgent(cfg, load_model=False)
        docs = samples.docs()
    except Exception as exc:
        return _stub(str(exc), save)

    good = medium = poor = needs_review = struct_issues = ph_issues = 0
    chrfs, sps_list = [], []
    examples: List[Dict] = []
    for d in docs:
        try:
            job = agent.run(d["src"], fmt=d.get("fmt", ""), save=False).to_dict()
        except Exception:
            continue
        score = chrf([job["output"]], [d["tgt"]])
        chrfs.append(score)
        sps_list.append(job["structure"].get("sps") or 0.0)
        if job["needs_review"]:
            needs_review += 1
        if (job["structure"].get("struct_match") or 1.0) < 0.999:
            struct_issues += 1
        if (job["placeholder_retention"] if job["placeholder_retention"] is not None else 1.0) < 1.0:
            ph_issues += 1
        if score >= 60:
            good += 1
        elif score >= 35:
            medium += 1
        else:
            poor += 1
            if len(examples) < 8:
                examples.append({"id": d["id"], "chrf": round(score, 2),
                                 "sps": job["structure"].get("sps")})
    n = max(1, len(chrfs))
    result = {"n_docs": len(chrfs), "mean_chrf": round(sum(chrfs) / n, 2),
              "mean_sps": round(sum(sps_list) / n, 4),
              "good": good, "medium": medium, "poor": poor,
              "needs_review": needs_review, "struct_issues": struct_issues,
              "placeholder_issues": ph_issues, "worst_examples": examples}
    if save:
        _save(result)
    logger.info("error analysis: good=%d medium=%d poor=%d needs_review=%d struct_issues=%d",
                good, medium, poor, needs_review, struct_issues)
    return result


def _stub(error: str, save: bool, **extra) -> Dict:
    result = {"n_docs": 0, "mean_chrf": 0.0, "mean_sps": 0.0, "good": 0, "medium": 0, "poor": 0,
              "needs_review": 0, "struct_issues": 0, "placeholder_issues": 0,
              "worst_examples": [], "error": error}
    result.update(extra)
    if save:
        _save(result)
    return result


def _save(result: Dict) -> None:
    try:
        d = run_dir() / "error_analysis"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"errors-{utc_stamp()}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.info("error_analysis: could not save (%s)", exc)


__all__ = ["error_analysis"]
