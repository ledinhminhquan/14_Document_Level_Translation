"""Decision-point logic for the document-translation agent (pure, testable).

Five explicit decision points on the cascade's intermediate artifacts:
* **D1** format detection + parse (round-trip identity check; degrade to plain on mismatch).
* **D2** segment + mask gate (skip non-translatable segments; require balanced masks).
* **D3** translate sanity gate (reject empty / wild length-ratio output).
* **D4** verify gate (placeholder-retention [hard] + round-trip back-translation chrF [soft]).
* **D5** reassemble + structure-validation gate (structure signature match + clean re-parse).
"""

from __future__ import annotations

import re
from typing import Any, Dict

from ..config import AgentConfig

_LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)


def letter_ratio(text: str) -> float:
    s = re.sub(r"\s", "", text or "")
    if not s:
        return 0.0
    return sum(1 for c in s if _LETTERS.match(c)) / len(s)


def segment_translatable(text: str, cfg: AgentConfig) -> bool:
    """D2 — is this segment worth sending to MT (enough letters)?"""
    return letter_ratio(text) >= cfg.min_letter_ratio


def translate_sane(src: str, hyp: str, cfg: AgentConfig) -> Dict[str, Any]:
    """D3 — basic sanity of one translation (non-empty, plausible length ratio)."""
    if not (hyp or "").strip():
        return {"ok": False, "branch": "empty"}
    ls, lh = max(1, len(src)), len(hyp)
    ratio = lh / ls
    if ratio < cfg.length_ratio_low or ratio > cfg.length_ratio_high:
        return {"ok": False, "branch": "length_ratio", "ratio": round(ratio, 3)}
    return {"ok": True, "branch": "ok", "ratio": round(ratio, 3)}


def verify_gate(placeholder_retention: float, verify_chrf, cfg: AgentConfig) -> Dict[str, Any]:
    """D4 — placeholder retention (hard) + round-trip chrF (soft)."""
    if placeholder_retention < 1.0:
        return {"ok": False, "branch": "placeholder_loss", "retention": placeholder_retention}
    if cfg.verify_enabled and verify_chrf is not None and verify_chrf < cfg.verify_min_chrf:
        return {"ok": False, "branch": "low_similarity", "chrf": verify_chrf}
    return {"ok": True, "branch": "ok"}


def structure_gate(sps_report: Dict[str, Any]) -> Dict[str, Any]:
    """D5 — structure preserved + output re-parses cleanly."""
    sm = sps_report.get("struct_match", 1.0)
    mv = sps_report.get("markup_validity", 1.0)
    ok = sm >= 0.999 and mv >= 1.0
    branch = "ok" if ok else ("structure_mismatch" if sm < 0.999 else "invalid_markup")
    return {"ok": ok, "branch": branch, "struct_match": sm, "markup_validity": mv}


__all__ = ["letter_ratio", "segment_translatable", "translate_sane", "verify_gate", "structure_gate"]
