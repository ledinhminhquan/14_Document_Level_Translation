"""Structure-Preservation Score (SPS) — does the translated document keep its structure?

A translation that wins chrF but breaks a table or code block is a failure, so SPS is
reported jointly with chrF. Components:

* **StructMatch** — agreement of structural-element counts (headings, lists, tables, code,
  tags) between source and output: ``1 - sum|count_src - count_out| / sum(count_src)``.
* **MarkupValidity** — does the output re-parse cleanly in its format (round-trips)?
* **PHR (placeholder-retention rate)** — fraction of inserted sentinels restored exactly
  once (computed during translation; aggregated here).

``sps`` is the (default-equal) weighted mean. Pure stdlib.
"""

from __future__ import annotations

from typing import Any, Dict, List

from . import router
from .segment import ParsedDoc


def struct_match(sig_src: Dict[str, int], sig_out: Dict[str, int]) -> float:
    keys = set(sig_src) | set(sig_out)
    total = sum(sig_src.get(k, 0) for k in keys)
    if total == 0:
        return 1.0
    diff = sum(abs(sig_src.get(k, 0) - sig_out.get(k, 0)) for k in keys)
    return round(max(0.0, 1.0 - diff / total), 4)


def markup_validity(out_text: str, fmt: str) -> float:
    """1.0 if the output re-parses cleanly (round-trips), else 0.0."""
    try:
        doc = router.parse_document(out_text, fmt=fmt)
        return 1.0 if doc.reassemble() == out_text else 0.5
    except Exception:
        return 0.0


def structure_preservation(src_text: str, out_text: str, fmt: str) -> Dict[str, Any]:
    sig_src = router.signature_of(src_text, fmt)
    sig_out = router.signature_of(out_text, fmt)
    sm = struct_match(sig_src, sig_out)
    mv = markup_validity(out_text, fmt)
    return {"struct_match": sm, "markup_validity": mv,
            "sig_src": sig_src, "sig_out": sig_out,
            "sps": round(0.5 * sm + 0.5 * mv, 4)}


def placeholder_retention_rate(retentions: List[float]) -> float:
    """Aggregate per-segment placeholder-retention values into one rate."""
    rs = [r for r in retentions if r is not None]
    return round(sum(rs) / len(rs), 4) if rs else 1.0


def sps_full(src_text: str, out_text: str, fmt: str, phr: float = 1.0) -> Dict[str, Any]:
    """The full SPS: structure-match + markup-validity + placeholder-retention."""
    base = structure_preservation(src_text, out_text, fmt)
    sps = round((base["struct_match"] + base["markup_validity"] + phr) / 3.0, 4)
    return {**base, "phr": round(phr, 4), "sps": sps}


__all__ = ["struct_match", "markup_validity", "structure_preservation",
           "placeholder_retention_rate", "sps_full"]
