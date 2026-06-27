"""Reversible sentinel masking of non-translatable spans.

Replaces high-risk spans an MT model must NOT alter — inline code, fenced code, URLs,
emails, Markdown link targets, HTML/XML tags, and placeholders (``{name}``, ``{{var}}``,
``%s``, ``${VAR}``, ``:named``) — with short ASCII sentinels ``[[PHn]]`` (Windows-cp1252
safe, no internal spaces, survives tokenization), recording ``sentinel -> original`` so
they can be restored verbatim after translation.

Returns ``(masked_text, mask_map)``. :func:`validate_restoration` checks every sentinel
came back exactly once — the basis of the agent's D4 placeholder-retention gate.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# ordered, longest/most-specific first
_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("code_fence", re.compile(r"```.*?```", re.DOTALL)),
    ("code_inline", re.compile(r"`[^`]+`")),
    ("md_link", re.compile(r"\]\([^)]+\)")),            # the (url) part of [text](url) — keep [text]
    ("html_tag", re.compile(r"</?[A-Za-z][^>]*>")),
    ("url", re.compile(r"\b(?:https?://|www\.)[^\s)<>\]]+")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("ph_double", re.compile(r"\{\{[^}]+\}\}")),
    ("ph_single", re.compile(r"\{[^}\s]+\}")),
    ("ph_dollar", re.compile(r"\$\{?[A-Za-z_]\w*\}?")),
    ("ph_printf", re.compile(r"%(?:\d+\$)?[sdfx]")),
    ("ph_named", re.compile(r"(?<!\w):[A-Za-z_]\w+")),
]

# which categories each toggle controls
_TOGGLE = {
    "mask_code": {"code_fence", "code_inline"},
    "mask_urls": {"url", "email", "md_link"},
    "mask_html_tags": {"html_tag"},
    "mask_placeholders": {"ph_double", "ph_single", "ph_dollar", "ph_printf", "ph_named"},
}
_NUMBER = re.compile(r"\bv?\d+(?:[.\-]\d+)+\b")   # version strings / codes (mask_numbers)


def _enabled_categories(cfg) -> set:
    cats = set()
    for toggle, group in _TOGGLE.items():
        if getattr(cfg, toggle, True):
            cats |= group
    if getattr(cfg, "mask_numbers", False):
        cats.add("number")
    return cats


def mask_text(text: str, cfg=None, sentinel_format: str = "[[PH{}]]",
              start: int = 0) -> Tuple[str, Dict[str, str], int]:
    """Mask non-translatable spans. Returns (masked_text, mask_map, next_index).

    ``start`` lets a caller keep sentinel indices unique across many segments.
    """
    if cfg is not None:
        sentinel_format = getattr(cfg, "sentinel_format", sentinel_format)
        cats = _enabled_categories(cfg)
    else:
        cats = {name for name, _ in _PATTERNS}
    mask_map: Dict[str, str] = {}
    idx = start

    patterns = [(n, p) for n, p in _PATTERNS if n in cats]
    if "number" in cats:
        patterns.append(("number", _NUMBER))

    for _name, pat in patterns:
        def _sub(m: "re.Match[str]") -> str:
            nonlocal idx
            sent = sentinel_format.format(idx)
            mask_map[sent] = m.group(0)
            idx += 1
            return sent
        text = pat.sub(_sub, text)
    return text, mask_map, idx


def restore(text: str, mask_map: Dict[str, str]) -> str:
    for sent in sorted(mask_map, key=len, reverse=True):
        text = text.replace(sent, mask_map[sent])
    return text


def validate_restoration(translated_masked: str, mask_map: Dict[str, str]) -> Dict[str, object]:
    """Check every inserted sentinel survived the translation exactly once.

    Returns ``{ok, retained, total, retention, missing[]}`` — the D4 hard gate.
    """
    total = len(mask_map)
    if total == 0:
        return {"ok": True, "retained": 0, "total": 0, "retention": 1.0, "missing": []}
    retained = 0
    missing: List[str] = []
    for sent in mask_map:
        c = translated_masked.count(sent)
        if c == 1:
            retained += 1
        else:
            missing.append(sent)
    return {"ok": not missing, "retained": retained, "total": total,
            "retention": round(retained / total, 4), "missing": missing}


__all__ = ["mask_text", "restore", "validate_restoration"]
