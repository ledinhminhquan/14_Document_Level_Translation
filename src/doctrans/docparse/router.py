"""Format detection + parse routing + the public structure-layer API.

``detect_format`` routes by extension then a content sniff; ``parse_document`` dispatches
to the per-format parser and **degrades to plain text** on any parse error (never raises).
``signature_of`` returns a format-specific structure signature used by the D5 validation
gate to confirm the translated document kept the same structure as the source.
"""

from __future__ import annotations

import json
from typing import Optional

from ..logging_utils import get_logger
from .segment import ParsedDoc
from . import markdown_doc, html_doc, json_doc, plain_doc

logger = get_logger(__name__)

_EXT = {".md": "markdown", ".markdown": "markdown", ".html": "html", ".htm": "html",
        ".json": "json", ".txt": "plain", ".text": "plain"}


def detect_format(text: str, filename: Optional[str] = None) -> str:
    if filename:
        for ext, fmt in _EXT.items():
            if filename.lower().endswith(ext):
                return fmt
    s = (text or "").lstrip()
    if s[:1] in "{[":
        try:
            json.loads(text)
            return "json"
        except Exception:
            pass
    low = s.lower()
    if "<html" in low or "</" in low or ("<" in low and ">" in low and ("<p" in low or "<div" in low or "<body" in low)):
        return "html"
    if any(line.lstrip().startswith(("#", "- ", "* ", "> ", "```", "|")) for line in s.splitlines()[:50]):
        return "markdown"
    return "plain"


def parse_document(text: str, fmt: Optional[str] = None, cfg=None, filename: Optional[str] = None) -> ParsedDoc:
    fmt = fmt or detect_format(text, filename)
    parsers = {"markdown": markdown_doc.parse_markdown, "html": html_doc.parse_html,
               "json": json_doc.parse_json, "plain": plain_doc.parse_plain}
    fn = parsers.get(fmt, plain_doc.parse_plain)
    try:
        doc = fn(text, cfg)
        # round-trip identity check (D1): re-joining must reproduce the source. JSON is
        # checked semantically (json.dumps re-formats whitespace), others byte-for-byte.
        if not _roundtrip_ok(doc.reassemble(), text, doc.fmt):
            logger.info("parse round-trip differs for fmt=%s; falling back to plain", fmt)
            return plain_doc.parse_plain(text, cfg)
        return doc
    except Exception as exc:
        logger.warning("parse failed for fmt=%s (%s); falling back to plain", fmt, exc)
        return plain_doc.parse_plain(text, cfg)


def _roundtrip_ok(rebuilt: str, original: str, fmt: str) -> bool:
    if fmt == "json":
        try:
            return json.loads(rebuilt) == json.loads(original)
        except Exception:
            return False
    return rebuilt == original


def signature_of(text: str, fmt: str) -> dict:
    if fmt == "markdown":
        return markdown_doc.structure_signature(text)
    if fmt == "html":
        return html_doc.structure_signature(text)
    if fmt == "json":
        try:
            return {"strings": _count_json_strings(json.loads(text))}
        except Exception:
            return {}
    return {"lines": len((text or "").split("\n")),
            "nonblank": sum(1 for l in (text or "").split("\n") if l.strip())}


def _count_json_strings(node) -> int:
    if isinstance(node, dict):
        return sum(_count_json_strings(v) for v in node.values())
    if isinstance(node, list):
        return sum(_count_json_strings(v) for v in node)
    return 1 if isinstance(node, str) else 0


__all__ = ["detect_format", "parse_document", "signature_of"]
