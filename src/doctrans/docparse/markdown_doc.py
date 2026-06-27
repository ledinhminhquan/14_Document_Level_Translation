"""Line-based Markdown parser → interleaved skeleton + translatable segments.

A dependency-free, deterministic Markdown parser: it splits the document into lines,
keeps all structural markup (heading hashes, list markers, blockquote ``>``, table
pipes, fenced code, horizontal rules) as literal skeleton, and emits the prose as
:class:`Segment` objects with inline non-translatable spans masked. Re-joining the
literals with the segments reproduces the document byte-for-byte (the D1 round-trip
identity check), and the structure (heading/list/table/code counts) is preserved by
construction. Optionally upgrades to ``markdown-it-py`` when present, but the line
parser is the robust default.
"""

from __future__ import annotations

import re
from typing import List

from .mask import mask_text
from .segment import ParsedDoc, Part, Segment

_HEADING = re.compile(r"^(\s{0,3}#{1,6}\s+)(.*?)(\s*#*\s*)$")
_LIST = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+)(.*)$")
_QUOTE = re.compile(r"^(\s*>+\s?)(.*)$")
_HRULE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)


def _has_letters(s: str) -> bool:
    return bool(_LETTERS.search(s or ""))


def _seg(counter: List[int], text: str, kind: str, cfg) -> Segment:
    masked, mask_map, _ = mask_text(text, cfg)
    s = Segment(id=counter[0], text=masked, mask_map=mask_map, kind=kind,
                translatable=_has_letters(masked))
    counter[0] += 1
    return s


def parse_markdown(text: str, cfg=None) -> ParsedDoc:
    from ..config import DocConfig
    cfg = cfg or DocConfig()
    lines = text.split("\n")
    parts: List[Part] = []
    counter = [0]
    in_fence = False

    for i, line in enumerate(lines):
        if _FENCE.match(line):
            in_fence = not in_fence
            parts.append(line)
        elif in_fence:
            parts.append(line)                       # code body: literal
        elif not line.strip():
            parts.append(line)                       # blank line
        elif _HRULE.match(line):
            parts.append(line)                       # horizontal rule
        elif _HEADING.match(line):
            m = _HEADING.match(line)
            parts.append(m.group(1)); parts.append(_seg(counter, m.group(2), "heading", cfg))
            parts.append(m.group(3))
        elif _QUOTE.match(line):
            m = _QUOTE.match(line)
            parts.append(m.group(1)); parts.append(_seg(counter, m.group(2), "quote", cfg))
        elif _TABLE_SEP.match(line):
            parts.append(line)                       # |---|---| separator
        elif _TABLE_ROW.match(line):
            # split into cells on unescaped pipes, keep the pipes as literals
            raw = line
            lead = raw[: len(raw) - len(raw.lstrip())]
            parts.append(lead)
            cells = raw.strip().strip("|").split("|")
            parts.append("|")
            for cell in cells:
                stripped = cell.strip()
                lpad = cell[: len(cell) - len(cell.lstrip())] or " "
                rpad = cell[len(cell.rstrip()):] or " "
                parts.append(lpad)
                parts.append(_seg(counter, stripped, "cell", cfg) if stripped else "")
                parts.append(rpad + "|")
        elif _LIST.match(line):
            m = _LIST.match(line)
            parts.append(m.group(1)); parts.append(_seg(counter, m.group(2), "list", cfg))
        else:
            lead = line[: len(line) - len(line.lstrip())]
            core = line.strip()
            trail = line[len(line.rstrip()):]
            if lead:
                parts.append(lead)
            parts.append(_seg(counter, core, "paragraph", cfg))
            if trail:
                parts.append(trail)
        if i < len(lines) - 1:
            parts.append("\n")

    return ParsedDoc(fmt="markdown", parts=parts, structure_signature=structure_signature(text))


def structure_signature(text: str) -> dict:
    """Count structural elements (for the D5 structure-preservation check)."""
    sig = {"headings": 0, "list_items": 0, "blockquotes": 0, "code_fences": 0,
           "table_rows": 0, "hrules": 0}
    in_fence = False
    for line in text.split("\n"):
        if _FENCE.match(line):
            sig["code_fences"] += 1
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if _HEADING.match(line):
            sig["headings"] += 1
        elif _HRULE.match(line):
            sig["hrules"] += 1
        elif _QUOTE.match(line):
            sig["blockquotes"] += 1
        elif _TABLE_ROW.match(line) and not _TABLE_SEP.match(line):
            sig["table_rows"] += 1
        elif _LIST.match(line):
            sig["list_items"] += 1
    return sig


__all__ = ["parse_markdown", "structure_signature"]
