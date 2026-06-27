"""HTML parser → interleaved skeleton + translatable text nodes (dependency-free).

A regex tokenizer splits the document into tags (kept verbatim as literal skeleton —
so every ``<tag ...>`` survives byte-for-byte, including attributes like ``href``) and
the text between them (emitted as :class:`Segment` objects with inline non-translatables
masked). ``<script>`` / ``<style>`` bodies are kept literal. Re-joining reproduces the
document exactly. Uses only the stdlib; ``beautifulsoup4`` is an optional speed/robustness
upgrade but not required.
"""

from __future__ import annotations

import re
from typing import List

from .mask import mask_text
from .segment import ParsedDoc, Part, Segment

_TAG = re.compile(r"<[^>]+>|<!--.*?-->", re.DOTALL)
_TAGNAME = re.compile(r"</?\s*([A-Za-z][\w:-]*)")
_LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)
_RAW_TAGS = {"script", "style"}


def _has_letters(s: str) -> bool:
    return bool(_LETTERS.search(s or ""))


def parse_html(text: str, cfg=None) -> ParsedDoc:
    from ..config import DocConfig
    cfg = cfg or DocConfig()
    parts: List[Part] = []
    counter = [0]
    in_raw = False
    pos = 0
    for m in _TAG.finditer(text):
        # text before this tag
        chunk = text[pos:m.start()]
        if chunk:
            if in_raw or not _has_letters(chunk):
                parts.append(chunk)
            else:
                masked, mask_map, _ = mask_text(chunk, cfg)
                parts.append(Segment(id=counter[0], text=masked, mask_map=mask_map, kind="text"))
                counter[0] += 1
        tag = m.group(0)
        parts.append(tag)
        name_m = _TAGNAME.match(tag)
        if name_m:
            name = name_m.group(1).lower()
            if name in _RAW_TAGS:
                in_raw = not tag.startswith("</")
        pos = m.end()
    tail = text[pos:]
    if tail:
        if in_raw or not _has_letters(tail):
            parts.append(tail)
        else:
            masked, mask_map, _ = mask_text(tail, cfg)
            parts.append(Segment(id=counter[0], text=masked, mask_map=mask_map, kind="text"))
    return ParsedDoc(fmt="html", parts=parts, structure_signature=structure_signature(text))


def structure_signature(text: str) -> dict:
    counts: dict = {}
    for tag in _TAG.findall(text):
        m = _TAGNAME.match(tag)
        if m:
            name = m.group(1).lower()
            counts[name] = counts.get(name, 0) + 1
    return counts


__all__ = ["parse_html", "structure_signature"]
