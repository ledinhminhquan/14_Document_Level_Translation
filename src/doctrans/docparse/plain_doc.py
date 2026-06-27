"""Plain-text parser → paragraph segments (the ultimate fallback, stdlib only).

Splits on lines; each non-blank line with letters becomes a translatable segment (inline
non-translatables masked), blank lines and code/URL-looking lines stay literal. Re-joining
reproduces the text exactly. This is the floor every other format degrades to.
"""

from __future__ import annotations

import re
from typing import List

from .mask import mask_text
from .segment import ParsedDoc, Part, Segment

_LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)
_URL_LINE = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$")


def _has_letters(s: str) -> bool:
    return bool(_LETTERS.search(s or ""))


def parse_plain(text: str, cfg=None) -> ParsedDoc:
    from ..config import DocConfig
    cfg = cfg or DocConfig()
    lines = text.split("\n")
    parts: List[Part] = []
    counter = [0]
    for i, line in enumerate(lines):
        if line.strip() and _has_letters(line) and not _URL_LINE.match(line):
            lead = line[: len(line) - len(line.lstrip())]
            core = line.strip()
            trail = line[len(line.rstrip()):]
            if lead:
                parts.append(lead)
            masked, mask_map, _ = mask_text(core, cfg)
            parts.append(Segment(id=counter[0], text=masked, mask_map=mask_map, kind="paragraph"))
            counter[0] += 1
            if trail:
                parts.append(trail)
        else:
            parts.append(line)
        if i < len(lines) - 1:
            parts.append("\n")
    return ParsedDoc(fmt="plain", parts=parts,
                     structure_signature={"lines": len(lines),
                                          "nonblank": sum(1 for l in lines if l.strip())})


__all__ = ["parse_plain"]
