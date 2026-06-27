"""JSON parser → translate only string *values*, never keys/numbers/booleans/structure.

Walks the parsed JSON tree, replaces each translatable string value with a unique
placeholder, and records a flat list of segments + their tree paths. Reassembly writes
the translations back into the (deep-copied) tree and re-serializes with ``json.dumps``,
so keys, numbers, nesting, and ordering are byte-stable. Stdlib only.
"""

from __future__ import annotations

import copy
import json
from typing import Any, List, Tuple

from .mask import mask_text
from .segment import ParsedDoc, Part, Segment

import re
_LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)


def _has_letters(s: str) -> bool:
    return bool(_LETTERS.search(s or ""))


class _JsonDoc(ParsedDoc):
    """A ParsedDoc whose reassembly re-serializes the JSON tree."""

    def __init__(self, obj: Any, segments: List[Segment], paths: List[Tuple], indent):
        super().__init__(fmt="json", parts=list(segments))
        self._obj = obj
        self._paths = paths
        self._indent = indent
        self.structure_signature = {"strings": len(segments)}

    def reassemble(self) -> str:
        tree = copy.deepcopy(self._obj)
        for seg, path in zip(self.segments, self._paths):
            ref = tree
            for key in path[:-1]:
                ref = ref[key]
            ref[path[-1]] = seg.output()
        return json.dumps(tree, ensure_ascii=False, indent=self._indent)


def parse_json(text: str, cfg=None) -> ParsedDoc:
    from ..config import DocConfig
    cfg = cfg or DocConfig()
    obj = json.loads(text)
    indent = 2 if ("\n" in text) else None
    segments: List[Segment] = []
    paths: List[Tuple] = []
    counter = [0]

    def walk(node: Any, path: Tuple):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, path + (k,))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, path + (i,))
        elif isinstance(node, str) and _has_letters(node):
            masked, mask_map, _ = mask_text(node, cfg)
            segments.append(Segment(id=counter[0], text=masked, mask_map=mask_map, kind="value"))
            paths.append(path)
            counter[0] += 1

    walk(obj, tuple())
    return _JsonDoc(obj, segments, paths, indent)


__all__ = ["parse_json"]
