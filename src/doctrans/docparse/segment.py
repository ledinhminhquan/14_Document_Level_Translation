"""Core data model for the structure layer: ``Segment`` + ``ParsedDoc``.

A document is parsed into an **interleaved list of parts**: literal skeleton strings
(structure — never translated) and :class:`Segment` objects (the translatable leaf text,
with non-translatable spans already masked to sentinels). The model only ever sees
``Segment.text``; reassembly joins the literals with each segment's (unmasked)
translation in document order, so structure is byte-stable.

This is the universal interface across every format (Markdown / HTML / JSON / plain);
each parser just produces the parts list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union


@dataclass
class Segment:
    id: int
    text: str                                  # translatable text, masked spans -> sentinels
    translatable: bool = True
    mask_map: Dict[str, str] = field(default_factory=dict)   # sentinel -> original span
    kind: str = "text"                          # heading | list | paragraph | cell | value | ...
    translation: str = ""                       # filled after MT (already unmasked)

    def output(self) -> str:
        """The text to emit on reassembly: the translation if present, else the source."""
        if self.translation:
            return self.translation
        return restore_text(self.text, self.mask_map) if self.mask_map else self.text

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "translatable": self.translatable,
                "text": self.text, "translation": self.translation,
                "n_masks": len(self.mask_map)}


Part = Union[str, Segment]


@dataclass
class ParsedDoc:
    fmt: str
    parts: List[Part]
    structure_signature: Dict[str, int] = field(default_factory=dict)

    @property
    def segments(self) -> List[Segment]:
        return [p for p in self.parts if isinstance(p, Segment)]

    @property
    def translatable_segments(self) -> List[Segment]:
        return [s for s in self.segments if s.translatable]

    def reassemble(self) -> str:
        out: List[str] = []
        for p in self.parts:
            out.append(p if isinstance(p, str) else p.output())
        return "".join(out)


def restore_text(text: str, mask_map: Dict[str, str]) -> str:
    """Replace every sentinel in ``text`` with its original span (longest sentinel first)."""
    for sent in sorted(mask_map, key=len, reverse=True):
        text = text.replace(sent, mask_map[sent])
    return text


__all__ = ["Segment", "ParsedDoc", "Part", "restore_text"]
