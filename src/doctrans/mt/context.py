"""Document-level concat-k context for the MT core.

The proven, architecture-free way to give a sentence-level MT model document context
(Tiedemann & Scherrer 2017): prepend the ``k`` previous sentences to the current one,
joined by a ``<BRK>`` separator, and train/decode the model to output **only the current
target sentence** (the "k-to-1" recipe). This module builds those context-augmented
source strings, left-truncating the *context* (never the current sentence) to fit the
model's position budget.
"""

from __future__ import annotations

from typing import List, Optional


def build_context_source(sentences: List[str], i: int, k: int, brk: str = "<BRK>") -> str:
    """Source string for sentence ``i`` = (up to) k previous sentences + ``i``, BRK-joined."""
    start = max(0, i - k)
    ctx = sentences[start:i]
    cur = sentences[i]
    if not ctx:
        return cur
    return (f" {brk} ".join(ctx) + f" {brk} " + cur)


def build_pairs_with_context(src_sentences: List[str], tgt_sentences: List[str], k: int,
                             brk: str = "<BRK>") -> List[dict]:
    """Training examples: each {source: ctx+current, target: current-target} (k-to-1)."""
    pairs = []
    n = min(len(src_sentences), len(tgt_sentences))
    for i in range(n):
        pairs.append({"source": build_context_source(src_sentences, i, k, brk),
                      "target": tgt_sentences[i]})
    return pairs


def translate_with_context(translator, sentences: List[str], k: int = 0, brk: str = "<BRK>",
                           use_context: Optional[bool] = None) -> List[str]:
    """Translate each sentence, optionally conditioning on its k predecessors.

    ``use_context`` defaults to True only for a transformer translator that was trained
    with the ``<BRK>`` separator; the dictionary baseline ignores context (k=0 effect).
    """
    if use_context is None:
        use_context = k > 0 and getattr(translator, "name", "") == "transformer"
    if not use_context or k <= 0:
        return translator.translate_batch(list(sentences))
    srcs = [build_context_source(sentences, i, k, brk) for i in range(len(sentences))]
    return translator.translate_batch(srcs)


__all__ = ["build_context_source", "build_pairs_with_context", "translate_with_context"]
