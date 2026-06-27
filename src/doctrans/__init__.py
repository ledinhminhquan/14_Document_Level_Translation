"""doctrans — Structured Document-Level Machine Translation.

Translate a whole **structured document** (Markdown / HTML / DOCX / JSON / plain text)
from one language to another while **(a) preserving structure byte-stably** (headings,
lists, tables, code blocks, links, inline markup, placeholders) and **(b) using
document-level context** (surrounding sentences) for discourse-consistent output.

    parse (per format) -> extract translatable text + MASK non-translatable spans
        -> translate-with-context (the TRAINABLE MT core, concat-k)
        -> restore masks -> reassemble (mutate-in-place) -> validate structure

The only trainable NLP component is the **MT model** (reusing the P13 `s2st` MT
plumbing). The structure layer and the agent (decisions D1-D5) are deterministic. The
model never sees structure and the structure layer never sees the model — leaf text is
translated through opaque sentinels. Everything degrades gracefully so the package, the
agent, and the tests run on numpy/scikit-learn alone (a dictionary MT + a regex/pure-text
parser stand in for torch/transformers + the markdown/html/docx libraries).
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
