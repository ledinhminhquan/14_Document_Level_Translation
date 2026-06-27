# Architecture

**Project P14 — Structured Document-Level Machine Translation** (`doctrans`)
Le Dinh Minh Quan · 23127460 · NLP-in-Industry final assignment

`doctrans` translates a *structured* document (Markdown / HTML / JSON / plain) while preserving its structure byte-stably **and** using document-level context. The only trainable component is the MT model — the NLP heart. Everything around it (structure layer + agent) is deterministic code.

## The localization spine

The end-to-end pipeline is a single linear spine:

```
parse (per format)
  -> extract translatable text + MASK non-translatable spans
    -> translate-with-context (concat-k)
      -> restore masks
        -> reassemble (mutate in place)
          -> validate structure
```

The defining invariant: **the model never sees structure; the structure layer never sees the model.** Leaf prose is translated through opaque sentinels (`[[PHn]]`, ASCII / Windows-cp1252 safe), so the MT core only ever handles plain sentences. The contract that makes the whole system trustworthy is the **D1 round-trip identity check**: re-joining the parsed pieces must reproduce the input byte-for-byte before any translation happens.

## The interleaved skeleton + Segment model

A parsed document is an **interleaved list of literal skeleton strings and translatable `Segment` objects**:

```
[ skeleton_0, Segment_0, skeleton_1, Segment_1, skeleton_2, ... ]
```

Skeleton strings (heading hashes, list markers, table pipes, tags, JSON punctuation) are carried verbatim and never touched. `Segment`s hold translatable prose plus their mask table. Reassembly is **mutate-in-place**: each `Segment.text` is replaced by its translation, then the list is concatenated. With no translation applied, concatenation is the byte-identical round trip (D1).

## Parser degradation ladder

Every parser is dependency-light and degrades native -> regex -> plain:

| Format | Primary parser | Skeleton kept as literal | Optional upgrade |
|--------|---------------|--------------------------|------------------|
| Markdown | line parser | heading hashes, list/blockquote markers, table pipes, fenced code, rules | `markdown-it-py` |
| HTML | regex tokenizer | every `<tag ...>`, `script`/`style` bodies verbatim | `beautifulsoup4` |
| JSON | value walker (`json.dumps`) | keys, numbers, structure | — |
| Plain | paragraph / line splitter | line breaks | — |
| DOCX | — | — | `python-docx` |

JSON translates only string **values**; keys, numbers, and structure are byte-stable through `json.dumps`. If a parser's D1 check fails, the agent degrades to the plain parser; a hard parse failure flags/aborts.

## Masking scheme

Non-translatable spans are replaced with reversible sentinels `[[PHn]]` and recorded `sentinel -> original`: inline code, fenced code, URLs, emails, Markdown link targets, HTML tags, and placeholders (`{name}`, `{{var}}`, `%s`, `${VAR}`, `:named`). Masks are restored verbatim after translation. **D4 placeholder-retention** is a hard gate: every `[[PHn]]` must reappear exactly once.

## Repo layout — `src/doctrans` subpackages

| Subpackage | Responsibility |
|-----------|----------------|
| `data` | dataset loaders (opus-100, news_commentary), synthetic structure wrapping, offline seed |
| `models` | model registry (m2m100-418M default, NLLB / mBART-50 / opus-mt alternatives) |
| `docparse` | parsers, skeleton/Segment model, masking, reassembly, D1/D4 checks |
| `mt` | translation core, concat-k context assembly, baselines (zero-shot / dictionary / identity) |
| `training` | `Seq2SeqTrainer` wrapper (predict_with_generate, chrF metric, resume-safe) |
| `agent` | deterministic FSM (D1-D5), tool contract, `ToolTrace`, optional advisory LLM brain |
| `api` | FastAPI endpoints, Gradio demo |
| `analysis` | chrF / BLEU / d-chrF / SPS, sentence-vs-concat-k ablation |
| `autoreport` | model/data cards, run reports |
| `monitoring` | chrF + SPS + needs-review rate + latency drift tracking |
| `automation` | continual-learning re-fine-tune loop with non-regression gate |
| `grading` | assignment self-check harness |

## Document-level context

Context is **concat-k**: prepend `k` previous source sentences plus a `<BRK>` separator (default `k=2`), train the model to emit only the current target sentence (k-to-1). Context is left-truncated, never the current sentence. Using it at inference requires a context-fine-tuned model (recorded `context: true` in metadata); `news_commentary`'s monotonic per-article ids give real adjacency.

## Offline vs Colab paths

- **Offline** (default, CI, grading): only `numpy` / `scikit-learn` / `pyyaml` / `pydantic`. Heavy deps (`torch`, `transformers`) load lazily. Runs against the seed corpus + an en->fr dictionary; the dictionary baseline and full structure pipeline execute with no torch and no network. Configured via `DOCTRANS_ARTIFACTS_DIR` / `_DATA_DIR` / `_MODEL_DIR` / `_RUN_DIR` and `HF_HOME`. CLI is ASCII-only.
- **Colab / GPU** (H100 / A100 / L4 / T4): full HF fine-tuning (eff. batch ~32 via grad-accum; T4 fp16 only), real m2m100 inference, context fine-tuning on `news_commentary`.

Run `doctrans evaluate` for live numbers (chrF / d-chrF / SPS / placeholder-retention). The structure layer is deterministic and versioned, never trained; structure-breaking or low-confidence output is flagged for human review and fail-soft keeps the source rather than emitting a broken document.
