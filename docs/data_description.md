# Data Description — `doctrans` (P14)

This document describes every data source the **Structured Document-Level Machine
Translation** system touches. The only trainable component is the MT core (the NLP
heart); the structure layer and the agent FSM are deterministic code and require **no**
training data. There is **no public corpus of structured parallel documents**, so the
structure layer is exercised with a synthetic harness and a fully offline seed. For live
numbers on any split, run `doctrans evaluate`.

Default direction is **English -> French**; every dataset below is direction-configurable
(swap the language pair and the model's `src_lang` / `forced_bos_token_id`).

## Data sources at a glance

| Role | Dataset | Pair | Why it is used | Structure | License flag |
|---|---|---|---|---|---|
| Fine-tune (sentence MT) | `Helsinki-NLP/opus-100` (en-fr) | en-fr | Large, clean sentence pairs to specialize the MT core | none (sentence) | **UNKNOWN — flag** |
| Document context | `Helsinki-NLP/news_commentary` (en-fr) | en-fr | Monotonic per-article ordering gives **real** concat-k context | per-article order | **UNKNOWN — flag** |
| Structure layer | Synthetic (generated) | en-fr | Wrap sentence pairs in Markdown / HTML / JSON / DOCX | full | derived (inherits source flag) |
| Offline seed | Bundled in-repo | en-fr | Run end-to-end with no `torch` / no network | small Markdown | repo-internal (clean) |
| Permissive alt. | ParaCrawl (CC0) | en-fr | License-clean fallback | none, **no doc order** | CC0 |

## Fine-tune corpus — `opus-100` en-fr

The translation field carries `{en, fr}` sentence pairs (~1M). This is the workhorse for
specializing the MT core (default `facebook/m2m100_418M`) at the **sentence** level. It
contains no document structure and no discourse ordering, so it trains fluency and
adequacy only — not cross-sentence context.

## Document-context corpus — `news_commentary` en-fr

Document-level context is learned here. Each row carries a **monotonic per-article id**,
so consecutive rows are adjacent sentences of the same article. The **concat-k** builder
walks these in order, prepends the `k` previous source sentences plus a `<BRK>`
separator, and trains the model to emit **only** the current target sentence (k-to-1,
default `k=2`, context **left-truncated** so the current sentence is never cut). A model
trained this way is tagged `context: true` in its metadata; inference only uses context
against such a model. Honest caveat: when doc-parallel data is scarce, concat-k can
underperform sentence-level on raw BLEU — the win shows up on discourse phenomena
(see `docs/evaluation.md`).

## Synthetic structure layer

No HF corpus pairs *structured* documents. So the structure harness takes verified
sentence pairs and **wraps** them in Markdown, HTML, JSON, and DOCX skeletons —
headings, lists, tables, code blocks, links, inline markup, and placeholders
(`{name}`, `{{var}}`, `%s`, `${VAR}`, `:named`). This is what stresses the localization
spine and the structure tests: D1 round-trip identity (re-join is byte-for-byte), D4
placeholder-retention (every `[[PHn]]` returns exactly once), and the
Structure-Preservation Score.

## Offline seed (no torch, no network)

A small bundle of Markdown docs + gold French + an en->fr lookup dictionary ships in the
repo so the **entire** pipeline — parse, mask, translate (dictionary backend), restore,
reassemble, validate — runs with only `numpy / scikit-learn / pyyaml / pydantic`. Heavy
deps are lazy. This guarantees CI and reviewers can reproduce results with zero downloads.

## License hygiene (consolidated)

- `opus-100` en-fr — **license UNKNOWN: flagged.**
- `news_commentary` en-fr — **license UNKNOWN: flagged.**
- NLLB checkpoint / IWSLT2017 — **CC-BY-NC (-ND for IWSLT): flagged**, non-commercial.
- ParaCrawl — **CC0** (permissive) but **no document order**, so unusable for concat-k.

**Clean release path:** `facebook/m2m100_418M` (MIT) + the bundled offline seed —
fully permissive, network-free, and sufficient to demonstrate every claim. Any run that
touches a flagged corpus is recorded in run metadata for license review.
