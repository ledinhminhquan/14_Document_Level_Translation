# Data Card -- `doctrans` (P14: Structured Document-Level Machine Translation)

This card documents every data source used by `doctrans`, how each is used, and the honest limitations. The **only** trainable component is the MT core; the structure/localization layer and the agent FSM are deterministic code (versioned, not trained). For live numbers, run `doctrans evaluate`.

## 1. Datasets at a glance

| Dataset id | Role | Task | Size | Key columns | License flag | Usage in `doctrans` |
|---|---|---|---|---|---|---|
| `Helsinki-NLP/opus-100` (en-fr) | MT fine-tune | Sentence MT | ~1M pairs | `translation.{en,fr}` | UNKNOWN (flag) | Primary supervised fine-tuning of the MT core (en->fr). |
| `Helsinki-NLP/news_commentary` (en-fr) | Document context | Doc-level MT | per-article ordered | `id`, `translation.{en,fr}` | UNKNOWN (flag) | Monotonic per-article `id`; consecutive rows are adjacent sentences -> **real concat-k context** (k previous + `<BRK>`). |
| FLORES / ContraPro-style sets | Eval (context) | Contrastive / d-chrF | small | source, references, contrast | research-eval | Discourse evaluation: pronoun/term-consistency fixes from context. |
| **Synthetic structured docs** | Structure eval/train | Structured MT | generated | Markdown/HTML/JSON/DOCX skeletons + segments | derived | Wrap sentence pairs in structure to exercise parse->mask->translate->reassemble. |
| **Offline seed** | Smoke/CI/demo | End-to-end | tiny | Markdown docs + gold FR + en->fr dictionary | local | Runs the whole pipeline with **no torch / no network**. |

> Note: no HuggingFace corpus of **structured parallel documents** exists. Structure is therefore **synthetic** (see Section 2). Permissive alternative `ParaCrawl` (CC0) lacks document order; `IWSLT2017` carries a CC-BY-NC-ND flag -- not on the clean path.

## 2. Synthetic structured-document generation

Because no structured parallel corpus exists, `doctrans` **synthesizes** structured documents by wrapping verified sentence pairs (from opus-100 / news_commentary) into deterministic skeletons:
- **Markdown**: headings, lists, blockquotes, tables, fenced/inline code, links, rules.
- **HTML**: tags + verbatim `script`/`style` bodies around translatable leaf text.
- **JSON**: string **values** are translatable; keys/numbers/structure stay byte-stable.
- **DOCX**: paragraphs wrapping sentence pairs (optional `python-docx` upgrade).

Non-translatable spans (inline/fenced code, URLs, emails, link targets, HTML tags, placeholders `{name}` `{{var}}` `%s` `${VAR}` `:named`) are masked with reversible sentinels `[[PHn]]` (ASCII / Windows-cp1252 safe). This yields gold structure for the **Structure-Preservation Score** (SPS = struct-match x markup-validity x placeholder-retention) and the **D1 round-trip identity** check (skeleton + segments re-join byte-for-byte).

## 3. Offline seed (no torch / no network)

A tiny local bundle: a few **Markdown docs**, their **gold French**, and an **en->fr dictionary**. It powers CI smoke tests, the dictionary baseline, and the Gradio/CLI demo. Verified offline: dictionary MT chrF ~82 vs identity copy-source floor ~21; document chrF ~81 at SPS 1.0 with placeholder-retention 1.0.

## 4. Splits & preprocessing

- **Splits**: standard train/validation/test per source; `group_by_length` for batching; context split keeps article boundaries intact (left-truncate context, never the current sentence).
- **Preprocessing (deterministic)**: format-detect & **parse** (native->regex->plain degradation ladder) -> **extract** translatable leaf text -> **mask** non-translatable spans to `[[PHn]]` -> (optional) **concat-k context** prepend (k=2 source-side, `<BRK>`, k-to-1) -> translate -> **restore** masks -> **reassemble** in-place -> validate. Baselines: zero-shot MT (floor), dictionary lookup (offline floor), identity copy-source.

## 5. Limitations & ethical notes

- **No real structured parallel data** -> structure is synthetic; structure generalization to wild documents is a known gap (fail-soft keeps source on parse/validate failure).
- **Context can underperform**: concat-k may not beat sentence-level when doc-parallel data is scarce; discourse gains show on contrastive/term-consistency sets, not raw BLEU.
- **License hygiene**: opus-100 and news_commentary licenses are **UNKNOWN (flagged)**; NLLB / IWSLT carry **CC-BY-NC** restrictions. Clean redistributable path = **m2m100 (MIT) + offline seed**.
- **Privacy/use**: request logging is metadata-only; `doctrans` is a localization **aid** -- structure-breaking or low-confidence output is **flagged for human review**, never presented as final.
