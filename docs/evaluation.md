# Evaluation Protocol & Baselines

This document defines how **doctrans** is measured: translation quality, document-level
quality, structure fidelity, and the discourse story behind concat-k context. All numbers
below are illustrative of the verified offline seed run; for live figures run
`doctrans evaluate`, which prints every metric in this document against the configured
test set.

## What we measure (and why)

| Layer | Metric | Type | Gate |
| --- | --- | --- | --- |
| Sentence MT | **chrF** (headline) | soft | reported |
| Sentence MT | BLEU | soft | reported |
| Document MT | **d-chrF** (document-level chrF) | soft | reported |
| Structure | **SPS** = struct-match x markup-validity x placeholder-retention | hard/soft | SPS = 1.0 target |
| Discourse | ContraPro-style accuracy + term-consistency | soft | context-vs-no-context delta |

**chrF is the headline.** It is the character-n-gram F-score (sacrebleu, with a
pure-python fallback so the metric runs with no torch/network). chrF correlates better
than BLEU with human judgement for morphologically rich targets like French and is robust
on short leaf segments. BLEU is reported alongside for comparability with prior work, not
as the decision metric.

## Document-level chrF (d-chrF)

Sentence chrF averages over independent leaves and hides cross-segment errors. **d-chrF**
scores the *reassembled* translatable text of the whole document against a gold document,
so discourse drift, pronoun/gender mismatches, and terminology flips that span segments
actually count. The verified offline seed reaches **document chrF ~81** at full structure
fidelity.

## Structure-Preservation Score (SPS)

SPS is the product of three factors, so any structural failure drags the whole score down:

- **struct-match** — the output structure signature (heading/list/table/code skeleton)
  equals the source signature after translation. This is the D5 structure-validate gate.
- **markup-validity** — the output re-parses cleanly in its own format (valid Markdown /
  well-formed HTML / valid JSON).
- **placeholder-retention** — every `[[PHn]]` sentinel returns exactly once (the D4 HARD
  gate). Verified offline: **placeholder-retention 1.0**.

The seed run reports **SPS 1.0**: structure is byte-stable because the model never sees it
(leaf prose only) and reassembly is the inverse of parsing (the D1 round-trip identity).

## The context-vs-no-context story

The whole point of concat-k is **discourse**, and BLEU/chrF are largely **blind** to it:
fixing one pronoun or one term in a long sentence barely moves an n-gram score. We
therefore evaluate context with targeted signals:

- **ContraPro-style contrastive accuracy** — for each item the model must rank the
  discourse-correct target above a near-identical distractor (e.g. correct French
  grammatical gender/number for an antecedent introduced in a *previous* sentence). A
  context model should win where a sentence-level model cannot.
- **Term-consistency** — does a term introduced earlier stay translated the same way later
  in the document?

**Honest caveat:** concat-k needs a context-fine-tuned model (`context:true` in metadata,
trained on `news_commentary` where consecutive rows are adjacent sentences). When
doc-parallel data is scarce, concat-k can *underperform* sentence-level on raw BLEU/chrF
while still fixing discourse cases. We report this rather than hide it.

## The 2x2 ablation + floors

Run `doctrans evaluate` to produce the core ablation:

| | Sentence-level | concat-k (k=2) |
| --- | --- | --- |
| **Zero-shot MT** | floor to beat | context w/o fine-tune |
| **Fine-tuned MT** | strong sentence baseline | full system (context:true) |

**Floors** (must be beaten, all run offline):

- **identity** — copy source as "translation"; verified chrF **~21**.
- **dictionary** — offline word-lookup en->fr; verified chrF **~82** (a strong lexical
  floor on the seed; beats it for fluency/agreement, not vocabulary).
- **zero-shot MT** — the untuned model; the floor the fine-tuned system must clear.

## How to reproduce

`doctrans evaluate` loads the test set, runs each baseline + the full agent pipeline,
and emits chrF, BLEU, d-chrF, SPS (with its three factors), the ContraPro-style accuracy,
term-consistency, and the 2x2 ablation table — all reproducible offline from the seed
(small Markdown docs + gold French + en->fr dictionary).
