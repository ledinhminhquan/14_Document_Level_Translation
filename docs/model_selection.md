# Model Selection & Optimization

This document records the model choices for **doctrans** (P14). The only trainable component is the MT core; the structure layer (parse -> mask -> reassemble -> validate) and the agent FSM are deterministic code. The MT model never sees document structure — it translates leaf prose through opaque sentinels (`[[PHn]]`).

## The trainable MT core

**Default: `facebook/m2m100_418M` (MIT).** Chosen for: a permissive license (clean dependency path), a 1024-position budget that leaves headroom for prepended document context, and direct multilingual control (set `src_lang`, force `forced_bos_token_id` for the target). It reuses the P13 s2st seq2seq plumbing, so training and generation code are shared.

### Alternatives

| Model | License | Why / when | Notes |
|---|---|---|---|
| `facebook/m2m100_418M` | MIT | **Default** — clean license, 1024 positions | src_lang + forced_bos_token_id |
| `nllb-200-distilled-600M` | CC-BY-NC (flag) | Higher quality, 200 langs | FLORES language codes; non-commercial |
| `mbart-large-50` | permissive | mBART-50 many-to-many alternative | larger footprint |
| `opus-mt-en-fr` | Apache-2.0 | Single-pair, 75M, fast | 512 positions — tight for context |

### Baselines (floors to beat)

- **Zero-shot MT** — the un-fine-tuned model; the floor a fine-tune must beat.
- **Dictionary word-lookup** — offline en->fr dictionary; runs with no torch/network.
- **Identity (copy-source)** — absolute floor.

Verified offline: dictionary MT chrF ~82 vs identity floor ~21. Run `doctrans evaluate` for live numbers.

## Document-level context: concat-k

Document context is supplied as **concat-k**: prepend the k previous source sentences plus a `<BRK>` separator, and train the model to emit **only** the current target sentence (k-to-1, default **k=2** source-side). Context is **left-truncated** so the current sentence is never cut — this is why the 1024-position budget matters (m2m100 has the headroom; `opus-mt`'s 512 does not). Inference uses context only when the model is **context-fine-tuned** (`context: true` in metadata); a sentence-level model is run sentence-by-sentence. The honest caveat: concat-k can underperform when document-parallel data is scarce, so we report a 2x2 sentence-vs-concat-k ablation rather than claiming a universal win. Discourse gains show up on ContraPro-style contrastive sets and term-consistency, not raw BLEU.

## GPU profile

| GPU | Precision | Effective batch |
|---|---|---|
| H100 / A100 | bf16 + tf32 | ~32 via grad-accum |
| L4 | bf16 | ~32 via grad-accum |
| T4 | fp16 only | ~32 via grad-accum |

## Seq2SeqTrainer settings

HF `Seq2SeqTrainer` with:

- `predict_with_generate=True`; **metric for best model = chrF** (headline; BLEU reported alongside).
- **Label smoothing** for generalization.
- **bf16/tf32** on Ampere+ (fp16 on T4); effective batch ~32 via gradient accumulation.
- **Resume-safe** checkpointing.
- **`group_by_length`** to cut padding waste — important once context is prepended and lengths vary widely.

Fine-tune data: `Helsinki-NLP/opus-100` en-fr (license UNKNOWN flag). Document-context data: `Helsinki-NLP/news_commentary` en-fr, whose monotonic per-article ids make consecutive rows adjacent sentences — real concat-k context. No HF corpus of structured parallel documents exists, so structure is synthetic (sentence pairs wrapped in Markdown/HTML/DOCX). Clean license path = m2m100 (MIT) + the offline seed.
