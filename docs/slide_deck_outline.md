# Slide Deck Outline — Structured Document-Level Machine Translation (doctrans)

P14 — NLP in Industry final assignment. Le Dinh Minh Quan (23127460).
~12 slides, ~15 minutes. Speaker talking points per slide. All numbers below are illustrative defaults; show **live figures** by running `doctrans evaluate` before the talk.

---

## Slide 1 — Title

- **Structured Document-Level Machine Translation** — translate a whole document, keep its structure byte-stable.
- doctrans, P14. Author: Le Dinh Minh Quan (23127460).
- One-line hook: "The model translates prose; deterministic code guards the structure; an FSM agent decides when to trust the result."
- Default direction English -> French (configurable).

## Slide 2 — Business Problem

- Real localization is not sentence-in / sentence-out: docs are Markdown, HTML, JSON, DOCX with headings, lists, tables, code, links, placeholders.
- Naive MT **breaks structure** (mangles `{name}`, drops `<tag>`, reflows tables) and **ignores context** (inconsistent terminology, wrong pronoun/gender across sentences).
- Cost: manual re-formatting and review erase MT's speed gains.
- Need: structure preserved **byte-for-byte** + **document-level** consistency + a confidence gate that flags bad output instead of shipping it.

## Slide 3 — Proposed Solution

- A **localization spine**: parse -> extract text + mask non-translatable spans -> translate-with-context -> restore masks -> reassemble -> validate.
- Clean separation: the **model never sees structure**; the **structure layer never sees the model** (prose flows through opaque sentinels `[[PHn]]`).
- Only the **MT model is trainable** (the NLP heart); structure layer + agent are deterministic, versioned code.
- Guarantee: re-joining literal skeleton + translated segments reproduces the document exactly (round-trip identity).

## Slide 4 — System Architecture

- Document = interleaved list of **literal skeleton strings** + translatable **Segment** objects.
- Parsers (degradation ladder native -> regex -> plain, dependency-light): Markdown line parser; HTML regex tokenizer (keeps every tag + script/style verbatim); JSON (translate string **values** only); plain (paragraph/line).
- **Masking** (reversible `[[PHn]]`, cp1252-safe): inline/fenced code, URLs, emails, link targets, HTML tags, placeholders `{name} {{var}} %s ${VAR} :named`.
- Optional upgrades: markdown-it-py / beautifulsoup4 / python-docx. Heavy deps lazy; core runs on numpy/scikit-learn/pyyaml/pydantic.

## Slide 5 — Data & Languages

| Role | Dataset | Why |
|---|---|---|
| Fine-tune MT | Helsinki-NLP/opus-100 en-fr (~1M) | large parallel pairs (license: UNKNOWN flag) |
| Document context | Helsinki-NLP/news_commentary en-fr | per-article ids -> consecutive rows are **real** adjacent sentences for concat-k |
| Eval (context) | FLORES / ContraPro-style | discourse / contrastive checks |

- **No HF corpus of structured parallel documents exists** -> structure is **synthetic** (wrap sentence pairs in Markdown/HTML/DOCX).
- **Offline seed**: small Markdown docs + gold French + an en->fr dictionary -> everything runs with no torch / no network.
- License hygiene: opus-100 / news_commentary unknown; NLLB / IWSLT CC-BY-NC flagged; permissive ParaCrawl (CC0) lacks doc order.

## Slide 6 — Trainable MT Core + Document Context

- Default model **facebook/m2m100_418M** (MIT, 1024 positions — headroom for context; `src_lang` + `forced_bos_token_id`). Alts: nllb-200-distilled-600M, mbart-50, opus-mt-en-fr (Apache, 75M).
- Training: HF Seq2SeqTrainer, `predict_with_generate`, label smoothing, **metric = chrF**, bf16/tf32, resume-safe, group_by_length; eff. batch ~32 via grad-accum (T4 fp16 only).
- **Document-level context = concat-k**: prepend k previous source sentences + `<BRK>`, train to emit **only the current target** (k-to-1, default k=2). Left-truncate context, never the current sentence.
- Inference uses context only with a **context-fine-tuned** model (`context:true` in metadata).

## Slide 7 — Agentic Component (deterministic FSM, D1-D5)

- **D1 Parse**: detect format + parse; **round-trip identity** check; degrade to plain on mismatch; hard failure -> flag/abort.
- **D2 Segment + Mask gate**: skip non-translatable segments (letter-ratio < 0.30); masks reversible & balanced.
- **D3 Translate**: with k context; reject empty / length-ratio outside [0.4, 3.0]; re-translate budget N=2.
- **D4 Verify gate**: placeholder-retention **HARD** (every `[[PHn]]` back exactly once); back-translation chrF **soft >= 0.30** else flag.
- **D5 Reassemble + validate**: structure signature match + output re-parses; repair budget M=1; else flag, **fail-soft keep source**.
- Uniform tool contract + **ToolTrace** audit. Optional LLM brain (anthropic, OFF) writes an advisory terminology note ONLY — never edits text, structure, or placeholders.

## Slide 8 — Evaluation Results

- Headline: **chrF** (+ BLEU) on MT; **document-level chrF (d-chrF)**; **Structure-Preservation Score** SPS = struct-match x markup-validity x placeholder-retention.
- Baselines to beat: zero-shot MT (floor), dictionary lookup (offline floor), identity copy-source (floor).
- Ablation **2x2**: sentence-level vs concat-k, context vs no-context.
- Verified offline seed: dictionary MT chrF ~82 vs identity floor ~21; **document chrF ~81 at SPS 1.0, placeholder-retention 1.0**.
- Honest caveat: context gains show on **contrastive / term-consistency** sets, not raw BLEU; concat-k can underperform when doc-parallel data is scarce.
- Live numbers: run `doctrans evaluate`.

## Slide 9 — Deployment

- **FastAPI**: `POST /translate-text` (`{text, fmt?}` -> translated doc + structure report + decision trace); `POST /translate-document` (file upload, registered only when python-multipart present); `GET /healthz`, `GET /version`.
- **Gradio** demo; **Docker** + **HF Space**.
- Metadata-only request logging (no document bodies stored).
- Config via env: `DOCTRANS_ARTIFACTS_DIR/_DATA_DIR/_MODEL_DIR/_RUN_DIR`, `HF_HOME`. CLI is ASCII-only.

## Slide 10 — Continual Learning & Monitoring

- Loop: collect domain docs/segments + references -> re-fine-tune MT (resume-safe) -> **chrF non-regression gate** -> new `model_version`.
- Structure layer is **deterministic code** — versioned, not trained.
- Monitor: chrF + SPS + **needs-review rate** + latency.
- Drift signal: rising needs-review / falling SPS triggers retrain or rule fix.

## Slide 11 — Ethics, Privacy & Risks

- doctrans is a **localization aid**: structure-breaking or low-confidence output is **flagged for human review**, never presented as final.
- **Fail-soft** keeps the source rather than emitting a broken document.
- License hygiene: clean path = **m2m100 (MIT) + offline seed**; CC-BY-NC and unknown-license corpora are flagged.
- Privacy: metadata-only logging; offline mode needs no network. Optional LLM brain is advisory-only and OFF by default.

## Slide 12 — Takeaways & Future Work

- Takeaway: separating an **opaque-sentinel structure layer** from a **trainable MT core**, gated by a deterministic D1-D5 FSM, gives byte-stable structure + document-level context with an honest confidence story.
- Future: stronger doc-parallel data for concat-k; richer DOCX/PPTX/XLIFF parsers; more language pairs; learned segment-skip; per-domain terminology memory.
- Reproducible offline: dictionary + seed docs make the whole pipeline runnable with no torch / no network.
- Close: "Show, don't tell — run `doctrans evaluate` for the live board."
