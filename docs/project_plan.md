# Project Plan & Teamwork — `doctrans` (P14)

**Project:** Structured Document-Level Machine Translation
**Author:** Le Dinh Minh Quan (student 23127460) — single-author project
**Package:** `doctrans` · **Default direction:** English -> French (configurable)

## 1. Goal in one sentence

Translate a structured document (Markdown / HTML / JSON / plain) **byte-stably preserving structure** (headings, lists, tables, code blocks, links, inline markup, placeholders) while using **document-level context**, with the MT model as the only trainable component and a deterministic structure layer + agent around it.

## 2. Build order (dependency-ordered)

The build follows the localization spine, bottom-up, so each layer is testable before the next depends on it.

| # | Stage | Modules | Exit criterion |
|---|-------|---------|----------------|
| 0 | Foundations | config (env-var paths), logging, model/run registry, offline seed (small Markdown docs + gold French + en->fr dictionary) | runs offline with only `numpy`/`scikit-learn`/`pyyaml`/`pydantic`; seed loads |
| 1 | Structure layer | `docparse` — per-format parsers (degradation ladder native->regex->plain), masking ([[PHn]] sentinels), reassemble | **D1 round-trip identity** (re-join == source byte-for-byte) on seed |
| 2 | MT core + context | m2m100_418M wrapper (src_lang + forced_bos), baselines (zero-shot / dictionary / identity), concat-k context builder (k=2, `<BRK>`, k-to-1, left-truncate) | dictionary MT beats identity floor on seed |
| 3 | Training + eval | `Seq2SeqTrainer` (predict_with_generate, label smoothing, metric=chrF, resume-safe, group_by_length); chrF/BLEU, d-chrF, SPS; 2x2 ablation (sentence vs concat-k) | `doctrans evaluate` emits versioned JSON metrics |
| 4 | Agent | deterministic FSM D1–D5, uniform tool contract, ToolTrace audit; optional LLM advisory brain (OFF) | end-to-end translate with full decision trace |
| 5 | API / UI / CLI | FastAPI (`/translate-text`, `/translate-document`, `/healthz`, `/version`), Gradio demo, ASCII-only CLI | endpoints return doc + structure report + trace |
| 6 | Reports / ops | monitoring (chrF + SPS + needs-review + latency), continual-learning loop, grading harness, autopilot | non-regression gate green; reports reproduce |

## 3. Reproducibility

- **Paths via env vars:** `DOCTRANS_ARTIFACTS_DIR`, `_DATA_DIR`, `_MODEL_DIR`, `_RUN_DIR`, `HF_HOME` — no hard-coded locations.
- **Versioned JSON:** every metric/run/model carries a `model_version` and (for context models) `context:true`; outputs are diffable JSON, not free text.
- **Offline-testable:** the seed (Markdown + gold French + dictionary) lets the full spine and eval run with **no torch and no network**; heavy deps (`transformers`, `torch`) are lazy-imported.
- **CI without GPU:** D1–D5 gates, round-trip identity, placeholder-retention, and SPS run on the dictionary/identity baselines, so CI is green on CPU. GPU profiles (H100/A100/L4/T4) only affect real fine-tuning, not correctness tests.
- **Deterministic structure layer:** versioned code, not trained — reproducibility of structure is a code-level guarantee, independent of model weights.

## 4. Risks & honest caveats

- No HF corpus of structured parallel documents exists -> structure is **synthetic** (wrap sentence pairs in Markdown/HTML/DOCX); call this out in reports.
- **concat-k can underperform** when doc-parallel data is scarce — the document-level win shows on ContraPro-style contrastive / term-consistency sets, not raw BLEU.
- License hygiene: opus-100 / news_commentary licenses UNKNOWN (flagged); NLLB / IWSLT are CC-BY-NC. Clean path = **m2m100 (MIT) + seed**.
- **Fail-soft:** structure-breaking or low-confidence output is FLAGGED for human review and the source is kept, never presented as final.

## 5. Teamwork

Single-author build (Le Dinh Minh Quan, 23127460). The plan is sequenced so each stage is independently verifiable; refer to running **`doctrans evaluate`** for live numbers rather than any figure quoted from memory.
