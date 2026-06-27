# Model Card: doctrans (Structured Document-Level Machine Translation)

**System:** `doctrans` (P14) - Structured, document-level MT for localization
**Author:** Le Dinh Minh Quan (student 23127460), NLP-in-Industry final assignment
**Default direction:** English -> French (configurable)
**Card scope:** the end-to-end document-translation system AND its single trainable component, the fine-tuned MT core.

> All numbers below are illustrative of the verified offline path. Run `doctrans evaluate` for live, current metrics.

---

## 1. Model Details

`doctrans` is a localization *system*, not a single model. Only one component is trainable: the **MT core**. Everything around it (parsers, masking, the agent FSM) is deterministic, versioned code.

| Component | Type | Trainable | Notes |
|---|---|---|---|
| MT core | seq2seq Transformer | **Yes** | default `facebook/m2m100_418M` (MIT, 1024 positions) |
| Structure layer | deterministic | No | parse -> mask -> restore -> reassemble; D1 round-trip identity |
| Agent | deterministic FSM | No | 5 decisions D1-D5, uniform tool contract + ToolTrace |
| LLM brain (optional) | external API | No | OFF by default; advisory terminology note only |

**MT core alternatives:** `nllb-200-distilled-600M` (CC-BY-NC flag, FLORES codes), `mbart-50`, single-pair `opus-mt-en-fr` (Apache, 75M, 512 positions). Training: HF `Seq2SeqTrainer` (`predict_with_generate`, label smoothing, metric=chrF, bf16/tf32, resume-safe, `group_by_length`).

**Document-level context (concat-k):** prepend k previous source sentences + a `<BRK>` separator; the model is trained to output ONLY the current target sentence (k-to-1, default k=2, left-truncate context never the current sentence). Requires a context-fine-tuned model (`context: true` in metadata) to use at inference.

## 2. Intended Use

- **Primary:** translate a structured document (Markdown / HTML / JSON / plain) while preserving structure byte-stably and exploiting document context.
- **Users:** localization engineers, doc/content teams.
- **Out of scope:** safety-critical or legal-binding translation presented as final without human review; languages/pairs the MT core was not fine-tuned on; free-form chat. The system is a localization **aid** - low-confidence or structure-breaking output is FLAGGED, never presented as final.

## 3. Factors

| Factor | Coverage |
|---|---|
| Formats | Markdown, HTML, JSON, plain (degradation ladder native -> regex -> plain) |
| Languages | en -> fr default; configurable via `src_lang` + `forced_bos_token_id` |
| Non-translatable spans | inline/fenced code, URLs, emails, link targets, HTML tags, placeholders (`{name}`, `{{var}}`, `%s`, `${VAR}`, `:named`) masked as `[[PHn]]` |
| Context regime | sentence-level vs concat-k (2x2 ablation) |

## 4. Metrics

| Metric | Role | Verified offline |
|---|---|---|
| chrF | headline MT quality | dictionary MT ~82 vs identity floor ~21 |
| BLEU | secondary MT quality | sacrebleu + pure-python fallback |
| d-chrF | document-level chrF | document ~81 |
| SPS | Structure-Preservation Score = struct-match x markup-validity x placeholder-retention | 1.0 |
| placeholder-retention | every `[[PHn]]` back exactly once (D4 HARD gate) | 1.0 |

## 5. Training Data (and license flags)

| Dataset | Use | License flag |
|---|---|---|
| `Helsinki-NLP/opus-100` en-fr (1M) | MT fine-tune | UNKNOWN |
| `Helsinki-NLP/news_commentary` en-fr | document-context (per-article id -> REAL concat-k) | UNKNOWN |
| FLORES / ContraPro-style | context/contrastive eval | - |
| Synthetic structured docs | structure eval (no HF corpus of parallel structured docs exists) | derived |
| Offline seed (small Markdown + gold fr + en->fr dictionary) | runs with no torch/network | local |

Permissive alternative: ParaCrawl (CC0, no doc order). IWSLT2017 carries a CC-BY-NC-ND flag. Clean license path = m2m100 (MIT) + offline seed.

## 6. Quantitative Analyses

- **Baselines to beat:** zero-shot MT (floor), dictionary word-lookup (offline floor), identity copy-source (~chrF 21).
- **Ablation:** sentence-level vs concat-k x with/without context.
- **Context story:** discourse fixes (pronouns, lexical consistency) surface on ContraPro-style contrastive sets and term-consistency, NOT raw BLEU. Honest caveat: concat-k can *underperform* when doc-parallel data is scarce.

## 7. Ethical Considerations & Caveats

- **Fail-soft:** D5 keeps the source rather than emit a broken document.
- **Human-in-the-loop:** structure-breaking or low-confidence (length-ratio outside [0.4, 3.0], back-translation chrF < 0.30) output is flagged for review.
- **License hygiene:** opus-100 / news_commentary licenses are UNKNOWN; NLLB / IWSLT are CC-BY-NC. Prefer the m2m100 (MIT) + seed path for redistribution.
- **Optional LLM brain** never rewrites a translation, changes structure, or alters placeholders - advisory terminology note only.
- **Logging:** metadata-only request logging.

For current figures, run `doctrans evaluate`.
