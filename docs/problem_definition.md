# Problem Definition & Business Value

## Problem

Real-world translation rarely operates on bare sentences. It operates on **structured documents**: Markdown READMEs, HTML help pages, JSON localization bundles, product docs. These carry two things a translator must respect simultaneously — the **prose** (which must change language) and the **structure** (headings, lists, tables, fenced code, links, inline markup, and runtime placeholders like `{name}`, `%s`, `${VAR}`) which must survive *byte-stably*.

Feeding such a document straight into a raw MT model corrupts it. The model rewrites code blocks, translates URL targets and `{placeholders}`, drops table pipes, mangles heading hashes, and reflows lists. The output may read fluently yet be **structurally broken** — a localized page that fails to render, an i18n file that breaks string interpolation at runtime, a code sample that no longer compiles. Sentence-by-sentence translation also discards **document-level context**, so pronouns, formality, and terminology drift across the document.

**doctrans** solves both. A deterministic structure layer parses each format, extracts only translatable leaf prose, and masks every non-translatable span behind opaque sentinels `[[PHn]]`. The MT core — the single trainable, NLP component — translates that prose *with k previous sentences as context* (concat-k). Masks are restored and the document is reassembled **in place** so it reproduces byte-for-byte except for the translated text. The model never sees structure; the structure layer never sees the model.

## Why the MT Stage Is the NLP Core

Everything else (parsers, masking, agent FSM, reassembly) is deterministic, testable code. The **only learned component** is the seq2seq MT model (default `facebook/m2m100_418M`, MIT), fine-tuned on opus-100 en-fr and given real document context from news_commentary. Translation quality — and the document-level discourse fixes from concat-k — is where the machine learning lives and where evaluation focuses.

## Jobs To Be Done (by persona)

| Persona | Job to be done | What doctrans gives them |
|---|---|---|
| Localization engineer | Translate a JSON/HTML bundle without breaking interpolation or markup | Placeholder-retention = 1.0, structure preserved, flagged segments for review |
| Documentation writer | Ship a French README/docs page that still renders | Markdown round-trips byte-stably; code blocks untouched |
| Developer / i18n owner | Wire MT into a CI/CD localization step | FastAPI `/translate-text`, decision trace, fail-soft on errors |
| MT researcher | Measure whether document context actually helps | Sentence-level vs concat-k 2x2 ablation, d-chrF, contrastive sets |

## Success Metrics

- **chrF (headline)** + BLEU on the MT core — beats zero-shot / dictionary / identity baselines.
- **Structure-Preservation Score (SPS)** = struct-match x markup-validity x placeholder-retention.
- **Placeholder-retention** — every `[[PHn]]` restored exactly once (a hard gate).
- **Document-level chrF (d-chrF)** + contrastive discourse accuracy for the context story.
- **Human post-edit rate** (needs-review rate) — the operational signal: rising needs-review or falling SPS indicates drift.

Run `doctrans evaluate` for live numbers. The system is a localization **aid**: structure-breaking or low-confidence output is flagged for human review and never presented as final, and fail-soft keeps the source rather than emitting a broken document.
