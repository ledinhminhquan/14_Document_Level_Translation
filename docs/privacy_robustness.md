# Privacy & Robustness

This document describes how **doctrans** (P14, Structured Document-Level Machine
Translation) handles potentially sensitive documents and how it stays robust
against structural corruption, placeholder loss, and component failure. The
guiding principle is that doctrans is a localization **aid**: any output that is
structure-breaking or low-confidence is **flagged for human review, never
presented as final**, and on hard failure the system **fails soft and keeps the
source** rather than emitting a broken document.

## Data privacy

Documents may carry confidential prose, internal URLs, customer placeholders, or
embedded metadata. doctrans is designed to minimize exposure:

- **Local-first, offline-capable.** The package runs with only
  `numpy/scikit-learn/pyyaml/pydantic`; heavy deps (torch, transformers) are
  lazy. With the offline seed (small Markdown docs + gold French + an en->fr
  dictionary) the full pipeline runs with **no network and no torch**, so
  documents never leave the machine.
- **Local path I/O.** Inputs are read from and written to local paths
  controlled by `DOCTRANS_DATA_DIR` / `DOCTRANS_ARTIFACTS_DIR` /
  `DOCTRANS_RUN_DIR`; `HF_HOME` scopes any model cache.
- **Metadata-only logging.** The FastAPI service logs request metadata
  (format, segment count, decision trace, SPS, latency) — **not** document
  content. Decision traces reference segments by index/sentinel, not by raw
  payload.
- **Transient working files.** Intermediate artifacts are scoped to the run
  directory and are not persistent stores of customer text.
- **Optional LLM brain is OFF by default.** The advisory `anthropic` brain, when
  explicitly enabled, writes only a terminology-consistency note; it never sees
  or rewrites structure or placeholders. Leaving it off keeps all text on-device.

## Robustness

### Structure-corruption prevention
The structure layer is **deterministic code**, separate from the model. Three
gates protect it:

- **Round-trip identity (D1).** A document is an interleaved list of literal
  skeleton strings + `Segment` objects; re-joining must reproduce the input
  **byte-for-byte** before translation proceeds. Mismatch -> degrade to the plain
  parser; hard parse failure -> flag/abort.
- **Byte-stable skeleton.** The model never sees structure (headings, list
  markers, table pipes, fenced code, JSON keys/numbers); these ride as literals,
  so they cannot be paraphrased away. JSON re-emits via `json.dumps`.
- **Structure-validate (D5).** Output must match the structure signature and
  re-parse cleanly. Repair budget M=1; otherwise flag and keep source.

### Placeholder / markup loss
Non-translatable spans (inline/fenced code, URLs, emails, link targets, HTML
tags, `{name}`/`{{var}}`/`%s`/`${VAR}`/`:named`) are masked as reversible,
cp1252-safe sentinels `[[PHn]]`.

- **Hard retention gate (D4).** Every `[[PHn]]` must return **exactly once** —
  this is a HARD pass/fail, not advisory.
- **Re-translate budget N=2** on length-ratio or emptiness violations (D3,
  accept ratio in `[0.4, 3.0]`).
- **Soft back-translation check.** Round-trip chrF `>= 0.30` else flag.
- **Fail-soft.** Unrecoverable retention/structure failure keeps the source
  segment rather than shipping corrupted markup.

### Graceful degradation & abstention
- **Parser ladder:** native -> regex -> plain, all dependency-light.
- **MT fallback ladder:** trained m2m100 -> zero-shot MT -> dictionary lookup ->
  identity copy-source floor, so a translation always returns something
  inspectable.
- **Abstention is a first-class outcome.** Any gate failure raises the
  needs-review flag instead of forcing output.

## License hygiene
Training corpora carry honest license flags (opus-100 / news_commentary
UNKNOWN; NLLB / IWSLT CC-BY-NC). The **clean path** is `facebook/m2m100_418M`
(MIT) plus the offline seed; structure is synthetic by construction (no
structured parallel corpus exists), so no proprietary documents are required to
reproduce the system.

Run `doctrans evaluate` for live SPS, placeholder-retention, and d-chrF numbers
on the current build.
