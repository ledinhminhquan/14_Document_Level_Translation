# Deployment

`doctrans` ships as a FastAPI service plus a Gradio demo, packaged for Docker and Hugging Face Spaces. The structure layer is deterministic; the only model artifact is the fine-tuned MT core, pinned by `model_version`. Run `doctrans evaluate` for live chrF / SPS numbers — none are hard-coded here.

## FastAPI endpoints

The API exposes the full localization spine (parse -> mask -> translate-with-context -> restore -> reassemble -> validate) behind two translation routes plus health and version probes.

| Method | Path | Request | Response |
| --- | --- | --- | --- |
| `POST` | `/translate-text` | `{ "text": str, "fmt": "auto\|markdown\|html\|json\|plain" }` | `{ "translated": str, "fmt": str, "structure_report": {...}, "decision_trace": [...] }` |
| `POST` | `/translate-document` | `multipart/form-data` file upload (+ optional `fmt`) | same shape as `/translate-text`, byte-stable in the uploaded format |
| `GET` | `/healthz` | — | `{ "status": "ok" }` |
| `GET` | `/version` | — | `{ "package": "doctrans", "model_version": str, "context": bool }` |

- **`/translate-text`** — JSON in, JSON out. `fmt` defaults to `auto` (D1 format detection). `structure_report` carries the Structure-Preservation Score components (struct-match, markup-validity, placeholder-retention); `decision_trace` is the D1–D5 `ToolTrace` audit, including any flagged / fail-soft segments.
- **`/translate-document`** — registered **only when `python-multipart` is present** (multipart-gated). Absent the dependency, the route is silently omitted and `/translate-text` remains available, so the core service never fails to boot over an optional dep.
- **`/healthz`** — liveness probe for Docker / Spaces orchestration; no model load.
- **`/version`** — returns the pinned `model_version` and whether the loaded model is context-fine-tuned (`context:true` enables concat-k at inference).

## Gradio demo

A single-page Gradio app wraps `/translate-text`: paste a Markdown/HTML/JSON/plain document, pick a format (or `auto`), see the translated document, the structure report, and the decision trace side by side. Intended for inspection, not throughput.

## Docker + Hugging Face Space

A single Dockerfile builds the offline-capable image (numpy / scikit-learn / pyyaml / pydantic only); heavy deps (`torch`, `transformers`, `gradio`) are layered when present. The same image runs locally and as a Hugging Face Space. Paths are env-driven: `DOCTRANS_ARTIFACTS_DIR`, `DOCTRANS_MODEL_DIR`, `DOCTRANS_RUN_DIR`, `HF_HOME`.

## Lazy deps + offline degradation

Every heavy import is lazy, so the package boots and serves with the minimal stack. When `torch`/`transformers` are unavailable, the service degrades along the documented ladders: parsers fall back native -> **regex -> line/plain**, and the MT core falls back to the **offline dictionary word-lookup** translator over the Markdown seed. Structure preservation is unaffected (deterministic code), so round-trip identity, masking, and SPS still hold offline.

## Logging + model_version pinning

Request logging is **metadata-only**: format, segment count, decision outcomes, latency, and `model_version` — never document text. `model_version` is pinned at startup and surfaced via `/version`; the continual-learning loop only advances it after the chrF non-regression gate passes, keeping deployed behavior reproducible and auditable.
