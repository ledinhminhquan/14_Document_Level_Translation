# Ethics & Responsible AI — `doctrans` (P14)

**Structured Document-Level Machine Translation** — Le Dinh Minh Quan (23127460).

## Stance: a localization AID, not an autonomous translator

`doctrans` is a tool that *assists* a human localizer. It is **not** a system that ships final translations unattended. The deterministic agent FSM enforces this: any output that breaks structure or falls below a confidence gate is **flagged for human review** and never presented as final. When the agent cannot produce a verified document, it **fails soft** — it keeps the *source* document intact rather than emit a corrupted or untrustworthy translation. Silence-with-source beats a confident wrong answer.

## Primary risks and mitigations

| Risk | Where it bites | Mitigation in `doctrans` |
|---|---|---|
| **Mistranslation harm** | Legal, medical, financial docs where a wrong word changes meaning | Output is advisory; D4 back-translation chrF (soft >=0.30) and length-ratio bounds `[0.4, 3.0]` flag suspect segments; needs-review rate is monitored. Domain-critical content is explicitly out of scope for unreviewed use. |
| **Structure / placeholder corruption** | Broken Markdown/HTML/JSON, dropped `{name}`/`%s`/`${VAR}`, mangled links or code | **Hard gates**: D1 round-trip identity check, D4 placeholder-retention (every `[[PHn]]` restored exactly once), D5 structure-signature match + re-parse. Failures trigger repair (budget M=1) then fail-soft to source. Model never sees structure; structure layer never sees the model. |
| **Bias across languages/domains** | Uneven quality by domain, register, named entities; gender/discourse defaults | Document-level concat-k context targets discourse consistency (pronouns, terminology) over isolated guesses. Honest caveat published: concat-k can underperform when doc-parallel data is scarce. Reported per-domain, not as a single number. |
| **License misuse** | Training/eval data and model weights with non-commercial or unknown terms | License hygiene tracked per asset: opus-100 / news_commentary marked **UNKNOWN**; NLLB / IWSLT marked **CC-BY-NC**. The clean, distributable path is **m2m100-418M (MIT) + the offline seed corpus**. Non-commercial assets are flagged, never silently shipped. |
| **Over-trust / automation bias** | User treats flagged output as final | UI/report always surfaces the **decision trace** and structure report; flagged segments are visibly marked; no "looks done" without a passed verify gate. |
| **Privacy** | Uploaded documents may contain PII | Metadata-only request logging — document bodies are not retained. Package runs fully offline (numpy/scikit-learn/pyyaml/pydantic only; heavy deps lazy). |

## Human-in-the-loop

The agent's five decisions (D1 parse, D2 segment+mask, D3 translate, D4 verify, D5 reassemble) each emit a `ToolTrace`. A human reviewer is the **final authority** on any flagged document. The optional LLM brain (`anthropic`, **OFF by default**) is strictly advisory: it may write a terminology-consistency note and **never** rewrites a translation, alters structure, or touches placeholders.

## Accountability & transparency

- Numbers in docs are illustrative; reproduce live figures by running **`doctrans evaluate`** (headline chrF, d-chrF, SPS = struct-match x markup-validity x placeholder-retention).
- The structure layer is deterministic, versioned code — auditable and not subject to model drift.
- Drift is monitored: rising needs-review rate or falling SPS triggers re-review and possible re-fine-tuning (behind a chrF non-regression gate).

**Bottom line:** preserve the document, surface uncertainty, keep a human in charge, and respect data/model licenses.
