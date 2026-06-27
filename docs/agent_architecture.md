# Agent Architecture

The `doctrans` agent is a **deterministic finite-state machine (FSM)**, not an LLM
controller. Every decision is reproducible from the input: given the same document
and model, the agent always takes the same path and emits the same audit trace. The
agent orchestrates the deterministic localization spine
(`parse -> extract+mask -> translate-with-context -> restore -> reassemble -> validate`);
the only learned component it calls is the MT model, and it calls it through an opaque
interface so the agent never reasons about structure and the model never sees it.

## Uniform tool contract + ToolTrace

Every action the FSM can take is a **tool** behind one signature: it receives a typed
input, returns a typed output plus a status (`ok` / `flag` / `abort`), and appends a
`ToolStep` to a shared `ToolTrace`. Each step records the tool name, inputs digest,
output digest, the gate decision, and any threshold that was checked. The `ToolTrace`
is the audit log returned in every API response — a human reviewer can replay exactly
why a document passed, was flagged, or fell back to source.

## The five gates (D1–D5)

| Gate | Decision | HARD check (exact threshold) | On failure |
|------|----------|------------------------------|------------|
| **D1** | Format detect + parse | **Round-trip identity**: skeleton + segments must re-join byte-for-byte | Degrade native -> regex -> plain; hard parse failure -> flag/abort |
| **D2** | Segment + mask gate | Skip non-translatable segments by **letter-ratio < 0.30**; masks reversible & **balanced** | Skip segment as literal; unbalanced masks -> flag |
| **D3** | Translate (with k context) | Reject empty output; **length-ratio outside [0.4, 3.0]** | Re-translate, budget **N = 2**; then flag |
| **D4** | Verify gate | **Placeholder-retention (HARD)**: every `[[PHn]]` returns exactly once; back-translation **chrF >= 0.30** (soft) | Retention fail -> flag/abort; low chrF -> flag |
| **D5** | Reassemble + structure-validate | **Structure-signature match** + output re-parses cleanly | Repair, budget **M = 1**; else flag, fail-soft keep source |

- **D1 round-trip identity** is the spine's core invariant: a document is an
  interleaved list of literal skeleton strings + `Segment` objects, and re-joining must
  reproduce the source exactly before any translation happens.
- **D2 letter-ratio 0.30** cheaply distinguishes prose from code/IDs/numeric cells so
  the model is only invoked on real text; masking turns inline code, URLs, emails, link
  targets, HTML tags, and placeholders into reversible `[[PHn]]` sentinels.
- **D4 placeholder-retention is the one non-negotiable HARD gate**: a translation that
  drops or duplicates a sentinel can never ship, because restoring masks would corrupt
  the document.

## Retry / repair budgets

- **D3 re-translate N = 2**: bounded retries (e.g. nudged generation) before flagging.
- **D5 repair M = 1**: one deterministic reassembly-repair attempt before fail-soft.

Budgets are finite by design — the FSM never loops, guaranteeing termination.

## Optional LLM terminology brain (OFF by default)

An optional `anthropic` brain can be enabled to write an **advisory
terminology-consistency note only**. It never rewrites a translation, never changes
structure, and never touches placeholders. It is observational, off by default, and its
output appears only in the report alongside the deterministic trace.

## Fail-soft posture

`doctrans` is a localization **aid**, not an autonomous publisher. Structure-breaking or
low-confidence output is **flagged for human review, never presented as final**. When
D5 cannot validate, the agent keeps the **source document** rather than emitting a broken
one — a safe, reviewable default. Run `doctrans evaluate` for live gate pass-rates,
needs-review rate, and SPS on the current model and corpus.
