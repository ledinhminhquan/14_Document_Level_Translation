# Continual Learning & Monitoring

How `doctrans` stays current after deployment, and how it stays honest about
quality drift over time. The guiding split: **the MT core is trainable and
re-fine-tuned on a loop; the structure layer is deterministic code that is
*versioned, never trained*.** Two different artifacts, two different lifecycles.

## The re-fine-tune loop (MT core only)

```
collect domain docs -> build context-aware pairs -> re-fine-tune (resume-safe)
   -> chrF non-regression gate -> promote -> new model_version
```

1. **Collect.** Gather in-domain documents plus human references — ideally the
   segments that the agent flagged for review (D3/D4/D5), since those are exactly
   where the current model is weakest. Re-running the localization spine yields
   aligned `(source segment, gold target)` pairs already shorn of structure.
2. **Build context.** Re-apply concat-k (default k=2, source-side, left-truncated)
   so the new data trains the document-level behavior, not just sentence MT. New
   checkpoints keep `context: true` in metadata so inference uses context.
3. **Train.** HF `Seq2SeqTrainer`, resume-safe (`group_by_length`, label
   smoothing, `metric_for_best_model=chrF`). Re-fine-tunes the existing
   checkpoint; it does not start from scratch.
4. **Gate.** A new checkpoint is promoted **only if** it does not regress chrF on
   the frozen eval set (the non-regression gate below).
5. **Promote.** Passing checkpoints become a new `model_version`; the structure
   layer version is unchanged.

### chrF non-regression gate

| Check | Rule | On fail |
|-------|------|---------|
| Headline chrF | `chrF_new >= chrF_prod - epsilon` | reject, do not promote |
| Document chrF (d-chrF) | no regression on doc-level set | reject |
| SPS | `>=` current (deterministic, should hold) | investigate structure layer |
| Sanity floors | beats identity (~21) and dictionary (~82 offline) | reject |

Run via `doctrans evaluate` against the held-out set; the gate consumes its
report. The structure layer is **not** retrained — if SPS moves, that is a code
bug or a parser change, tracked through the structure-layer version, never the
model.

## The monitor-log report

Production logs metadata only (no document bodies). `doctrans` rolls up three
live signals; consult `doctrans evaluate` / the monitor log for current numbers.

| Signal | What it catches | Drift direction |
|--------|-----------------|-----------------|
| **needs-review rate** | rising D3/D4/D5 flags = model losing the domain | up = bad |
| **mean SPS** | structure breakage (struct-match x markup x placeholder) | down = bad |
| **latency** | context length / retry-budget creep | up = watch |

Rising needs-review **and** falling SPS together is the canonical drift signal
and a trigger to start a new collect step.

## Alarm fatigue

Thresholds must be tuned, not maximal. If every borderline length-ratio trips a
flag, reviewers stop reading flags and real structure breaks slip through. Keep
the HARD gates strict (placeholder-retention, round-trip identity, structure
signature) and the SOFT gates (back-translation chrF >= 0.30, length-ratio band
[0.4, 3.0]) calibrated so the needs-review rate stays actionable. A flag nobody
trusts is worse than no flag.

## Monitoring vs observability

- **Monitoring** answers *"is it healthy?"* — the three aggregate signals above,
  trended against the gate.
- **Observability** answers *"why did *this* document fail?"* — the per-request
  `ToolTrace` audit and decision trace (D1–D5) let you replay any single
  translation, see which gate fired, and reproduce it deterministically.

You need both: monitoring tells you drift is happening; observability tells you
which segment, parser, or sentinel caused it. Because the structure layer is
deterministic, any structure failure is fully reproducible from its trace — and
fixed in code under a new structure-layer version, not by retraining.
