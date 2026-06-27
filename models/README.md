# models/

Trained artifacts live here (and under `DOCTRANS_MODEL_DIR`) and are **git-ignored**.

- `mt/<version>/` — the fine-tuned MT core (HF format) + `model_meta.json` (records
  `context: true` when trained with concat-k); `mt/latest` points at the newest version.
- `mt/dictionary_mt.json` — the persisted dictionary MT baseline / offline fallback.

The structure layer (parse / mask / reassemble / validate) is **deterministic code**, not
a trained model. Rebuild with:
```bash
doctrans train-baseline      # dictionary baseline (no GPU)
doctrans train-mt            # fine-tune the MT core with document context (GPU)
```
