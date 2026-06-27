# Colab Guide — training doctrans on an H100 (auto-adapts A100/L4/T4)

Run `DocTrans_Colab_Training_H100_AUTOPILOT.ipynb`: upload the repo, set a few controls, **Run all**.

## 0. What gets trained
- **Only the MT core** is fine-tuned — HF `Seq2SeqTrainer` with **document-level concat-k
  context** (k previous sentences + a `<BRK>` separator), selected on **chrF**.
- **The structure layer** (parse / mask / reassemble / validate) is **deterministic code**,
  not trained. The notebook evaluates the fine-tuned MT against the zero-shot / dictionary /
  identity baselines and reports document chrF + the Structure-Preservation Score (SPS).

## 1. Put the repo where Colab can see it (pick ONE)
- **GitHub (recommended):** push this folder to `https://github.com/<you>/doctrans`, set
  `GIT_REPO_URL` in cell 0.
- **Drive:** upload `14_Document_Level_Translation/` to `MyDrive/doctrans/doctrans` (repo root
  = `.../doctrans/doctrans`); leave `GIT_REPO_URL` as the placeholder.

```
MyDrive/doctrans/
├── doctrans/      <- the repo, if using Drive
└── artifacts/     <- created automatically; the MT model + reports persist here
```

## 2. Runtime
`Runtime -> Change runtime type -> GPU`. H100 ideal but optional — cell 6 auto-profiles
batch/precision for **H100/A100/L4/T4** (T4 has no bf16 -> fp16).

## 3. Controls (cell 0)
- `MT_BASE` — the trainable MT core (`facebook/m2m100_418M` MIT default, **1024 positions**
  for context headroom; NLLB-600M stronger but **CC-BY-NC**; opus-mt cheapest, 512 positions).
- `SRC_LANG`/`TGT_LANG`, `MT_CONFIG` (e.g. `en-fr`) — the direction.
- `CONTEXT_K` — document context window (k previous sentences; default 2).
- `MAX_TRAIN_SAMPLES`, `EPOCHS` — training budget.

## 4. Run all
The **autopilot** (cell 9) does everything: baseline -> fine-tune MT (concat-k) ->
evaluate -> analysis -> **report.pdf + slides.pptx + grading + submission_bundle.zip**.
Resume-safe: re-run cell 9 after a disconnect.

## 5. Read the results (cell 11)
Look for the fine-tuned MT core's **chrF** beating the zero-shot / dictionary / identity
baselines, plus document **chrF** and **SPS** (structure-preservation). The model metadata
records `context: true` so inference uses document context automatically.

## 6. Test the trained model (cell 12)
Cell 12 translates `sample_data/sample.md` (a structured Markdown doc with a heading, list,
code block, link, and inline-code placeholder) and prints the translated document — the
structure (and the code/URL) should be **preserved**.

## 7. Deliverables (cell 13)
`report.pdf`, `slides.pptx`, `submission_bundle.zip` under
`artifacts/submission/submission-<stamp>/` (on Drive).

## Troubleshooting
- **"Set GIT_REPO_URL..."** — neither a repo URL nor a Drive copy was found; do step 1.
- **bf16 error on T4** — Turing has no bf16; cell 6 falls back to fp16.
- **OOM** — lower `MAX_TRAIN_SAMPLES`, pick `m2m100_418M` / `opus-mt-en-fr`, or reduce batch.
- **Context window too large** — keep `CONTEXT_K` small for opus-mt (512 positions); m2m100/
  NLLB have 1024 positions.
- **License** — the redistributable stack is m2m100 (MIT) + the seed; opus-100 /
  news_commentary are license-unknown (training only), NLLB is CC-BY-NC.
