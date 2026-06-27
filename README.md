# doctrans — Structured Document-Level Translation

> **P14** of the NLP-in-Industry final assignment · Le Dinh Minh Quan (student 23127460)

Translate a **structured document** (Markdown / HTML / JSON / plain text) into another
language while **preserving its structure byte-stably** (headings, lists, tables, code
blocks, links, inline markup, placeholders) and using **document-level context** for
discourse-consistent prose. The pipeline is the localization spine —
`parse → mask non-translatable spans → translate-with-context → reassemble → validate` —
with a **trainable machine-translation core** (the NLP heart) in the translate step and a
deterministic **agent** (decisions D1–D5) that hard-gates placeholder-retention and
structure-preservation. **The model never sees structure and the structure layer never sees
the model.** Default direction **English → French** (configurable). Reuses the P13 `s2st` MT
plumbing.

Runs fully offline (line/regex parsers + dictionary MT + pure-python metrics) and upgrades
to a fine-tuned `m2m100` + the optional markdown-it/bs4/python-docx parsers when present.

## Assignment requirements → how delivered

| Requirement | Where |
|---|---|
| Problem definition & business value | [docs/problem_definition.md](docs/problem_definition.md) |
| Data description + data card | [docs/data_description.md](docs/data_description.md), [docs/data_card.md](docs/data_card.md), `src/doctrans/data/` |
| Model selection + **baseline** | [docs/model_selection.md](docs/model_selection.md); zero-shot / dictionary / identity baselines vs the fine-tuned MT core |
| Training + evaluation w/ metrics | `src/doctrans/training/`, [docs/evaluation.md](docs/evaluation.md) (chrF/BLEU + SPS + placeholder-retention) |
| **Agentic AI component** | `src/doctrans/agent/`, [docs/agent_architecture.md](docs/agent_architecture.md) (D1–D5 FSM) |
| Deployment / serving | `src/doctrans/api/`, `app/`, `deploy/`, [docs/deployment.md](docs/deployment.md) |
| Continual learning + monitoring | `src/doctrans/monitoring/` + `automation/`, [docs/continual_learning_monitoring.md](docs/continual_learning_monitoring.md) |
| Privacy + robustness | [docs/privacy_robustness.md](docs/privacy_robustness.md) |
| Project plan + teamwork | [docs/project_plan.md](docs/project_plan.md) |
| Ethics & responsible AI | [docs/ethics_statement.md](docs/ethics_statement.md) |
| Report.pdf + slides.pptx | `doctrans autopilot` → `artifacts/submission/…` (auto-generated) |

## The localization spine (and what's trainable)
```
document in (md/html/json/txt)
   -> parse (D1) -> skeleton (literal) + translatable Segments + MASK code/URLs/placeholders (D2)
   -> translate-with-context (the TRAINABLE MT core, concat-k) (D3)
   -> restore masks -> verify placeholder-retention + round-trip (D4)
   -> reassemble (mutate-in-place) -> validate structure (D5)
document out (SAME format, structure intact)
```
The **only fine-tuned stage is the MT model**; the structure layer + agent are deterministic.

## Repo layout
```
14_Document_Level_Translation/
├── src/doctrans/
│   ├── config.py  cli.py  logging_utils.py
│   ├── data/        # dataset (opus-100/news_commentary) + samples (md docs + en->fr dict) + download
│   ├── models/      # model registry
│   ├── docparse/    # THE STRUCTURE LAYER: segment + mask + {markdown,html,json,plain}_doc + router + structure_score
│   ├── mt/          # translator (m2m100 + dictionary, reused from P13) + context (concat-k)
│   ├── training/    # train_mt (concat-k) + train_baseline + evaluate + tune + metrics (chrF/BLEU)
│   ├── agent/       # state + policy (D1-D5) + tools + llm_orchestrator + doctrans_agent (FSM)
│   ├── api/         # schemas + dependencies + main (FastAPI) + ui (Gradio) + app_combined
│   ├── analysis/    # error_analysis + latency
│   ├── autoreport/  # artifact_loader + charts + report_pdf + slides_pptx
│   ├── monitoring/  # drift_report (operational monitor-log)
│   ├── automation/  # autopilot (one button)
│   └── grading/     # checklist (rubric self-check)
├── configs/  data/  models/  tests/  docs/ (15 md)  notebooks/  app/  deploy/  scripts/  sample_data/
├── Dockerfile  docker-compose.yml  Makefile  pyproject.toml  requirements*.txt  .github/workflows/ci.yml
```

## Data & models

| id | role | license / flag |
|---|---|---|
| `Helsinki-NLP/opus-100` (`en-fr`) | MT fine-tune corpus | unknown |
| `Helsinki-NLP/news_commentary` (`en-fr`) | document-context corpus (real adjacency) | unknown |
| `facebook/m2m100_418M` | MT core (default, 1024 positions) | MIT |
| `facebook/nllb-200-distilled-600M` | MT core (stronger) | CC-BY-NC (flag) |

No structured parallel-document corpus exists, so structure is **synthetic** (real sentence
pairs wrapped in Markdown/HTML) + an offline seed. See [docs/data_card.md](docs/data_card.md).

## Quickstart (offline, no GPU, no downloads)
```bash
pip install -e .                 # core only (numpy/sklearn/pyyaml/pydantic) — no torch
export PYTHONPATH=src

doctrans train-baseline          # dictionary MT baseline (instant, CPU)
doctrans evaluate --fast         # chrF/BLEU vs baselines + document chrF/SPS
doctrans demo-agent --fast       # run the document agent on the seed docs (D1-D5)
doctrans translate --file sample_data/sample.md --out sample.fr.md   # structure preserved
```
Add the full stack: `pip install -e ".[all]"`.

## The agent (decisions D1–D5)
`parse (D1 format+round-trip) → segment+mask (D2) → translate-with-context (D3) →
verify (D4: placeholder-retention HARD + round-trip chrF) → reassemble+structure-validate (D5)`
— uniform tool contract, `ToolTrace` audit, optional LLM terminology brain (OFF by default).
Fail-soft: structure-breaking / low-confidence output is **flagged for human review** and the
source is kept for any unrecoverable segment. See [docs/agent_architecture.md](docs/agent_architecture.md).

## Train on Colab (H100, auto-adapts A100/L4/T4)
Open `notebooks/DocTrans_Colab_Training_H100_AUTOPILOT.ipynb`, set the controls, **Run all**.
Resume-safe, GPU-auto-profiled, Colab-safe install. Walkthrough:
[notebooks/COLAB_GUIDE.md](notebooks/COLAB_GUIDE.md).

## One-button autopilot
```bash
doctrans autopilot     # data → baseline → train-mt → evaluate → tune → analysis → report.pdf + slides.pptx + grade + bundle
```

## Serve
```bash
doctrans serve --ui    # FastAPI at :8000 (+ Gradio demo at /ui)
# POST /translate-text       {text, fmt?}     -> translated document (same format) + structure report
# POST /translate-document   (file upload)    -> translated file content
```

## Tests
```bash
pip install -e ".[dev]" && pytest -q     # CPU-only, offline, no downloads
```

## Evaluation
**chrF (headline)** + **BLEU** on prose; document-level **chrF**; **Structure-Preservation
Score (SPS)** = structure-match × markup-validity × placeholder-retention. A translation that
wins chrF but breaks a table is a failure, so they are reported jointly. The fine-tuned MT
core must beat the zero-shot / dictionary / identity baselines. See [docs/evaluation.md](docs/evaluation.md).

## License
MIT (code). Datasets/models keep their own licenses — see the flags above. The redistributable
stack is m2m100 (MIT) + the seed; the training corpora are license-unknown, NLLB is CC-BY-NC.
