# Deploying doctrans as a Hugging Face Space (Docker SDK)

The repo ships a `Dockerfile` that runs the FastAPI service + the Gradio demo (`/ui`).

1. Create a new Space → **Docker** SDK (blank template).
2. Push this repo to the Space. The root `Dockerfile` is used.
3. Expose port **8000** (the API; the demo is mounted at `/ui`). To serve only the demo,
   set the Space `CMD`/`app_port` to run `python app/gradio_app.py` on **7860**.
4. (Optional) Space **secret** `DOCTRANS_LLM_API_KEY` enables the LLM terminology brain. It
   is **OFF by default** and the agent runs fully on rules without it.

## Notes
- The image installs core + serving + report + parser extras. The heavy ML libs
  (torch/transformers/sentencepiece) are optional; without them the Space answers in
  **degraded mode** (line/regex parsers + dictionary MT) — useful for a light demo.
- To serve a *trained* MT core, mount/copy the artifacts dir and point
  `DOCTRANS_ARTIFACTS_DIR` at it (or bake it into the image).
- Endpoints: `GET /healthz`, `GET /version`, `POST /translate-text`,
  `POST /translate-document` (multipart). Try `sample_data/sample.md`.
- **License note:** the redistributable stack is m2m100 (MIT) + the seed. The training
  corpora (opus-100 / news_commentary) carry unknown-license flags; NLLB is CC-BY-NC.
- The agent **flags** structure-breaking / low-confidence output for human review — it
  never presents an unreviewed, structure-broken document as final.
