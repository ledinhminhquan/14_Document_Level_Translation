# data/

Placeholder — **no large data is committed**. Datasets are fetched on demand from the
Hugging Face Hub and cached under the artifacts dir (`DOCTRANS_DATA_DIR`, default
`artifacts/data/`).

Verify reachability with streaming probes (no full download):
```bash
doctrans data
```

Datasets (see `docs/data_card.md` for the full card):

| id | role | license / flag |
|---|---|---|
| `Helsinki-NLP/opus-100` (`en-fr`) | MT fine-tune corpus | unknown |
| `Helsinki-NLP/news_commentary` (`en-fr`) | document-context corpus (real adjacency) | unknown |

**No HF corpus of structured (Markdown/HTML/DOCX) parallel documents exists**, so the
structure layer is exercised on synthetic structured docs + the offline seed in
`src/doctrans/data/samples.py` (which lets the whole pipeline run with no network/torch).
