# Sample data

Tiny, committed samples for trying the system without any download.

- `sample.md` — a structured Markdown document (headings, list, code block, link,
  inline code placeholder). Translate it, structure preserved:
  ```bash
  doctrans translate --file sample_data/sample.md --out sample.fr.md
  ```
- `sample_translate_request.json` — an example body for `POST /translate-text`.

The full offline corpus (Markdown docs + gold + an en→fr dictionary) lives in
`src/doctrans/data/samples.py`.
