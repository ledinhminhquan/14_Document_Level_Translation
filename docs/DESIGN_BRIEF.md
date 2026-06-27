# P14 — Structured Document-Level Translation (`doctrans`) — DESIGN BRIEF

> Saved to `D:\NLP Industry Projects\14_Document_Level_Translation\docs\DESIGN_BRIEF.md`.
> Production repo + H100 Colab notebook for a **structure-preserving, document-level Machine Translation** system. The *only* trainable NLP core is the **MT model**; the structure layer (parse / mask / reassemble / validate) and the agent FSM are deterministic and reuse the **P13 `s2st` MT plumbing** (translator, training loop, metrics) directly.
>
> **Status of facts.** Every HF id, license, config, column, size, param count, and `max_position_embeddings` below was verified live on the HF Hub by the five upstream research reports (A datasets, B models+context, C structure, D agentic, E deploy+eval), under authenticated HF user `ledinhminhquan`. Verification date: **2026-06-27**. Items the reports could not pin exactly are tagged **[UNVERIFIED]**. License risks are tagged **[FLAG: …]**. Do **not** silently "upgrade" any flagged license.

---

## 1. Problem & Scope

### 1.1 What we build
**Structured Document-Level Translation:** translate a whole **structured document** (Markdown / HTML / DOCX / JSON / plain text) from a source language to a target language while **(a) preserving the document's structure byte-stable** (headings, lists, tables, code blocks, links, tags, run formatting) and **(b) using DOCUMENT-LEVEL context** (surrounding sentences) so the translation is discourse-consistent (pronouns, terminology, register).

```
DOCUMENT in (md/html/docx/json/txt)
   → PARSE per format → extract translatable text nodes + MASK non-translatable spans
   → TRANSLATE-with-context (the trainable MT core, concat-k)
   → RESTORE masks → REASSEMBLE (mutate-in-place + serialize) → VALIDATE structure
DOCUMENT out (SAME format, structure intact)
```

- The **trainable NLP core = the MT model** (HF `Seq2SeqTrainer` fine-tune). This is the only thing trained; everything else (parsers, masking, reassembly, validation, agent) is deterministic code.
- The **structure layer never sees raw markup, and the model never sees structure** (Report D invariant): the parser splits the doc into a *prose stream* (translated) and an opaque *skeleton* (preserved verbatim via numbered placeholders), then re-slots translations into the untouched skeleton.
- **Document-level context** is realized as **concatenation-based context ("concat-k")** with a separator tag — the standard, architecture-free doc-MT method (Report B; Tiedemann & Scherrer 2017). The MT core is trained/used to translate the *current* sentence conditioned on `k` neighbours.

### 1.2 Default direction (configurable)
- **Default: English ↔ French (`en↔fr`), configurable** via config (mirrors P13's `src_lang`/`tgt_lang` and the m2m100 default in `MtConfig`). The default base model and the primary fine-tune corpus both line up cleanly for en↔fr (Reports A, B).
- Direction and language codes are config-driven and **depend on the chosen base model's code scheme** (see §3.1 — m2m100 uses ISO-639-1 `en`/`fr`; NLLB uses `eng_Latn`/`fra_Latn`; mBART-50 uses `en_XX`/`fr_XX`; Marian is fixed-direction bilingual).

### 1.3 Why this shape (structure-aware cascade, not raw doc→doc)
Sending raw Markdown/HTML through an MT model corrupts structure (eaten list markers, dropped headings, translated URLs/code/placeholders). The reference systems (Report D: Okapi, XLIFF, DeepL Document, Google Document Translate, OmegaT, Translate Toolkit) **all** implement the same spine: *filter → text units → segment → translate → merge into a byte-stable skeleton*. We reimplement that spine deterministically and put a **trainable, context-aware MT core** in the translate step, wrapped by a **QA agent (D1–D5)** that gates structure-preservation and placeholder-retention. This is QA-able, debuggable, and offline-degradable.

---

## 2. Datasets

All ids resolved live on the Hub (Report A). **There is NO HF corpus with clean en↔fr parallel data + explicit document-boundary metadata, and NO HF dataset with Markdown/HTML/DOCX-structured parallel documents.** Therefore **structure is 100% SYNTHETIC** (we wrap/segment Markdown/HTML/DOCX/JSON around real sentence pairs) and **document-level context is reconstructed** from corpora whose row order preserves intra-article/intra-talk adjacency. The MT core itself is fine-tuned on real sentence/paragraph parallel data.

### 2.1 Primary fine-tune corpus (the MT core) — OPUS-100 en-fr

| Field | Value |
|---|---|
| **EXACT id** | **`Helsinki-NLP/opus-100`**, config **`en-fr`** — RESOLVES |
| Columns | `translation` {`en`, `fr`} |
| Sizes (en-fr) | train **1.0M** / validation **2.0K** / test **2.0K** |
| **License** | **unknown — [FLAG: UNKNOWN LICENSE]** (treat as research/educational; common in academia, document the risk) |
| Notes | English-centric, broad-domain, clean 1M pairs; some noise (subtitle timestamps). Reused verbatim from P13 (`MtConfig.mt_dataset="Helsinki-NLP/opus-100"`, `mt_config="en-fr"`). |

### 2.2 Document-context corpus (real doc-ordered adjacency) — News-Commentary en-fr

| Field | Value |
|---|---|
| **EXACT id** | **`Helsinki-NLP/news_commentary`**, config **`en-fr`** — RESOLVES |
| Columns | `id` (string), `translation` {`en`, `fr`} |
| Sizes (en-fr) | train **209.5K** (no val/test split — carve your own) |
| **License** | **unknown — [FLAG: UNKNOWN LICENSE]** |
| Why this for context | Verified: the `id` is a **monotonic per-article sentence index** (0,1,2,…), so **consecutive rows are adjacent sentences from the same news article** → reconstruct multi-sentence context windows for **concat-k** training. This is the best *real* doc-ordered en↔fr signal on HF. |

### 2.3 Permissive alternative for context (talk order) — IWSLT2017 en-fr [FLAG]

| Field | Value |
|---|---|
| **EXACT id** | **`IWSLT/iwslt2017`**, configs **`iwslt2017-en-fr`**, **`iwslt2017-fr-en`** — RESOLVES |
| Columns | `translation` {`en`, `fr`} |
| Sizes (en-fr) | train **232,825** / validation **890** / test **8,597** |
| **License** | **`cc-by-nc-nd-4.0` — [FLAG: NON-COMMERCIAL + NO-DERIVATIVES]** |
| Doc boundaries | Source = TED talks (talk = document); sentences in talk order so consecutive rows share a talk, **but no explicit talk/document-id column** is exposed. |

### 2.4 Permissive (CC0) sentence corpora — ParaCrawl (no doc order)

| Field | Value |
|---|---|
| **EXACT ids** | **`ParaCrawl/para_crawl`** config **`enfr`**; also **`Helsinki-NLP/opus_paracrawl`** (en-fr via lang pair) — both RESOLVE |
| Columns | `translation` {`en`, `fr`} |
| Sizes | `para_crawl` train **~31M+** (10M<n<100M); `opus_paracrawl` 10M<n<100M |
| **License** | **`cc0-1.0` (permissive — clean)** |
| Notes | Use **only** if you specifically need a permissive license. Web-mined, **shuffled sentence pairs, NO document order** (cannot supply real context), noisier. |

### 2.5 Supplementary / eval corpora

| EXACT id | Config | Columns | Sizes | License | Use |
|---|---|---|---|---|---|
| `Helsinki-NLP/opus_books` | `en-fr` | `id`, `translation` {en,fr} | train **127.1K** (train only) | **other — [FLAG]** (copyright-free aligned books; verify per-book) | Clean literary domain; supplementary / eval |
| `wmt/wmt14` | `fr-en` | `translation` {fr,en} | train **40.8M** / val **3.0K** / test **3.0K** | **unknown — [FLAG]** (card warns CommonCrawl portion misaligned) | Only for a massive high-resource run (overkill) |
| `Helsinki-NLP/opus_dgt` | en-fr (lang pair) | `translation` | 1M<n<10M | **unknown — [FLAG]** (EU JRC TMs; segment-level, no doc order) | Domain (legal/EU) supplementary |
| `GEM/wiki_lingua` | has `fr`, `en` | article/summary (summarization schema) | ~770k total (18 langs) | **`cc-by-nc-sa-3.0` — [FLAG: NON-COMMERCIAL]** | Article-level but **summarization, not MT** — not a clean parallel source |

### 2.6 Structured/subtitle parallel data — none structure-bearing

| EXACT id | Notes | License |
|---|---|---|
| `Helsinki-NLP/open_subtitles` (id is `open_subtitles`, NOT `OpenSubtitles`) | en-fr large; flat sentence streams, **no Markdown/HTML structure**; card mandates attribution link to opensubtitles.org | **unknown — [FLAG]** |
| `Helsinki-NLP/qed_amara` (`Helsinki-NLP/qed` does NOT exist) | QED corpus, 225 langs; script-loader (viewer 501) | **unknown, research-only — [FLAG: RESEARCH-ONLY]** ("made public for RESEARCH purpose only", QCRI all rights reserved) |

### 2.7 Synthetic data (the structure + context generators — REQUIRED)
Because no HF dataset supplies structure or explicit doc boundaries, two generators (deterministic, seeded) sit on top of the real sentence corpora:
1. **Context-window builder.** Group consecutive same-article rows in `news_commentary` (via the monotonic `id`), or same-talk runs in IWSLT, into `(prev_k … current)` windows → **concat-k** training examples (§3.4). Sentence-shuffled corpora (ParaCrawl/OPUS-100) **cannot** supply real context — for those, context is empty (k=0) or self-only.
2. **Structure wrapper.** Programmatically wrap/segment **Markdown/HTML/DOCX/JSON** around sentence pairs (headings, lists, tables, code blocks, links, placeholders) to (a) train/eval structure preservation and (b) provide a **seed of small gold structured docs** for offline tests (§7).

### 2.8 Corrections to common id mistakes (Report A)
- `Helsinki-NLP/qed` → does NOT exist; correct id is **`Helsinki-NLP/qed_amara`**.
- `OpenSubtitles` → correct id is **`Helsinki-NLP/open_subtitles`**.
- `para_crawl` → **`ParaCrawl/para_crawl`** (also **`Helsinki-NLP/opus_paracrawl`**); `opus_dgt` → **`Helsinki-NLP/opus_dgt`**.

---

## 3. Models

### 3.1 The TRAINABLE MT core (the only fine-tuned stage)
All 7 ids verified live (Report B); config values from each repo's `config.json`. The base model's `max_position_embeddings` caps the **whole concatenated sequence** (context + current), so it is the binding constraint on the context window.

| EXACT HF id | Arch | Params | Lang-code scheme | `max_position_embeddings` | License | Flag |
|---|---|---|---|---|---|---|
| **`facebook/m2m100_418M`** ⟵ **DEFAULT** | m2m_100 | ~418M | `tokenizer.src_lang="en"` + `forced_bos_token_id=get_lang_id("fr")` (ISO-639-1, no in-text prefix) | **1024** (sinusoidal) | **MIT** | **clean ✅** |
| `facebook/mbart-large-50-many-to-many-mmt` | mbart | **611.1M** | lang tokens `en_XX`,`fr_XX`; `src_lang=`+`forced_bos_token_id` | **1024** | unspecified on repo metadata (model card / arXiv:2008.00401 → MIT for mBART-50) | **[FLAG: LICENSE NOT ON REPO METADATA]** — treat as MIT per FAIR but verify |
| `Helsinki-NLP/opus-mt-en-fr` | marian | ~75M | bilingual, fixed direction (no lang tokens) | **512** | **Apache-2.0** | **clean ✅** |
| `Helsinki-NLP/opus-mt-fr-en` | marian | **75.2M** | bilingual, fixed direction | **512** | **Apache-2.0** | **clean ✅** |
| `facebook/nllb-200-distilled-600M` | m2m_100 | ~600M | BCP-47+script `eng_Latn`,`fra_Latn`; `src_lang=`+`forced_bos_token_id` | **1024** | **CC-BY-NC-4.0** | **[FLAG: NON-COMMERCIAL]** — eval/research baseline only |
| `google/mt5-base` | mt5 | ~580M | none (raw multilingual T5; NOT translation-tuned → needs supervised FT) | T5 relative pos (no absolute cap; `relative_attention_num_buckets=32`) | **Apache-2.0** | **clean ✅** but not MT-pretrained |
| `google/madlad400-3b-mt` | t5 | **2940.4M (~3B)** | text prefix `<2fr>` prepended to source | `n_positions=512` (T5 relative; buckets=32, max_dist=128) | **Apache-2.0** | **clean ✅** but 3B → LoRA/QLoRA; better as zero-shot reference |

**Default decision: `facebook/m2m100_418M` (MIT, 1024 positions).** Same default as P13 `MtConfig.base_model`. MIT, 100 langs, the 1024-token window gives real headroom for concatenated context, simple `src_lang`/`forced_bos_token_id` API, and it reuses P13's `Seq2SeqTrainer` plumbing directly.

**Fallback ladder (config `base_model`):**
1. `facebook/mbart-large-50-many-to-many-mmt` — 1024 positions, stronger than m2m100_418M; accept the missing-license-tag flag.
2. `Helsinki-NLP/opus-mt-en-fr` / `opus-mt-fr-en` — Apache-2.0, tiny (75M), fastest to fine-tune; **only 512 positions → smallest context budget (k≈1–2 only)**. Good "lite" bilingual baseline.
3. `google/mt5-base` — Apache-2.0, T5 relative positions (graceful long context) but needs more supervised data/epochs (not MT-pretrained).
4. `google/madlad400-3b-mt` — strongest quality, Apache-2.0, but 3B (LoRA/QLoRA) and `n_positions=512`; use as zero-shot reference.

**Avoid as default:** `facebook/nllb-200-distilled-600M` — strong but **CC-BY-NC-4.0**. (Report E recommends NLLB as default; we **override** to keep the commercial-clean MIT default consistent with P13. NLLB stays as a flagged research/eval baseline; MADLAD-400-3B is the commercial-clean "bigger model" option.) **[Resolved discrepancy between Report B and Report E — m2m100_418M wins as default for license cleanliness + P13 reuse.]**

### 3.2 Baselines (the floor to beat)
- **Sentence-level no-context MT (k=0):** the base model fine-tuned without any context — the primary contrast for the "context story".
- **Zero-shot base MT:** the base checkpoint with no fine-tuning (sentence-level and concat-k variants → 2×2 ablation, §5).
- **DictionaryTranslator / identity:** reuse P13 `mt/translator.py` `DictionaryTranslator` (seed phrase table, accent-stripped lookup) and an identity passthrough as the absolute floor and the offline/no-torch path (§7).

### 3.3 Document-level context strategy (concat-k with a separator tag)
There are **no plug-and-play doc-level MT checkpoints on HF** (Report B hub searches returned nothing usable). The proven approach is **concatenation-based context** trainable on any base above with the existing `Seq2SeqTrainer` (Tiedemann & Scherrer 2017; refined by later work).

- **Recipe "concat-k":** prepend the `k` previous sentences to the current sentence, joined by a special separator token.
  - **`k-to-1` (DEFAULT):** context on the **source only**; train the model to output **only the current target sentence**. Easier alignment, no output-splitting at inference, avoids the length-bias/under-generation failure mode.
  - **`2-to-2` (optional):** context on both sides (`prev_2 <BRK> prev_1 <BRK> current` → same on target); mask context-target tokens in the loss (set labels before the last `<BRK>` to `-100`) so the model is optimized to translate the **current** sentence.
- **Separator token `<BRK>`:** add one special token (`tokenizer.add_special_tokens` + `model.resize_token_embeddings`). m2m100/mbart already have `</s>` that can be repurposed, but an explicit `<BRK>` is cleaner and matches the literature.
- **How much context (k):** evidence converges on **k = 1–2 previous sentences** (Fernandes et al. 2021, ACL: models effectively use only 1–2 sentences, diminishing returns beyond, target context used slightly more than source). **Practical default: k=2 source context (2-to-1)**; expose k=1 and k=3 as configurable; never exceed the position limit. (Report D's agent uses **k=3 preceding + 1 following** at inference for disambiguation — both are within evidence; keep `context_k` a single config knob, default training k=2, inference k configurable.)
- **Source vs target context:** target-side helps slightly more but **propagates errors** at inference. Safe default = source-side (`k-to-1`); use target context only with gold/clean previous translations during eval.

### 3.4 Position-limit constraint & truncation
The base model's `max_position_embeddings` caps the **whole concatenated sequence**:
- **m2m100 / mbart-50 / nllb (1024):** budget ≈ 1024 tokens. With ~25–40 tokens/sentence, k=2 + current ≈ 75–120 tokens → comfortable; even k=3–5 is safe. **1024 is the recommended target (most headroom).**
- **opus-mt Marian (512):** half the budget → keep **k=1–2 only**; truncate hard at 512.
- **mt5 / madlad (T5 relative):** no hard absolute cap, but train within ~512–1024 to match pretraining; quality + memory degrade beyond.

**Truncation strategy — left-truncate context, NEVER the current sentence:**
1. Tokenize the current sentence first; reserve its tokens + boundary tokens.
2. Fill the remaining budget with previous sentences **newest-first**, dropping the oldest whole sentence when the budget is hit (drop whole sentences, never partial — preserves `<BRK>` structure).
3. `max_source_length` = model limit (1024 m2m100 / 512 Marian), `truncation=True`, `padding="longest"`.

### 3.5 GPU profile table (H100 / A100 / L4 / T4) — auto-adapt
m2m100_418M (or NLLB-600M-scale) full fine-tune, seq_len up to 1024 (concat-k). Precision facts (Report E): H100 adds FP8 (Transformer Engine); A100 = BF16/FP16/TF32 (no FP8); L4 (Ada) FP8 + limited VRAM; **T4 (Turing) = FP16 only — no bf16, no tf32**. Auto-adapt: `bf16 = torch.cuda.is_bf16_supported()`; if False (T4) → `fp16=True, tf32=False`. Enable `gradient_checkpointing=True` when VRAM < 40 GB; LoRA/QLoRA (`peft`, `bitsandbytes`) for 1.3B/3.3B bases on small cards.

| GPU | VRAM | Precision | per-device batch | grad-accum | eff. batch | Notes |
|-----|------|-----------|------------------|------------|------------|-------|
| **H100 80GB** | 80 GB | bf16 + tf32 (FP8 opt.) | 32–48 | 1 | 32–48 | `predict_with_generate`, beams=4; can host 1.3B/3.3B |
| **A100 80GB** | 80 GB | bf16 + tf32 | 24–32 | 1 | 24–32 | no FP8; same recipe |
| **A100 40GB** | 40 GB | bf16 + tf32 | 12–16 | 2 | 24–32 | |
| **L4** | 24 GB | bf16 (FP8 infer) | 4–8 | 4 | 16–32 | grad-checkpointing on; eval beams=1–2 |
| **T4** | 16 GB | **fp16 only** | 2–4 | 8 | 16–32 | **`bf16=False, tf32=False, fp16=True`**; grad-checkpointing; LoRA/QLoRA |

---

## 4. The structure-preserving pipeline + the agent FSM (D1–D5)

### 4.1 Core invariant & data structure
**The model never sees structure, and structure never sees the model** (Report D). The skeleton is byte-stable; only leaf text is mutated, and only after its non-translatable spans are masked. The clean abstraction (Report C) is a flat list of **`Segment`** objects, each carrying: `id`, `text` (translatable, masks already substituted), `mask_map` (sentinel → original span), `translatable: bool`, and a **back-reference** (the parse-tree node itself / DOCX run / JSON parent+key) used to write the translation back. Translation touches only `segment.text`; reassembly walks the same tree and pulls translations by `id`.

### 4.2 Parser choice per format (degradation ladder)
Order of degradation: **native parser → regex-line parser → pure text** (split on blank lines, translate each block, never touch lines that look like code/URLs). Each parser is wrapped in try/except; a missing lib drops that format to the lighter path with a warning. The `Segment`/`mask_map` interface is identical across paths, so translate + metric code is format-agnostic.

| Format | Primary lib | License | Fallback | Key mechanic |
|---|---|---|---|---|
| **Markdown** | `markdown-it-py` (+ `mdformat`'s `MDRenderer` for roundtrip) | MIT | regex line-classifier → pure text | Parse to `SyntaxTreeNode(md.parse(text))`; collect only `inline` tokens; translate reconstructed inline text (inline markup masked); write back to `token.content`/children; re-render with `MDRenderer`. **Never touch** `fence`, `code_block`, `code_inline`, `html_block`, `html_inline`. Token fields relied on: `type`, `tag`, `content`, `children`, `nesting`, `markup`, `info`, `attrs`. |
| **HTML** | `beautifulsoup4` + stdlib `html.parser` (default) | MIT/BSD | stdlib `html.parser` → pure text | Translate NavigableString / element `.text`/`.tail`; set `.string` and `str(soup)`. `lxml` only as a speed extra. `href`/`src` never translated; `alt`/`title` optional. |
| **DOCX** | `python-docx` | MIT | unzip + stdlib `xml.etree` on `word/document.xml` → pure text | `paragraph.runs` each preserve formatting; set `run.text = translated`; `document.save()`. Group consecutive runs per paragraph into one segment, translate, redistribute (or run-granularity with minor fragmentation). |
| **JSON** | stdlib `json` | PSF | stdlib `json` (no fallback needed) | Recursively walk; translate only `str` **values** (optionally key-filtered), never keys/numbers/bools. `json.dumps(..., ensure_ascii=False, indent=…)`. |
| **plain** | stdlib only | — | (is the floor) | Split on blank lines into paragraph/line nodes; protect code/URL-looking lines. Needs zero parser libs. |

**Walk-and-map rule (universal):** every extracted text node gets a stable `id`; the back-reference is the tree node object itself (Markdown `inline` token / HTML NavigableString / DOCX run / JSON parent+key). **Mutate in place, then serialize** — far more robust than offset-based reinsertion.

### 4.3 Masking non-translatable spans (sentinel scheme — recommended)
Strategy **(B) tag/placeholder masking** (default; no aligner needed; degrades cleanly) over **(A) detag-and-project** (needs a word aligner, fragile — use only if an aligner is available). Replace each protected span with a **sentinel**, record `mask_map[sentinel] = original_span`, translate, restore verbatim.

**Sentinel design (critical for NMT survival):** short, **ASCII, no internal spaces**, numeric index so order can be checked. Recommended **`[[PHn]]`** (ASCII-safe) — Report C also notes `⟦PHn⟧`; **prefer the ASCII `[[PHn]]` form for opaque NMT and Windows cp1252 safety** (§7 gotchas). Avoid leading/trailing punctuation MT strips; avoid `<...>` for XML-aware models. Verify the tokenizer keeps the sentinel as ≤ a few stable pieces.

**Ordered protector list (apply longest/most-specific first):**
1. Fenced/indented **code blocks** + inline code `` `...` `` — mask whole span.
2. **Math** `$...$`, `$$...$$` (if applicable).
3. **HTML/XML tags** / self-contained components — mask the tag, keep inner text translatable.
4. **URLs** (`https?://…`, `www.…`) and **emails**.
5. **Markdown link/image targets** — mask the `(url)`/`(path)` part, keep `[text]`/`![alt]` translatable.
6. **Placeholders:** `{name}`, `{{var}}`, `%s`/`%d`/`%1$s`, `$VAR`, `${VAR}`, `:named`, ICU `{count, plural, …}`.
7. **Numbers / codes** (version strings `v2.3.1`, SKUs, formatted dates/IDs) — optional.
8. **Inline emphasis markers** when flattening inline Markdown: `**bold**` → `[[PHi]]bold[[PHj]]` (mask markers, translate inner words) — keeps emphasis aligned without an aligner.

**Restoration + checks:** after MT, replace every sentinel with its original span; assert (a) every sentinel that went in came back **exactly once**, (b) none altered/split. Failure → fallback (re-translate that segment unmasked, or keep source).

### 4.4 Reassembly + structure validation
**Mutate-in-place then serialize, never string-splice** (per-format detail in §4.2). **Validation gate (mdformat's trick):** re-parse the output and compare its structure-signature to the input's; assert equality before returning. Catches MT that ate a list marker, dropped a heading, or merged table cells.

### 4.5 The deterministic agent FSM (D1–D5)
A **deterministic FSM** (same input + config → same path). Each state operates on the prior artifact, has an explicit gate, and on fail takes a bounded, logged action (re-translate / repair / flag). **No randomness; LLM brain OFF by default** (§4.6). Mirrors the P13/P02 agent pattern (uniform `run()->dict`, never raise past orchestrator, ToolTrace audit).

```
[IN] → D1 → D2 → D3 → D4 ──pass──→ D5 ──pass──→ [OUT]
              ↑           │                │
              └─re-translate (≤N=2)────────┘   (D4 fail loops to D3; D5 repair ≤M=1)
```

| State | Operates on | Explicit decision | Threshold / gate | Fail action |
|---|---|---|---|---|
| **D1 Format detect + parse** | raw bytes | route by format; parse OK? | extension → MIME/magic-bytes → content sniff (priority order); **round-trip identity** `reserialize(parse(x))==x` for text formats (structural-equality for OOXML) | low detection confidence OR round-trip differs → **fall back to `plain`** (lossless) + log downgrade; hard parse failure → **FLAG, abort** (never translate an unparseable doc) |
| **D2 Segment + mask** | AST | segment (SRX/`pysbd`-style, abbreviation/decimal-safe) + protect; pure-non-text skip? | **residual letter ratio < 30%** of non-space chars → `translatable=false`, route around D3; masks must be **reversible & balanced** (every `[[PHi]]` ↔ one literal, indices contiguous `0..n-1`, no collision) | un-maskable (clash/unbalanced) → widen delimiter, retry once; still failing → **FLAG segment, pass through verbatim** (never send mangled masks to MT) |
| **D3 Context + translate** | masked translatable segs | translate with **k context** (training k=2; inference k configurable, ~3 prev +1 next) + running terminology/entity table for consistency | sanity: reject empty output, output-lang == source-lang (lang-ID on result), or **length ratio outside [0.4, 3.0]** | re-translate once with **reduced** context window (large windows cause under-translation); persistent fail → hand to D4 flagged |
| **D4 Verify gate** | translated segs + source | **two independent checks, BOTH must pass** | **(1) Placeholder/tag retention (hard, exact):** multiset of `[[PHi]]` in output == input (same tokens, same count) — non-negotiable. **(2) Round-trip back-translation chrF (soft):** back-translate tgt→src, chrF vs original src: **≥50 pass**; **45–50 borderline** (pass + tag for review — within 2–5pt noise band); **<45 fail**. Guard: if back-trans lang-ID ≠ source lang, discard chrF, treat as fail. | loop to **D3, re-translate ≤ N=2** (1st retry shrinks k→1, 2nd masks more aggressively); after N → **emit best-of-attempts + FLAG** for human review (never silently drop) |
| **D5 Reassemble + structure-validate** | verified segs + skeleton | unmask (`[[PHi]]`→literals, 1:1) → re-join → slot leaves into untouched skeleton → serialize | **all three:** node-tree match (same count/types/nesting/order, only leaf text differs; JSON identical key set/shape); **zero residual `[[PHi]]`**; output **re-parses cleanly** in its format | **repair** (re-slot offending node / per-node replacement) + re-validate **≤ M=1**; still mismatched → **FLAG document, emit source structure with translated leaves where safe + diff report** (never ship malformed output) |

**Standard chrF facts for thresholds (Report D):** chrF range **0–100**; good systems **>60**; SOTA high-resource **55–65**; differences **below 2–5 points are within noise**.

### 4.6 Optional LLM "brain" (Anthropic, OFF by default, rule fallback)
Default path is fully deterministic. When `--llm-brain on` (provider `anthropic`, e.g. a Claude model), it is an **advisory strict-superset** on the deterministic core:
- **MAY (advisory only):** suggest a preferred target term at D3 when a source term was rendered inconsistently (terminology-consistency note); **vote** on whether an ambiguous span is non-translatable (D2 may accept only if it **tightens** — more masking — never loosens); annotate flagged D4/D5 segments with a human-readable reason.
- **MUST NOT:** silently change content (no rewriting translations, no insert/remove text, no altering structure/placeholders); bypass D4/D5 (anything it influences re-enters the same gates); become required (if unavailable/erroring/low-confidence → FSM falls back to rules and proceeds unchanged).

---

## 5. Baselines + Evaluation

### 5.1 Headline metrics
**chrF (text quality) + Structure-Preservation Score (SPS, format fidelity) reported JOINTLY.** A translation that wins chrF but breaks a table/code block is a failure. chrF is the primary text metric (character n-gram, morphology-robust, better human correlation than BLEU on diverse langs); BLEU reported alongside for comparability.

### 5.2 Metric battery
1. **Text quality:** **chrF / chrF++** (primary) + **BLEU** (secondary) on restored prose. Reuse **P13 `training/metrics.py`** (`chrf`, `bleu`, `translation_metrics` — pure-python, no torch) for the offline path; use sacreBLEU (`evaluate.load("sacrebleu")`, `evaluate.load("chrf")`) when available, and report the sacreBLEU signature for reproducibility.
2. **Document-level:** **d-BLEU / d-chrF** — realign sentence outputs into the full document, score the whole doc to capture discourse/consistency (AFRIDOC-MT).
3. **Structure-Preservation Score (SPS, custom):** parse src and output with the *same* parser; `SPS = w1·StructMatch + w2·(TagMatch+MarkupValidity) + w3·PHR + w4·NTI` (default equal weights; optionally weight PHR/NTI higher — a broken placeholder is a "major" MQM error, 10 vs 1). Report each sub-metric:
   - **(a) StructMatch (block):** `1 − (Σ_type |count_S − count_T|) / (Σ_type count_S)` over `{headings(+level), list_items, table_rows×cols, code_blocks, blockquotes, links, images}`; stricter variant = normalized Levenshtein over the type-tagged element sequence (catches reordering).
   - **(b) TagMatch + MarkupValidity (inline):** per-segment exact multiset match of tags/emphasis/link-image syntax, plus a hard `MarkupValidity∈{0,1}` = "output re-parses well-formed." (Mirrors localization "tag mismatch" QA and WMT **Tagged-BLEU**.)
   - **(c) PHR — Placeholder-Retention Rate:** `(# sentinels/placeholders restored exactly once, unaltered) / (# inserted)` — the single highest-signal safety metric; target ≈ **1.0**.
   - **(d) NTI — Non-Translatable Integrity:** `(# protected spans appearing byte-identical in output) / (# protected spans)`.
   - **Hard AST-equality gate** (mdformat-style) as a binary structure guard on top.
4. **Context-vs-no-context (the context story):** BLEU/chrF are largely **insensitive to discourse fixes**, so use **contrastive accuracy on ContraPro-style test sets** (`github.com/ZurichNLP/ContraPro`) — model must rank the contextually-correct pronoun/term above distractors; report by antecedent distance. Also a **term-consistency rate** (same source term → same target term across the doc).

### 5.3 Baselines (2×2 ablation)
| | Zero-shot (no FT) | Fine-tuned |
|---|---|---|
| **Sentence-level (no context, k=0)** | B1 | B3 |
| **Document-context (concat-k)** | B2 | **B4 (system)** |

- **B3 vs B1 / B4 vs B2** → value of fine-tuning.
- **B2 vs B1 / B4 vs B3** → value of context. Expect **modest chrF/BLEU deltas** but **clear gains on ContraPro accuracy + term-consistency** — that is the context story (BLEU "turns a blind eye" to discourse). **Honest caveat:** concat-k can **underperform** sentence-level if doc-parallel data is scarce — report it.
- Plus the floor baselines: `DictionaryTranslator` / identity (§3.2), and a sentence-level no-context MT.

### 5.4 `Seq2SeqTrainer` recipe (reuse P13)
```python
Seq2SeqTrainingArguments(
  bf16=True, tf32=True,                       # auto-adapt: T4 → fp16=True, tf32=False
  predict_with_generate=True,                 # real chrF/BLEU at eval via generate()
  generation_max_length=1024, generation_num_beams=4,   # 512 for Marian
  metric_for_best_model="chrf", greater_is_better=True,
  load_best_model_at_end=True,
  eval_strategy="steps", save_strategy="steps", save_total_limit=3,
  learning_rate=2e-5, warmup_ratio=0.05, num_train_epochs=3,
  group_by_length=True, label_smoothing_factor=0.1)
```
Add `<BRK>`, `resize_token_embeddings`, build concat-k examples in the preprocessing `map`. **Resume-safe** via `get_last_checkpoint(output_dir)` → `trainer.train(resume_from_checkpoint=ckpt)`. `compute_metrics` uses sacreBLEU chrF + BLEU (or P13 pure-python fallbacks). Need **document-aligned bitext** (news_commentary / IWSLT) so previous-sentence context is real — sentence-shuffled corpora won't give a real context signal.

---

## 6. Deployment + Continual Learning

### 6.1 FastAPI surface
```
POST /translate-document   consumes multipart/form-data   [REQUIRES python-multipart — GATE route]
  form: file: UploadFile (.md/.html/.txt/.docx/.json), target_lang, source_lang=None|"auto",
        use_context: bool=True, context_k: int=2, format: str|None (override autodetect)
  returns: translated doc in SAME format (octet-stream) OR json:
    { filename, format, target_lang, translated_document,
      structure_report: { headings:[n_src,n_tgt], lists:.., tables:.., code_blocks:.., links:..,
                          structure_score: 0.0-1.0 },
      placeholder_retention: 1.0,
      decision_trace: [ {seg_id, type:"prose|opaque", src, masked, mt_out, restored,
                         context_used:[ids], n_placeholders} ... ] }
  headers: X-Source-Format, X-Structure-Score, X-Placeholder-Retention
  errors: 415 unsupported format, 422 bad lang code, 413 too large, 503 model loading

POST /translate-text       consumes application/json   [NO multipart needed]
  body:   { text, target_lang, source_lang=None, use_context: bool=False, context_k: int=0 }
  returns:{ translated_text, source_lang_detected, n_segments }

GET  /healthz   -> { status:"ok", model_loaded:bool, device:"cuda|cpu" }
GET  /version   -> { model:"m2m100_418M", checkpoint_sha, revision:git_sha, build_time,
                     chrF_on_eval, supported_formats:[...] }
```
**Multipart gating (load-bearing — P05/P13 lesson):** `UploadFile`/`File()` routes hard-require `python-multipart`; without it FastAPI raises at route registration. Gate the **document route** so `/translate-text`, `/healthz`, `/version` still work if multipart is absent:
```python
try:
    import multipart  # python-multipart
    HAS_MULTIPART = True
except ImportError:
    HAS_MULTIPART = False
if HAS_MULTIPART:
    @app.post("/translate-document")
    async def translate_document(file: UploadFile, target_lang: str = Form(...), ...): ...
else:
    @app.post("/translate-document")
    async def _disabled(): raise HTTPException(503, "install python-multipart to enable file upload")
```
Use `UploadFile` (spools to disk) over `bytes` to avoid loading whole files into memory.

### 6.2 Gradio demo (`app/`)
- `gr.File(file_types=[".md",".html",".txt",".docx",".json"])` upload **or** `gr.Code`/`gr.Textbox` paste of Markdown.
- `gr.Dropdown` target language.
- Outputs: `gr.Markdown(value=translated)` **rendered** doc + `gr.JSON`/`gr.Dataframe` **structure-preservation report** + `gr.Dataframe`/`gr.JSON` **decision trace** (per-segment src→masked→MT→restored, context used). Note: `gr.Markdown` renders CommonMark/GFM, not Mermaid.
- Mount under FastAPI: `gr.mount_gradio_app(app, demo, path="/demo")` (one process serves API + UI).

### 6.3 Dependencies
- Trainable core / metrics: `transformers` (Seq2SeqTrainer, `get_last_checkpoint`), `datasets`, `accelerate`, `sacrebleu`, `evaluate` (chrF+BLEU), `sentencepiece`, `torch`; `peft`+`bitsandbytes` for LoRA/QLoRA.
- Structure layer (all permissive, optional, try/except): `markdown-it-py`+`mdformat` (MIT), `beautifulsoup4` (uses stdlib parser by default), `python-docx` (MIT); `lxml` speed extra. Core pure-text path needs only stdlib (`json`, `re`, `xml.etree`, `zipfile`, `html.parser`).
- Serving: `fastapi`, `uvicorn[standard]`, `gradio`, **`python-multipart`** (hard-req for `/translate-document`; route gated).
- Segmentation: `pysbd` optional (regex SRX-style fallback).

### 6.4 Continual learning loop
1. **Collect** domain docs (parsed into prose + structure, sentence-aligned with concat-k context).
2. **Fine-tune** from the current promoted checkpoint (resume-safe `get_last_checkpoint`).
3. **Promotion gate (on a FROZEN held-out eval set):** require `chrF_new ≥ chrF_prod − ε` (ε≈0.5 chrF) **AND** `placeholder_retention ≥ 0.99` **AND** `structure_score` non-regressing (optionally ContraPro accuracy ≥ prod).
4. **Promote** only if all gates pass → bump `/version` (`checkpoint_sha`, `chrF_on_eval`), hot-swap model; else keep prod, log diff. Mirrors the P12 drift/promotion pattern.

---

## 7. Offline / no-torch fallbacks + GOTCHAS

### 7.1 Offline / no-torch fallbacks (so tests / agent / eval run with no torch / network)
- **Pure-text path:** stdlib Markdown/HTML parsing (`html.parser`, regex line-classifier) or a regex segmenter (split on blank lines) — needs zero parser libs. Same `Segment`/`mask_map` interface as the native path.
- **Dictionary MT:** reuse P13 `mt/translator.py` `DictionaryTranslator` over **seed phrase pairs** (accent-stripped lookup) + identity passthrough — substitutes for the transformer slot so the whole pipeline + agent + eval run with no torch (mirrors P08's TF-IDF/IdentityReranker pattern).
- **Seed of small structured docs** (`data/samples.py`-style): a handful of Markdown docs with **headings / lists / code fences / links + gold en↔fr translations**, plus matching HTML/JSON variants → tests, agent runs, SPS/PHR/chrF eval all execute offline. Metrics use the **pure-python `training/metrics.py`** (chrf/bleu) so no `evaluate`/sacrebleu needed offline.

### 7.2 GOTCHAS
- **Streaming dataset probes:** `download_dataset` MUST use `streaming=True` probes (P09/P10 lesson) — opus-100 (1M), news_commentary (209K), wmt14 (40.8M / 7.8GB), ParaCrawl (~31M) will otherwise download huge files and hang.
- **License flags:** OPUS-100 / news_commentary / wmt14 / open_subtitles / opus_dgt = **unknown**; opus_books = **other**; IWSLT2017 = **CC-BY-NC-ND**; wiki_lingua = **CC-BY-NC-SA**; qed_amara = **research-only**; ParaCrawl = **CC0 (clean)**. Models: m2m100/opus-mt/mt5/madlad = clean; **NLLB = CC-BY-NC**; **mBART-50 = license not on repo metadata**. Never silently upgrade a flagged license.
- **Masking must be reversible:** every `[[PHi]]` ↔ exactly one literal, contiguous indices, no collision with real text; assert balanced restoration; on failure re-translate unmasked or keep source. A broken placeholder is an MQM **major** error.
- **Structure validation is mandatory:** re-parse output and assert AST/structure-signature equality (mdformat trick) before returning; never ship malformed output (D5 repair ≤1, else FLAG + diff).
- **CLI ASCII-only on Windows cp1252:** CLI output must be ASCII-safe (Windows console cp1252). Prefer the **ASCII sentinel `[[PHn]]`** over `⟦PHn⟧` in any text that may print to a Windows console / be written without explicit UTF-8; always `json.dumps(..., ensure_ascii=False)` with UTF-8 encoding for file I/O but keep console logging ASCII. (P06 NFC/encoding lesson.)
- **`/translate-document` route only when `python-multipart` present** (gate it; keep `/translate-text` + `/healthz` working without it).
- **concat-k needs real doc order:** sentence-shuffled corpora (OPUS-100, ParaCrawl) cannot supply real context — use news_commentary/IWSLT for the context signal, else k=0.
- **concat-k length bias:** concat models can skip sentences (under-generation); the `k-to-1` default largely avoids it; for `2-to-2` mask context-target tokens in the loss (`-100`).
- **Default-model discrepancy:** Report E recommended NLLB-600M (CC-BY-NC) as default; we **override to m2m100_418M (MIT)** for license cleanliness + P13 reuse. NLLB stays a flagged research baseline; MADLAD-400-3B (Apache) is the commercial-clean larger option.

---

## 8. Reuse notes (from P13 `s2st` and the standard templates)

Reuse, near-verbatim where possible:
- **MT core / translator:** `src/s2st/mt/translator.py` → `src/doctrans/mt/translator.py`. Brings `DictionaryTranslator` (offline floor), `TransformerTranslator` (family auto-detect `_detect_family`, `_forced_bos`, `_set_src`, `translate`/`translate_batch`, `from_pretrained`), and `load_translator(cfg, prefer=...)`. **Extend** with concat-k context assembly + `<BRK>` handling.
- **Training:** `src/s2st/training/{train_mt.py, train_baseline.py, evaluate.py, metrics.py, tune.py}` → reuse the `Seq2SeqTrainer` loop, baseline trainer, and **pure-python `metrics.py`** (`chrf`, `bleu`, `wer`, `translation_metrics`). Add concat-k preprocessing + the SPS/PHR/NTI structure metrics.
- **Config:** P13 `config.py` `MtConfig` (already defaults to `facebook/m2m100_418M`, `src_lang`/`tgt_lang`, `max_source_length`, `num_beams`, training hyperparams) → `DocTransConfig` adds `formats`, `context_k`, `mask_*`, `structure_weights`, `mt_dataset="Helsinki-NLP/opus-100"`/`mt_config="en-fr"`.
- **Standard templates** (P02 exemplar / P13): `config.py` (dataclasses + YAML loader, env-var paths, unknown-key tolerant), `logging_utils.py` (logs to **stderr**, stdout stays pipeable JSON), `cli.py` (single argparse entrypoint; console_script `doctrans = doctrans.cli:main`), model **registry**, **autoreport** (report.pdf + slides.pptx), **monitoring** (drift/continual-learning), **grading** checklist, **autopilot** (one-button train→eval→benchmark→error-analysis→report+slides+zip).
- **API:** P13 `api/{main.py, app_combined.py, dependencies.py, schemas.py, ui.py}` → adapt endpoint set to §6.1; reuse the `python-multipart` gating pattern and `gr.mount_gradio_app` mounting.
- **Heavy deps lazy-imported** inside functions (torch/transformers/datasets/markdown-it-py/python-docx) → package imports + CPU-only tests run with only core deps; graceful degradation everywhere (P02 rule).
- **H100 notebook** (P13 pattern): Controls cell with `#@param`, GPU auto-profile (§3.5), Drive-persisted artifacts + HF cache, resume-safe `get_last_checkpoint`, Colab-safe install (never reinstall torch; `pip install -e . --no-deps`), built by a Python generator for valid JSON.

### Appendix — exact ids quick reference (commercial-clean default `en↔fr` stack in **bold**)
- **MT core (default): `facebook/m2m100_418M` — MIT, 1024 pos.** Fallbacks: `facebook/mbart-large-50-many-to-many-mmt` (1024; [FLAG: license not on repo]), `Helsinki-NLP/opus-mt-en-fr` / `opus-mt-fr-en` (Apache, 512), `google/mt5-base` (Apache), `google/madlad400-3b-mt` (Apache, 512). Research-only: `facebook/nllb-200-distilled-600M` [FLAG: CC-BY-NC].
- **Fine-tune corpus (default): `Helsinki-NLP/opus-100` `en-fr` — [FLAG: unknown].** Context: **`Helsinki-NLP/news_commentary` `en-fr` — [FLAG: unknown]** (real doc order). Permissive: `ParaCrawl/para_crawl` `enfr` / `Helsinki-NLP/opus_paracrawl` (CC0, no doc order). Context alt: `IWSLT/iwslt2017` `iwslt2017-en-fr` [FLAG: CC-BY-NC-ND]. Supplementary/eval: `Helsinki-NLP/opus_books` [FLAG: other], `wmt/wmt14` [FLAG: unknown], `Helsinki-NLP/opus_dgt` [FLAG: unknown].
- **Structure + document context: SYNTHETIC** (no HF dataset supplies them) — context-window builder over news_commentary/IWSLT + Markdown/HTML/DOCX/JSON structure wrapper + offline seed docs.
- **Eval contrastive: `github.com/ZurichNLP/ContraPro`** (pronoun/term contrastive accuracy).
