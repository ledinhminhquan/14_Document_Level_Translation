"""Generate the submission report.pdf for the doctrans document-translation system.

A 10-15 page report covering every Section-I deliverable: problem & business value, data
& languages (license flags), the structure-preserving pipeline + models, the trainable MT
core + document-level context, the agent (D1-D5), the evaluation (chrF/BLEU + SPS +
placeholder-retention), deployment, continual learning & monitoring, privacy & robustness,
and ethics. Live numbers come from ``run_dir()`` artifacts; missing metrics degrade to
placeholders. reportlab lazy-imported; a Markdown fallback is written if absent.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_now_iso
from . import charts as charts_mod
from .artifact_loader import (base_model, beats_baseline, doc_metric, has_eval, latency,
                              load_artifacts, model_name, model_version, mt_metric, read_doc)

logger = get_logger(__name__)

_SUBTITLE = ("Structured document-level translation: parse a Markdown / HTML / JSON document, "
             "MASK non-translatable spans (code, URLs, placeholders, tags), translate the prose "
             "with a TRAINABLE machine-translation core (fine-tuned m2m100) using document-level "
             "concat-k context, then reassemble byte-stably and validate structure. A deterministic "
             "agent (D1-D5) gates placeholder-retention and structure-preservation. Default en->fr.")

_SECTIONS = [
    ("1. Problem Definition & Business Value", "problem_definition.md"),
    ("2. Data & Languages", "data_description.md"),
    ("3. The Structure-Preserving Pipeline", "architecture.md"),
    ("4. The MT Core & Document-Level Context", "model_selection.md"),
    ("5. Agent Architecture (Decisions D1-D5)", "agent_architecture.md"),
    ("6. Evaluation Protocol & Baselines", "evaluation.md"),
    ("7. Deployment", "deployment.md"),
    ("8. Continual Learning & Monitoring", "continual_learning_monitoring.md"),
    ("9. Data Privacy & Robustness", "privacy_robustness.md"),
    ("10. Ethics & Responsible AI", "ethics_statement.md"),
]


def _builtin_sections(cfg: AppConfig, arts: Dict[str, Any]) -> Dict[str, str]:
    mc = mt_metric(arts, "model", "chrf")
    dc = mt_metric(arts, "dictionary", "chrf")
    dchrf = doc_metric(arts, "doc_chrf")
    dsps = doc_metric(arts, "doc_sps")
    if mc is not None and dc is not None:
        res_line = (f"In the latest eval the MT core reaches chrF **{mc:.1f}** vs the dictionary "
                    f"baseline **{dc:.1f}**"
                    + (f"; on whole documents chrF **{dchrf:.1f}** at structure-preservation "
                       f"**{dsps:.2f}**." if (dchrf is not None and dsps is not None) else "."))
    else:
        res_line = "Run `doctrans evaluate` to populate the live numbers here."
    return {
        "problem_definition.md": f"""
## What it does
Given a **structured document** (Markdown / HTML / JSON / plain text), produce its
translation in another language **with the structure intact** (headings, lists, tables,
code blocks, links, placeholders) and **document-level context** for discourse-consistent
prose. Default direction **{cfg.mt.src_lang} -> {cfg.mt.tgt_lang}** (configurable). The
model never sees structure and the structure layer never sees the model: leaf prose is
translated through opaque sentinels, then re-slotted into a byte-stable skeleton.

## The job-to-be-done
- **Localization / docs teams** — "translate our README / help-centre / UI strings without
  breaking Markdown, code, or placeholders."
- **Content platforms** — "publish multilingual articles preserving layout."
- **Developers** — "translate JSON/HTML resource files; never touch keys, URLs, or `{{vars}}`."
- **Researcher** — "a reproducible, structure-aware, document-level MT baseline."

## Why it is hard (and why this design helps)
Sending raw Markdown/HTML through an MT model corrupts structure (eaten list markers,
translated URLs/code, dropped placeholders). The fix is the localization spine
*parse -> mask -> translate -> reassemble -> validate*, with a **trainable, context-aware
MT core** in the translate step and a **QA agent** that hard-gates placeholder-retention
and structure-preservation. The only trainable component is the MT model (the NLP heart).

## Success metrics
- **Business:** human post-edit rate (how often a human must fix output), structure-break
  rate, terminology consistency.
- **Technical:** **chrF (headline) + BLEU** on prose; **Structure-Preservation Score (SPS)**
  and **placeholder-retention** for format fidelity; document-level chrF.
{res_line}
""",
        "data_description.md": f"""
## Fine-tune corpus (the MT core)
- **`Helsinki-NLP/opus-100`** config **`{cfg.data.mt_config}`**
  (`translation` {{en, fr}}, 1M pairs). **License flag: unknown** (research/educational).

## Document-context corpus (real adjacency for concat-k)
- **`Helsinki-NLP/news_commentary`** config **`{cfg.data.context_config}`** — its monotonic
  per-article `id` means consecutive rows are adjacent sentences from the same article, so the
  previous-sentence context for concat-k training is **real**. **License flag: unknown.**

## Structure data
**No HF corpus of structured (Markdown/HTML/DOCX) parallel documents exists**, so the
structure layer is exercised on **synthetic** structured documents wrapped around real
sentence pairs; an offline seed of small Markdown docs (with headings, lists, code, links,
placeholders) + gold French translations + an en->fr dictionary ships in `data/samples.py`
so the whole pipeline runs with **no network, no torch**.

## Languages & direction
Default **{cfg.mt.src_lang} -> {cfg.mt.tgt_lang}** (m2m100 ISO codes). Configurable to any
direction the base model supports.
""",
        "architecture.md": f"""
## The localization spine
`parse (per format) -> extract translatable text + MASK non-translatable spans ->
translate-with-context -> restore masks -> reassemble (mutate-in-place) -> validate structure`.
The document is represented as an interleaved list of **literal skeleton** strings (never
translated) and **Segment** objects (the translatable prose, with masks). Re-joining the
literals with the (unmasked) translations reproduces the document byte-stably.

## Parsers (degradation ladder: native -> regex -> plain)
- **Markdown** — a dependency-free line parser keeps heading hashes, list markers,
  blockquotes, table pipes, fenced code, and rules as literals; prose becomes Segments.
- **HTML** — a regex tokenizer keeps every `<tag ...>` (and `<script>`/`<style>` bodies)
  verbatim; text nodes become Segments.
- **JSON** — translate only string **values**; keys/numbers/structure are byte-stable
  (re-serialized with `json.dumps`).
- **plain** — paragraph/line segments; the floor everything degrades to.
Each parser does a **round-trip identity check** (re-join == source) and falls back to a
lighter path on mismatch, so structure is never silently corrupted.

## Masking non-translatable spans
Inline code, fenced code, URLs, emails, Markdown link targets, HTML tags, and placeholders
(`{{name}}`, `{{{{var}}}}`, `%s`, `${{VAR}}`, `:named`) are replaced with ASCII sentinels
`[[PHn]]` (Windows-cp1252 safe), recorded `sentinel -> original`, and restored verbatim
after translation. The agent's D4 gate asserts every sentinel survived exactly once.
""",
        "model_selection.md": f"""
## The trainable MT core (reused from P13 s2st)
- **Base:** `{base_model(arts)}` (default `facebook/m2m100_418M`, MIT, **1024 positions** —
  headroom for concatenated context; `src_lang`/`forced_bos_token_id` control). Alternatives:
  `nllb-200-distilled-600M` (CC-BY-NC, flag), `mbart-large-50-many-to-many-mmt`, single-pair
  `opus-mt-en-fr` (Apache, 75M, 512 positions -> smaller context budget).
- **Baselines:** zero-shot base MT (floor to beat), a **dictionary** word-lookup (offline
  floor + fallback), and an **identity** copy-source floor.

## Document-level context (concat-k)
The architecture-free way to add document context: prepend the **k previous sentences** to
the current one, joined by a `<BRK>` separator, and train the model to output only the
current target sentence (the k-to-1 recipe; default **k={cfg.mt.context_k_train}**). The
context is left-truncated to fit the model's position budget; the current sentence is never
truncated. Inference uses context only with a context-fine-tuned model (recorded in the
model metadata).

## Optimization
HF `Seq2SeqTrainer`, {cfg.mt.num_train_epochs} epochs, lr {cfg.mt.learning_rate:g}, label
smoothing {cfg.mt.label_smoothing}, `predict_with_generate` ({cfg.mt.num_beams} beams),
bf16+tf32 on Ampere+ / fp16 on T4; selected on **chrF**. Effective batch held ~32 across
GPU tiers via grad-accum.
{res_line}
""",
        "agent_architecture.md": f"""
## FSM
A deterministic finite-state machine; every tool returns a uniform dict and every transition
is logged to a trace. States: `parse -> translate -> verify -> reassemble`. An optional LLM
**brain** (`{cfg.agent.llm_model}`, OFF by default) only writes an advisory terminology note;
rules win and the agent runs with **zero paid API calls**.

## Five decisions (each acts on an intermediate artifact)
- **D1 - format detect + parse.** Route by extension/content; parse; a **round-trip identity**
  check (re-join == source) guards against silent corruption -> fall back to plain on mismatch.
- **D2 - segment + mask gate.** Skip non-translatable segments (letter ratio <
  {cfg.agent.min_letter_ratio}); require masks reversible & balanced.
- **D3 - translate.** Translate each segment (optionally with k-sentence context); reject
  empty output or a length ratio outside [{cfg.agent.length_ratio_low}, {cfg.agent.length_ratio_high}]
  and re-translate (budget {cfg.agent.max_retranslate}).
- **D4 - verify gate.** **Placeholder-retention (hard):** every `[[PHn]]` must come back
  exactly once. **Round-trip back-translation chrF (soft):** below {cfg.agent.verify_min_chrf}
  -> flag. Failure re-translates or flags for review (never drops content).
- **D5 - reassemble + structure-validation gate.** Re-slot translations into the untouched
  skeleton, then assert the output's structure signature matches the source and it re-parses
  cleanly; on mismatch -> repair (budget {cfg.agent.max_repair}) then flag.

The agent emits `{{output, fmt, placeholder_retention, structure (SPS), needs_review,
decisions[], trace[]}}`. Structure-breaking / low-confidence output is **flagged for review**.
""",
        "evaluation.md": f"""
## Protocol
Two layers. **MT-level:** chrF/BLEU on held-out sentence pairs (model vs dictionary vs
identity floor). **Document-level:** run the full agent on structured documents and report
whole-document chrF vs gold + the **Structure-Preservation Score (SPS)** + **placeholder-
retention** — a translation that wins chrF but breaks a table is a failure, so they are
reported jointly.

## Metrics
- **chrF (headline)** + **BLEU** (sacrebleu; pure-python fallback offline).
- **SPS** = structure-match (heading/list/table/code/tag counts) x markup-validity (re-parses)
  x placeholder-retention.
- Document-level chrF (d-chrF) + the agent **needs-review rate**.

## Baselines & the context story
- **Zero-shot MT** (floor to beat), **dictionary**, **identity copy-source**.
- **Context vs no-context (concat-k):** chrF/BLEU move modestly, but discourse fixes
  (pronouns, terminology) show on contrastive sets (ContraPro-style) + term-consistency.
  Honest caveat: concat-k can underperform sentence-level when doc-parallel data is scarce.
{res_line}
""",
        "deployment.md": f"""
## Serving
A FastAPI process (`{cfg.serving.api_title}`, {cfg.serving.api_version}). Heavy deps are
lazy-imported so it answers in degraded mode (line/regex parsers + dictionary MT) without
GPU libraries.

## Endpoints
- `POST /translate-text` — `{{text, fmt?}}` -> translated document (same format) + the
  structure-preservation report + decision trace.
- `POST /translate-document` — upload a `.md`/`.html`/`.json`/`.txt` file -> translated file
  content (registered only when `python-multipart` is present).
- `GET /healthz` / `GET /version`.

## UI / packaging
A Gradio demo (paste a document -> translated document + structure report). CLI
`doctrans translate / evaluate / serve`. Docker + HF Space. Requests are logged (metadata
only) to `{cfg.serving.request_log_subdir}/requests.jsonl`.
""",
        "continual_learning_monitoring.md": """
## Continual learning
Collect domain documents/segments (with reference translations), re-fine-tune the MT core
(resume-safe), and promote only if held-out chrF is non-regressing -> a new `model_version`.
The structure layer is deterministic code (versioned, not trained).

## Monitoring
- **Quality:** chrF/BLEU on a rolling slice + SPS + the human-review rate (post-edit load).
- **Operational:** p50/p95 latency, the format mix, throughput.
- **Drift:** a rising needs-review rate or a falling mean SPS vs an earlier window flags
  harder documents / a parser regression (the `monitor-log` report). Logs are metadata-only.
""",
        "privacy_robustness.md": """
## Data privacy
- Documents may contain sensitive content; the core path is local / no-network, the optional
  LLM brain is opt-in and OFF by default, and request logs store metadata only (not raw
  document text). Uploaded files are processed transiently.
- **License hygiene:** the redistributable stack is m2m100 (MIT) + the seed; opus-100 /
  news_commentary (training data) carry unknown-license flags, NLLB is CC-BY-NC.

## Robustness
- **Structure corruption** — the round-trip identity check (D1) + the structure-validation
  gate (D5) + the byte-stable skeleton prevent silent structure loss.
- **Placeholder/markup loss** — the hard placeholder-retention gate (D4) + re-translation +
  fail-soft (keep source for an unrecoverable segment) protect code/URLs/`{vars}`.
- **Graceful degradation** — every parser degrades native -> regex -> plain; the MT degrades
  to a dictionary; so the service answers even without torch / parser libs / a network.
- **Abstention** — structure-breaking or low-confidence output is flagged for human review.
""",
        "ethics_statement.md": """
## Intended use & dual-use
A localization **aid**: it produces a structure-preserving draft translation + a confidence
/ structure report for a human to review. Treating low-confidence or structure-flagged output
as final would be a misuse — the review flag exists for exactly that.

## Key risks & mitigations
- **Mistranslation harm** (legal/medical/safety docs) -> the verify + structure gates flag
  doubtful output; abstention is a feature.
- **Bias & fairness** -> MT quality varies by language/domain; per-direction numbers reported,
  no over-claiming on low-resource languages.
- **Structure/placeholder corruption** -> hard placeholder-retention + structure-validation
  gates; fail-soft keeps the source rather than emitting a broken document.
- **License misuse** -> opus-100/news_commentary (unknown) + NLLB (CC-BY-NC) flagged; the
  redistributable path is m2m100 (MIT) + the seed.

The default configuration is offline-capable, preserves structure byte-stably, flags
low-confidence / structure-breaking output, and attaches a decision trace to every result.
""",
    }


def _esc(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", s)
    s = s.replace("&", "&amp;").replace("<b>", "\x00b\x00").replace("</b>", "\x00/b\x00")
    s = s.replace("<font face='Courier'>", "\x00f\x00").replace("</font>", "\x00/f\x00")
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    s = (s.replace("\x00b\x00", "<b>").replace("\x00/b\x00", "</b>")
          .replace("\x00f\x00", "<font face='Courier'>").replace("\x00/f\x00", "</font>"))
    return s


def _md_to_flowables(md: str, styles, max_lines: int = 300):
    from reportlab.platypus import Paragraph, Preformatted, Spacer
    flow, lines, in_code, code, bullet = [], md.splitlines()[:max_lines], False, [], []

    def flush():
        nonlocal bullet
        for b in bullet:
            flow.append(Paragraph("- " + _esc(b), styles["Body"]))
        bullet = []

    for ln in lines:
        if ln.strip().startswith("```"):
            if in_code:
                flow.append(Preformatted("\n".join(code), styles["Code"])); code = []
            in_code = not in_code
            continue
        if in_code:
            code.append(ln); continue
        s = ln.rstrip()
        if not s:
            flush(); flow.append(Spacer(1, 5)); continue
        if s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            flow.append(Paragraph(_esc(s.lstrip("#").strip()), styles["H2" if level <= 2 else "H3"]))
        elif s.lstrip().startswith(("- ", "* ")):
            bullet.append(s.lstrip()[2:])
        else:
            flush(); flow.append(Paragraph(_esc(s), styles["Body"]))
    flush()
    return flow


def _results_tables(arts: Dict[str, Any], styles):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    flow = [Paragraph("Results - MT quality (held-out pairs)", styles["H3"])]
    rows = [["Metric", "identity floor", "dictionary", "model"]]
    if has_eval(arts):
        for key, label in [("chrf", "chrF ^"), ("bleu", "BLEU ^")]:
            iv = mt_metric(arts, "identity", key)
            dv = mt_metric(arts, "dictionary", key)
            mv = mt_metric(arts, "model", key)
            rows.append([label, f"{iv:.1f}" if iv is not None else "-",
                         f"{dv:.1f}" if dv is not None else "-",
                         f"{mv:.1f}" if mv is not None else "-"])
    else:
        rows.append(["run `evaluate`", "-", "-", "-"])
    t = Table(rows, hAlign="LEFT", colWidths=[120, 100, 100, 100])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")])]))
    flow += [t, Spacer(1, 6),
             Paragraph(f"MT model: <b>{model_name(arts)}</b> (base {base_model(arts)})"
                       + ("  (beats dictionary)" if beats_baseline(arts) else ""), styles["Body"]),
             Spacer(1, 8)]

    flow.append(Paragraph("Results - document fidelity (structured docs)", styles["H3"]))
    drows = [["Metric", "value"]]
    for key, label in [("doc_chrf", "Document chrF ^"), ("doc_sps", "Structure-Preservation (SPS) ^"),
                       ("doc_phr", "Placeholder-retention ^")]:
        v = doc_metric(arts, key)
        drows.append([label, f"{v:.3f}" if v is not None else "-"])
    dt = Table(drows, hAlign="LEFT", colWidths=[240, 120])
    dt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f855a")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eaf5ee")])]))
    flow += [dt, Spacer(1, 6),
             Paragraph("A translation that wins chrF but breaks structure is a failure, so chrF and "
                       "structure-preservation are reported jointly.", styles["Body"])]
    lat = latency(arts, "p50")
    if lat is not None:
        flow.append(Paragraph(f"Agent latency: per-document p50 ~ {lat:.0f} ms "
                              f"(p95 ~ {latency(arts, 'p95') or 0:.0f} ms).", styles["Body"]))
    flow.append(Spacer(1, 8))
    return flow


def generate_report(cfg: AppConfig, title: Optional[str] = None, author: Optional[str] = None,
                    out_path: Optional[str] = None) -> str:
    title = title or cfg.project_title
    author = author or cfg.author
    arts = load_artifacts(cfg)
    out = Path(out_path) if out_path else run_dir() / "report" / "report.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    builtins = _builtin_sections(cfg, arts)

    def section_md(fname: str) -> str:
        doc = read_doc(fname)
        if doc.strip():
            lines = doc.splitlines()
            return "\n".join(lines[:42]) if len(lines) > 42 else doc
        return builtins.get(fname, "")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer)
    except Exception as exc:
        logger.warning("reportlab unavailable (%s); writing markdown report", exc)
        md = f"# {title}\n\n{author} (Student {cfg.student_id})\n\n{_SUBTITLE}\n\n"
        for hd, fn in _SECTIONS:
            md += f"\n\n# {hd}\n" + section_md(fn)
        alt = out.with_suffix(".md")
        alt.write_text(md, encoding="utf-8")
        return str(alt)

    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle("T", parent=base["Title"], fontSize=22, leading=26),
        "H2": ParagraphStyle("H2", parent=base["Heading2"], textColor="#1a365d", spaceBefore=10),
        "H3": ParagraphStyle("H3", parent=base["Heading3"], textColor="#2b6cb0"),
        "Body": ParagraphStyle("B", parent=base["BodyText"], fontSize=9.5, leading=13),
        "Code": ParagraphStyle("C", parent=base["Code"], fontSize=7.5, leading=9, backColor="#f4f6f8"),
        "Meta": ParagraphStyle("M", parent=base["BodyText"], fontSize=11, leading=15),
    }
    try:
        built = dict(charts_mod.build_all(arts, out.parent / "charts"))
    except Exception as exc:
        logger.info("charts skipped (%s)", exc)
        built = {}

    story: List[Any] = [
        Spacer(1, 5 * cm), Paragraph(title, styles["Title"]), Spacer(1, 1 * cm),
        Paragraph(f"<b>{author}</b> - Student {cfg.student_id}", styles["Meta"]),
        Paragraph("NLP in Industry - Final Assignment (P14)", styles["Meta"]),
        Paragraph(_SUBTITLE, styles["Meta"]),
        Paragraph(f"Generated {utc_now_iso()}", styles["Body"]),
        Paragraph(f"MT core: <b>{model_version(arts)}</b> (base {base_model(arts)})", styles["Body"]),
    ]
    story.append(PageBreak())
    story += _results_tables(arts, styles)
    for name in ("quality", "fidelity", "buckets"):
        if name in built:
            story += [Image(str(built[name]), width=13 * cm, height=7.0 * cm), Spacer(1, 6)]
    story.append(PageBreak())

    for heading, fname in _SECTIONS:
        story.append(Paragraph(heading, styles["H2"]))
        story += _md_to_flowables(section_md(fname), styles)
        story.append(Spacer(1, 10))

    try:
        SimpleDocTemplate(str(out), pagesize=A4, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                          leftMargin=1.8 * cm, rightMargin=1.8 * cm, title=title, author=author).build(story)
    except Exception as exc:
        logger.warning("reportlab build failed (%s); writing markdown report", exc)
        md = f"# {title}\n\n{author} (Student {cfg.student_id})\n\n{_SUBTITLE}\n\n"
        for hd, fn in _SECTIONS:
            md += f"\n\n# {hd}\n" + section_md(fn)
        alt = out.with_suffix(".md")
        alt.write_text(md, encoding="utf-8")
        return str(alt)
    logger.info("Report -> %s", out)
    return str(out)


__all__ = ["generate_report"]
