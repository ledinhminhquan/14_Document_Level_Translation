"""The document-translation agent — a deterministic FSM over the structure-preserving cascade.

    parse (D1 format+parse) -> translate (D2 segment/mask gate + D3 translate)
        -> verify (D4 placeholder-retention + round-trip gate)
        -> reassemble (D5 structure-validation gate)

Holds the MT translator (+ an optional reverse translator for D4) loaded once. Runs fully
offline (regex/line parsers + dictionary MT) and upgrades to a fine-tuned/pretrained
seq2seq when present. The model never sees structure and structure never sees the model;
low-confidence / structure-breaking output is flagged for human review (needs_review).
Every step is timed and traced; same input + same models + brain disabled => identical output.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from ..config import AppConfig, ensure_dirs
from ..logging_utils import JsonlLogger, get_logger
from . import tools
from .llm_orchestrator import LLMBrain
from .state import JobState, JobStatus, ToolTrace

logger = get_logger(__name__)


class DocTransAgent:
    def __init__(self, cfg: Optional[AppConfig] = None, *, load_model: bool = True,
                 translator=None, back_translator=None):
        self.cfg = cfg or AppConfig()
        if translator is None:
            from ..mt.translator import load_translator
            translator = load_translator(self.cfg.mt, prefer="transformer" if load_model else "dictionary")
        self.translator = translator
        self.back_translator = back_translator
        if (self.back_translator is None and load_model
                and getattr(translator, "name", "") == "transformer" and self.cfg.agent.verify_enabled):
            try:
                from ..config import MtConfig
                from ..mt.translator import TransformerTranslator
                rev = MtConfig(**{**self.cfg.mt.__dict__})
                rev.src_lang, rev.tgt_lang = self.cfg.mt.tgt_lang, self.cfg.mt.src_lang
                self.back_translator = TransformerTranslator(translator.model, translator.tok, rev,
                                                             version=translator.version)
            except Exception as exc:
                logger.info("reverse translator unavailable (%s); D4 round-trip will skip", exc)
        self.brain = LLMBrain(self.cfg.agent)
        ensure_dirs()
        self._log = JsonlLogger(self.cfg.serving.request_log_path) if self.cfg.serving.log_requests else None

    def _step(self, job: JobState, name: str, fn: Callable[[], JobState], summary: str = "") -> JobState:
        t0 = time.perf_counter()
        try:
            job = fn()
            ok, err = True, None
        except Exception as exc:
            logger.warning("tool %s failed: %s", name, exc)
            ok, err = False, str(exc)
        job.add_trace(ToolTrace(tool=name, ok=ok, latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                                summary=summary or name, error=err))
        return job

    def run(self, source: str, *, fmt: str = "", filename: str = "", save: bool = True) -> JobState:
        job = JobState(source=source or "", fmt=fmt, filename=filename,
                       src_lang=self.cfg.mt.src_lang, tgt_lang=self.cfg.mt.tgt_lang)
        if not (source or "").strip():
            job.status = JobStatus.FAILED
            return job
        t0 = time.perf_counter()
        job = self._step(job, "parse", lambda: tools.tool_parse(job, self.cfg), summary="format+parse (D1)")
        job = self._step(job, "translate", lambda: tools.tool_translate(job, self.cfg, translator=self.translator),
                         summary="segment/mask + translate (D2,D3)")
        job = self._step(job, "verify", lambda: tools.tool_verify(job, self.cfg, back_translator=self.back_translator),
                         summary="verify gate (D4)")
        job = self._step(job, "reassemble", lambda: tools.tool_reassemble(job, self.cfg),
                         summary="reassemble + structure validation (D5)")

        if self.brain.available() and job.output:
            note = self.brain.consistency_note(job.src_lang, job.tgt_lang, job.output)
            if note:
                job.metrics["brain_note"] = note
                job.metrics["brain_used"] = True
        job.metrics["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        # drop the transient parse tree before returning
        if hasattr(job, "_doc"):
            delattr(job, "_doc")
        if save and self._log is not None:
            try:
                self._log.log("doc_translation", fmt=job.fmt, src_lang=job.src_lang, tgt_lang=job.tgt_lang,
                              status=job.status.value, needs_review=job.needs_review,
                              n_translatable=job.n_translatable, sps=job.structure.get("sps"),
                              metrics=job.metrics)
            except Exception:
                pass
        return job

    def translate_document(self, source: str, fmt: str = "", filename: str = "") -> dict:
        job = self.run(source, fmt=fmt, filename=filename, save=False)
        return {"output": job.output, "fmt": job.fmt, "sps": job.structure.get("sps"),
                "placeholder_retention": job.placeholder_retention, "needs_review": job.needs_review,
                "model_version": job.model_versions.get("mt", "?")}


_AGENT: Optional[DocTransAgent] = None


def get_agent(cfg: Optional[AppConfig] = None, **kwargs) -> DocTransAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = DocTransAgent(cfg, **kwargs)
    return _AGENT


__all__ = ["DocTransAgent", "get_agent"]
