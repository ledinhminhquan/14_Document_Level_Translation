"""Agent tools — each operates on the JobState and returns it.

Tools wrap the structure layer (parse/mask/reassemble/validate) + the MT core + the
D1-D5 policy. They run against the offline stack (regex/line parsers + dictionary MT) so
the whole pipeline runs offline for tests/CI. The orchestrator wraps each call with
timing/trace; tools never raise past it.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..config import AppConfig
from ..docparse import mask as maskmod
from ..docparse import router, structure_score
from ..docparse.segment import restore_text
from ..logging_utils import get_logger
from ..mt import context as ctxmod
from ..training.metrics import chrf
from . import policy
from .state import Decision, JobState, JobStatus

logger = get_logger(__name__)


def tool_parse(job: JobState, cfg: AppConfig) -> JobState:
    """D1 — detect format + parse into a structure-preserving ParsedDoc."""
    fmt = job.fmt or router.detect_format(job.source, job.filename or None)
    doc = router.parse_document(job.source, fmt=fmt, cfg=cfg.doc, filename=job.filename or None)
    job.fmt = doc.fmt
    job._doc = doc  # type: ignore[attr-defined]
    job.n_segments = len(doc.segments)
    job.n_translatable = len(doc.translatable_segments)
    roundtrip = doc.reassemble() == job.source
    job.add_decision(Decision("D1", "format_parse", doc.fmt if roundtrip else f"{doc.fmt}->plain",
                              score=job.n_translatable,
                              detail=f"fmt={doc.fmt}, segments={job.n_segments}, roundtrip={roundtrip}"))
    job.status = JobStatus.PARSED
    return job


def tool_translate(job: JobState, cfg: AppConfig, *, translator) -> JobState:
    """D2 segment+mask gate (done by the parser) + D3 translate with retry on bad output."""
    doc = getattr(job, "_doc", None)
    if doc is None:
        return job
    segs = doc.translatable_segments
    # D2 — confirm masks are balanced + segments are translatable (parser already split)
    n_skipped = len(doc.segments) - len(segs)
    job.add_decision(Decision("D2", "segment_mask", "ok",
                              detail=f"translatable={len(segs)}, skipped_nontext={n_skipped}"))

    use_context = bool(cfg.mt.use_context and getattr(translator, "name", "") == "transformer")
    masked_srcs = [s.text for s in segs]
    if use_context:
        hyps = ctxmod.translate_with_context(translator, masked_srcs, k=cfg.mt.context_k_infer,
                                             brk=cfg.mt.brk_token, use_context=True)
    else:
        hyps = translator.translate_batch(masked_srcs) if masked_srcs else []

    retentions: List[float] = []
    n_retrans = 0
    bad = 0
    for seg, masked_hyp in zip(segs, hyps):
        sane = policy.translate_sane(seg.text, masked_hyp, cfg.agent)
        ret = maskmod.validate_restoration(masked_hyp, seg.mask_map)
        if (not sane["ok"] or not ret["ok"]) and cfg.agent.max_retranslate > 0:
            # re-translate this segment once WITHOUT context (large windows cause issues)
            retry = translator.translate(seg.text)
            rret = maskmod.validate_restoration(retry, seg.mask_map)
            if rret["ok"] or rret["retention"] > ret["retention"]:
                masked_hyp, ret = retry, rret
            n_retrans += 1
        if not ret["ok"]:
            bad += 1
            # placeholder loss unrecoverable -> keep source for this segment (fail-soft)
            seg.translation = restore_text(seg.text, seg.mask_map)
        else:
            seg.translation = maskmod.restore(masked_hyp, seg.mask_map)
        retentions.append(ret["retention"])

    job._retentions = retentions  # type: ignore[attr-defined]
    job.n_retranslated = n_retrans
    job.model_versions["mt"] = getattr(translator, "name", "?") + ":" + getattr(translator, "version", "?")
    job.add_decision(Decision("D3", "translate", "ok" if bad == 0 else "some_placeholder_loss",
                              score=len(segs), detail=f"retranslated={n_retrans}, placeholder_loss={bad}"))
    job.status = JobStatus.TRANSLATED
    return job


def tool_verify(job: JobState, cfg: AppConfig, *, back_translator=None) -> JobState:
    """D4 — placeholder retention (hard) + round-trip back-translation chrF (soft)."""
    doc = getattr(job, "_doc", None)
    retentions = getattr(job, "_retentions", []) or [1.0]
    phr = structure_score.placeholder_retention_rate(retentions)
    job.placeholder_retention = phr

    verify_chrf = None
    if cfg.agent.verify_enabled and back_translator is not None and doc is not None:
        segs = [s for s in doc.translatable_segments if s.translation][:20]   # sample
        if segs:
            try:
                backs = back_translator.translate_batch([s.translation for s in segs])
                srcs = [restore_text(s.text, s.mask_map) for s in segs]
                verify_chrf = round(chrf(backs, srcs) / 100.0, 4)
            except Exception as exc:
                logger.info("verify back-translation skipped (%s)", exc)
    job.verify_chrf = verify_chrf
    gate = policy.verify_gate(phr, verify_chrf, cfg.agent)
    if not gate["ok"]:
        job.needs_review = True
    job.add_decision(Decision("D4", "verify_gate", gate["branch"], score=phr,
                              detail=f"placeholder_retention={phr}, verify_chrf={verify_chrf}"))
    return job


def tool_reassemble(job: JobState, cfg: AppConfig) -> JobState:
    """D5 — reassemble + structure-validation gate."""
    doc = getattr(job, "_doc", None)
    if doc is None:
        job.status = JobStatus.FAILED
        return job
    job.output = doc.reassemble()
    sps = structure_score.sps_full(job.source, job.output, job.fmt,
                                   phr=job.placeholder_retention if job.placeholder_retention is not None else 1.0)
    job.structure = sps
    gate = policy.structure_gate(sps)
    if not gate["ok"]:
        job.needs_review = True
    job.add_decision(Decision("D5", "structure_gate", gate["branch"], score=sps.get("sps"),
                              detail=f"struct_match={sps.get('struct_match')}, sps={sps.get('sps')}"))
    job.status = JobStatus.NEEDS_REVIEW if job.needs_review else JobStatus.COMPLETED
    if not job.rationale:
        job.rationale = (f"Translated {job.src_lang}->{job.tgt_lang} {job.fmt} doc "
                         f"({job.n_translatable} segments) via {job.model_versions.get('mt','MT')}; "
                         f"SPS={sps.get('sps')}, placeholder_retention={job.placeholder_retention}.")
    return job


__all__ = ["tool_parse", "tool_translate", "tool_verify", "tool_reassemble"]
