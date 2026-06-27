"""Shared state types for the document-translation agent (deterministic FSM)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    PARSED = "parsed"
    TRANSLATED = "translated"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"   # a quality gate fired (placeholder loss / structure mismatch / low chrF)
    FAILED = "failed"


@dataclass
class ToolTrace:
    tool: str
    ok: bool
    latency_ms: float
    summary: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "ok": self.ok, "latency_ms": self.latency_ms,
                "summary": self.summary, "error": self.error}


@dataclass
class Decision:
    id: str               # D1..D5
    name: str
    branch: str
    score: Optional[float] = None
    detail: str = ""
    llm_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "branch": self.branch,
                "score": self.score, "detail": self.detail, "llm_used": self.llm_used}


@dataclass
class JobState:
    # ---- inputs --------------------------------------------------------------
    source: str = ""
    fmt: str = ""                  # detected/declared format
    filename: str = ""
    src_lang: str = "en"
    tgt_lang: str = "fr"
    # ---- derived -------------------------------------------------------------
    status: JobStatus = JobStatus.PENDING
    n_segments: int = 0
    n_translatable: int = 0
    # ---- outputs -------------------------------------------------------------
    output: str = ""               # the translated document (same format)
    placeholder_retention: Optional[float] = None
    structure: Dict[str, Any] = field(default_factory=dict)   # sps report
    verify_chrf: Optional[float] = None
    n_retranslated: int = 0
    needs_review: bool = False
    rationale: str = ""
    # ---- audit ---------------------------------------------------------------
    decisions: List[Decision] = field(default_factory=list)
    trace: List[ToolTrace] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    model_versions: Dict[str, str] = field(default_factory=dict)

    def add_trace(self, t: ToolTrace) -> None:
        self.trace.append(t)

    def add_decision(self, d: Decision) -> None:
        self.decisions.append(d)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fmt": self.fmt, "filename": self.filename, "src_lang": self.src_lang, "tgt_lang": self.tgt_lang,
            "status": self.status.value, "n_segments": self.n_segments, "n_translatable": self.n_translatable,
            "output": self.output, "placeholder_retention": self.placeholder_retention,
            "structure": self.structure, "verify_chrf": self.verify_chrf,
            "n_retranslated": self.n_retranslated, "needs_review": self.needs_review, "rationale": self.rationale,
            "decisions": [d.to_dict() for d in self.decisions],
            "trace": [t.to_dict() for t in self.trace],
            "metrics": self.metrics, "model_versions": self.model_versions,
        }


__all__ = ["JobStatus", "ToolTrace", "Decision", "JobState"]
