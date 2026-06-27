"""Pydantic request/response schemas for the document-translation API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TranslateTextRequest(BaseModel):
    text: str = Field(..., description="Document content (markdown/html/json/plain) to translate")
    fmt: str = Field("", description="Optional format hint: markdown | html | json | plain")


class TranslateResponse(BaseModel):
    fmt: str = ""
    src_lang: str = ""
    tgt_lang: str = ""
    output: str = ""
    status: str = ""
    n_translatable: int = 0
    placeholder_retention: Optional[float] = None
    structure: Dict[str, Any] = {}
    verify_chrf: Optional[float] = None
    needs_review: bool = False
    rationale: str = ""
    decisions: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}
    model_versions: Dict[str, str] = {}
    disclaimer: str = "Machine translation — structure-breaking or low-confidence output is flagged for human review."


class HealthResponse(BaseModel):
    status: str
    mt: str
    formats: List[str]
    version: str


__all__ = ["TranslateTextRequest", "TranslateResponse", "HealthResponse"]
