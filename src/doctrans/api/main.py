"""FastAPI service for the structured document-level translation system.

Endpoints
---------
* ``GET  /healthz`` / ``GET /readyz`` / ``GET /version``
* ``POST /translate-text``     – {text, fmt?} -> translated document (same format)
* ``POST /translate-document`` – upload a .md/.html/.json/.txt file -> translated file content

The upload route is registered only when ``python-multipart`` is importable (so
``api.main`` imports anywhere). Structure-breaking / low-confidence output is flagged.
"""

from __future__ import annotations

from importlib.util import find_spec

from fastapi import FastAPI, HTTPException

from .. import __version__
from ..logging_utils import get_logger
from .dependencies import get_agent, get_config
from .schemas import HealthResponse, TranslateResponse, TranslateTextRequest

logger = get_logger(__name__)
cfg = get_config()
app = FastAPI(title=cfg.serving.api_title, version=cfg.serving.api_version)

_HAS_MULTIPART = find_spec("multipart") is not None or find_spec("python_multipart") is not None


def _resp(sd: dict) -> TranslateResponse:
    return TranslateResponse(
        fmt=sd["fmt"], src_lang=sd["src_lang"], tgt_lang=sd["tgt_lang"], output=sd["output"],
        status=sd["status"], n_translatable=sd["n_translatable"],
        placeholder_retention=sd["placeholder_retention"], structure=sd["structure"],
        verify_chrf=sd["verify_chrf"], needs_review=sd["needs_review"], rationale=sd["rationale"],
        decisions=sd["decisions"], trace=sd["trace"], metrics=sd["metrics"],
        model_versions=sd["model_versions"])


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    agent = get_agent()
    return HealthResponse(status="ok", mt=getattr(agent.translator, "name", "?"),
                          formats=["markdown", "html", "json", "plain"], version=__version__)


@app.get("/readyz")
def readyz() -> dict:
    get_agent()
    return {"status": "ready"}


@app.get("/version")
def version() -> dict:
    agent = get_agent()
    return {"app": __version__, "mt": getattr(agent.translator, "version", "?"),
            "direction": f"{cfg.mt.src_lang}->{cfg.mt.tgt_lang}", "model_version": cfg.serving.model_version}


@app.post("/translate-text", response_model=TranslateResponse)
def translate_text(req: TranslateTextRequest) -> TranslateResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="provide document text")
    job = get_agent().run(req.text, fmt=req.fmt, save=True)
    return _resp(job.to_dict())


if _HAS_MULTIPART:
    from fastapi import File, UploadFile

    @app.post("/translate-document", response_model=TranslateResponse)
    def translate_document(file: "UploadFile" = File(...)) -> TranslateResponse:
        raw = file.file.read()
        try:
            text = raw.decode("utf-8")
        except Exception:
            text = raw.decode("latin-1", errors="replace")
        job = get_agent().run(text, filename=file.filename or "", save=True)
        return _resp(job.to_dict())
else:  # pragma: no cover
    logger.info("python-multipart not installed; /translate-document disabled (/translate-text still works).")


__all__ = ["app"]
