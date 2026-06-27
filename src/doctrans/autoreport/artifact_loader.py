"""Collect generated run artifacts into one dict for the report + slides generators.

Reads the JSON written under ``run_dir()`` — the MT eval (chrF/BLEU model vs dictionary
vs identity + document-level chrF/SPS/PHR), the error analysis (chrF buckets + structure
issues), a latency benchmark, and a monitoring snapshot — plus the trained-MT metadata.
Every read is defensive: missing/malformed -> ``None``/``{}``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import AppConfig, run_dir
from ..models.model_registry import read_metadata, resolve_latest


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def load_artifacts(cfg: AppConfig) -> Dict[str, Any]:
    rd = run_dir()
    arts: Dict[str, Any] = {
        "eval": _load_json(rd / "eval" / "latest.json"),
        "error_analysis": _load_json(rd / "error_analysis" / "latest.json"),
        "benchmark": _load_json(rd / "benchmark" / "latest.json"),
        "tune": _load_json(rd / "tune" / "tune.json"),
        "monitoring": _load_json(rd / "monitoring" / "latest.json"),
    }
    try:
        latest = resolve_latest(cfg.mt.output_dir)
        arts["model_meta"] = read_metadata(latest) if latest else {}
    except Exception:
        arts["model_meta"] = {}
    return arts


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def mt_metric(arts: Dict[str, Any], which: str, key: str) -> Optional[float]:
    block = (arts.get("eval") or {}).get(which) or {}
    return _num(block.get(key))


def doc_metric(arts: Dict[str, Any], key: str) -> Optional[float]:
    block = (arts.get("eval") or {}).get("document") or {}
    return _num(block.get(key))


def has_eval(arts: Dict[str, Any]) -> bool:
    return bool((arts.get("eval") or {}).get("model"))


def summary_stat(arts: Dict[str, Any], key: str) -> Any:
    return ((arts.get("eval") or {}).get("summary") or {}).get(key)


def beats_baseline(arts: Dict[str, Any]) -> Optional[bool]:
    v = summary_stat(arts, "beats_dictionary")
    return bool(v) if isinstance(v, bool) else None


def model_name(arts: Dict[str, Any]) -> str:
    return str(summary_stat(arts, "model_name") or (arts.get("eval") or {}).get("model_name")
               or "dictionary (offline fallback)")


def model_version(arts: Dict[str, Any]) -> str:
    mv = arts.get("model_meta") or {}
    return str(mv.get("version") or "untrained (dictionary fallback)")


def base_model(arts: Dict[str, Any]) -> str:
    mv = arts.get("model_meta") or {}
    return str(mv.get("base_model") or "facebook/m2m100_418M")


def buckets(arts: Dict[str, Any]) -> Dict[str, Optional[float]]:
    ea = arts.get("error_analysis") or {}
    return {"good": _num(ea.get("good")), "medium": _num(ea.get("medium")), "poor": _num(ea.get("poor"))}


def latency(arts: Dict[str, Any], pct: str = "p50") -> Optional[float]:
    b = (arts.get("benchmark") or {}).get("latency_ms") or {}
    return _num(b.get(pct))


def read_doc(name: str) -> str:
    p = repo_root() / "docs" / name
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


__all__ = ["load_artifacts", "read_doc", "repo_root", "mt_metric", "doc_metric", "has_eval",
           "summary_stat", "beats_baseline", "model_name", "model_version", "base_model",
           "buckets", "latency"]
