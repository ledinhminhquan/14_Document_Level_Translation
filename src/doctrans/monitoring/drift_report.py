"""Production monitoring report from the serving request log (doc_translation JSONL).

Turns raw ``doc_translation`` events into a health picture: request volume, the format
mix, status mix, the **needs-review rate** (human post-edit load), the mean SPS, latency
(mean + p95), and a drift signal comparing a recent window to an earlier baseline (a
rising needs-review rate or falling SPS is the tell-tale of harder documents or a
degrading model). Stdlib only; never raises past its entrypoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)

_EVENT = "doc_translation"


def _read_logs(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("could not read request log %s: %s", path, exc)
        return rows
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 1)
    rank = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(float(ordered[rank]), 1)


def _mean(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _window_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    statuses: Dict[str, int] = {}
    formats: Dict[str, int] = {}
    needs_review = 0
    lats: List[float] = []
    spss: List[float] = []
    for r in rows:
        statuses[str(r.get("status", "?"))] = statuses.get(str(r.get("status", "?")), 0) + 1
        formats[str(r.get("fmt", "?"))] = formats.get(str(r.get("fmt", "?")), 0) + 1
        if bool(r.get("needs_review")):
            needs_review += 1
        if _is_num(r.get("sps")):
            spss.append(float(r["sps"]))
        metrics = r.get("metrics") or {}
        if isinstance(metrics, dict) and _is_num(metrics.get("latency_ms")):
            lats.append(float(metrics["latency_ms"]))
    return {"n": n, "status_distribution": statuses, "format_distribution": formats,
            "needs_review_rate": round(needs_review / n, 4), "mean_sps": _mean(spss),
            "mean_latency_ms": _mean(lats), "p95_latency_ms": _percentile(lats, 95)}


def _delta(base: Dict[str, Any], recent: Dict[str, Any], key: str) -> Optional[float]:
    a, b = base.get(key), recent.get(key)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return round(float(b) - float(a), 4)
    return None


def _drift(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(rows) < 6:
        return {"available": False, "reason": "need >=6 events to split baseline/recent windows"}
    half = len(rows) // 2
    base = _window_stats(rows[:half])
    recent = _window_stats(rows[half:])
    d_nr = _delta(base, recent, "needs_review_rate")
    d_sps = _delta(base, recent, "mean_sps")
    d_lat = _delta(base, recent, "mean_latency_ms")
    flags: List[str] = []
    if (d_nr or 0) > 0.15:
        flags.append("rising_needs_review_rate")
    if d_sps is not None and d_sps < -0.05:
        flags.append("falling_structure_preservation")
    if d_lat is not None and base.get("mean_latency_ms"):
        if d_lat / (base["mean_latency_ms"] or 1.0) > 0.5:
            flags.append("latency_regression")
    return {"available": True, "baseline_window": base, "recent_window": recent,
            "delta_needs_review_rate": d_nr, "delta_mean_sps": d_sps, "delta_mean_latency_ms": d_lat,
            "flags": flags, "alert": bool(flags)}


def _recommendations(overall: Dict[str, Any], drift: Dict[str, Any]) -> List[str]:
    recs: List[str] = []
    nr = overall.get("needs_review_rate") or 0.0
    sps = overall.get("mean_sps")
    flags = drift.get("flags") or []
    if nr > 0.4:
        recs.append("High needs-review rate ({:.0%}): check placeholder masking + the verify threshold, "
                    "and consider re-fine-tuning the MT core on the failing domain.".format(nr))
    if _is_num(sps) and sps < 0.9:
        recs.append("Mean structure-preservation {:.2f} is low: a parser/masking issue is corrupting "
                    "structure - inspect the failing formats.".format(sps))
    if "rising_needs_review_rate" in flags or "falling_structure_preservation" in flags:
        recs.append("Quality is drifting vs the baseline window: incoming documents may be harder "
                    "(new formats/domains) - collect a fresh slice and re-evaluate.")
    if not recs:
        recs.append("No action needed: monitoring metrics within healthy operating ranges.")
    return recs


def monitoring_report(cfg: AppConfig, log_path: Optional[str] = None, save: bool = True) -> Dict[str, Any]:
    path = Path(log_path) if log_path else cfg.serving.request_log_path
    rows = _read_logs(path)
    events = [r for r in rows if r.get("event", _EVENT) == _EVENT]
    if not events:
        logger.info("monitoring: no doc_translation events at %s", path)
        result = {"status": "no_data", "log_path": str(path), "n_events": 0, "request_volume": 0,
                  "overall": {"n": 0}, "drift": {"available": False, "reason": "no events"},
                  "recommendations": ["No request logs yet: exercise the agent / API to populate the log."],
                  "generated_at": utc_stamp()}
    else:
        overall = _window_stats(events)
        drift = _drift(events)
        result = {"status": "ok", "log_path": str(path), "n_events": len(events),
                  "request_volume": len(events), "overall": overall, "drift": drift,
                  "recommendations": _recommendations(overall, drift), "generated_at": utc_stamp()}
    if save:
        try:
            out = run_dir() / "monitoring"
            out.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(result, indent=2, ensure_ascii=False)
            (out / f"monitor-{utc_stamp()}.json").write_text(payload, encoding="utf-8")
            (out / "latest.json").write_text(payload, encoding="utf-8")
        except Exception as exc:
            logger.warning("monitoring: could not save report: %s", exc)
    logger.info("monitoring: %s events, needs_review=%.0f%% mean_sps=%s p95=%s ms, drift_alert=%s",
                result["n_events"], 100 * (result["overall"].get("needs_review_rate") or 0.0),
                result["overall"].get("mean_sps"), result["overall"].get("p95_latency_ms"),
                result.get("drift", {}).get("alert", False))
    return result


__all__ = ["monitoring_report"]
