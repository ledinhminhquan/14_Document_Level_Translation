"""Offline evaluate() + grading checklist + monitoring-report smoke tests."""

from __future__ import annotations

from pathlib import Path

from doctrans.training.evaluate import evaluate
from doctrans.training.metrics import chrf, bleu
from doctrans.monitoring.drift_report import monitoring_report
from doctrans.grading.checklist import build_checklist


def test_evaluate_offline(cfg):
    res = evaluate(cfg, save=False, load_model=False)
    assert res["model"]["chrf"] is not None
    assert res["summary"]["beats_identity"] is True
    assert res["document"]["doc_sps"] >= 0.99      # structure preserved on the seed docs


def test_chrf_bleu():
    assert chrf(["the cat"], ["the cat"]) > 95.0
    assert chrf(["the cat"], ["the cat"]) > chrf(["xyz"], ["the cat"])
    # corpus BLEU needs >=4-grams to be non-zero; use a long-enough exact match
    assert bleu(["the cat sat on the mat today"], ["the cat sat on the mat today"]) > 0.0


def test_monitoring_report_no_data(cfg):
    rep = monitoring_report(cfg, log_path="/nonexistent/path.jsonl", save=False)
    assert rep["status"] == "no_data"


def test_grade_runs():
    repo = Path(__file__).resolve().parents[1]
    res = build_checklist(repo)
    assert res["summary"]["total"] > 0
    assert res["summary"]["FAIL"] == 0   # full repo -> no FAILs
