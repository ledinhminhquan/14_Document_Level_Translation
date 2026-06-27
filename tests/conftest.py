"""Pytest fixtures — force everything offline + a temp artifacts dir."""

from __future__ import annotations

import os
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="doctrans-test-")
os.environ.setdefault("DOCTRANS_ARTIFACTS_DIR", _TMP)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture
def cfg():
    from doctrans.config import AppConfig
    c = AppConfig()
    c.data.use_hf = False
    return c
