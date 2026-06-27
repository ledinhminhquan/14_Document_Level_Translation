"""Typed configuration + YAML loader for the doctrans document-translation system.

Single source of truth for the trainable MT core (+ document-level concat-k context),
the document structure layer (formats, sentinel masking), the agent decision thresholds
(D1-D5), the datasets, and serving. Paths come from environment variables so nothing is
hard-coded. Default direction: English -> French (configurable via ``DataConfig.src_lang``
/ ``tgt_lang`` and ``MtConfig``).

Environment overrides
---------------------
* ``DOCTRANS_ARTIFACTS_DIR`` – base for data/models/runs (Drive on Colab)
* ``DOCTRANS_DATA_DIR``      – dataset cache / processed
* ``DOCTRANS_MODEL_DIR``     – trained models (the fine-tuned MT core)
* ``DOCTRANS_RUN_DIR``       – eval/benchmark/analysis JSON + translated documents
* ``HF_HOME``                – HuggingFace cache
* ``DOCTRANS_LLM_API_KEY``   – optional key for the LLM agent brain

Verified ids (confirmed on the HF Hub during research — keep exact):
  fine-tune  Helsinki-NLP/opus-100 (en-fr; license unknown flag)
  context    Helsinki-NLP/news_commentary (en-fr; monotonic per-article id -> real adjacency)
  mt         facebook/m2m100_418M (MIT, 1024 pos, default) · nllb-200-distilled-600M (CC-BY-NC flag) ·
             Helsinki-NLP/opus-mt-en-fr (Apache, 512 pos) — see docs/DESIGN_BRIEF.md
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def artifacts_dir() -> Path:
    return Path(_env("DOCTRANS_ARTIFACTS_DIR", "artifacts")).expanduser()


def data_dir() -> Path:
    return Path(_env("DOCTRANS_DATA_DIR", str(artifacts_dir() / "data"))).expanduser()


def model_dir() -> Path:
    return Path(_env("DOCTRANS_MODEL_DIR", str(artifacts_dir() / "models"))).expanduser()


def run_dir() -> Path:
    return Path(_env("DOCTRANS_RUN_DIR", str(artifacts_dir() / "runs"))).expanduser()


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Parallel corpora for fine-tuning + the document-context corpus."""
    # PRIMARY fine-tune corpus (VERIFIED): Helsinki-NLP/opus-100 en-fr (translation {en,fr}).
    mt_dataset: str = "Helsinki-NLP/opus-100"
    mt_config: str = "en-fr"
    # context corpus (VERIFIED): news_commentary en-fr — `id` is a monotonic per-article
    # sentence index -> consecutive rows are adjacent sentences (real concat-k context).
    context_dataset: str = "Helsinki-NLP/news_commentary"
    context_config: str = "en-fr"
    src_lang: str = "en"
    tgt_lang: str = "fr"
    use_hf: bool = True
    max_train_samples: int = 40000
    max_eval_samples: int = 2000
    seed: int = 42


@dataclass
class MtConfig:
    """The TRAINABLE MT core (the NLP heart) + document-level concat-k context."""
    base_model: str = "facebook/m2m100_418M"  # MIT, 1024 positions. alt nllb-200-distilled-600M (CC-BY-NC)
    src_lang: str = "en"
    tgt_lang: str = "fr"
    max_source_length: int = 256
    max_target_length: int = 160
    num_beams: int = 4
    # document-level context (concat-k, source-side, k-to-1)
    context_k_train: int = 2          # previous sentences prepended as source context at train time
    context_k_infer: int = 2          # context window at inference
    use_context: bool = False         # apply concat-k at inference (ONLY with a context-trained model)
    brk_token: str = "<BRK>"          # separator between context sentences and the current one
    # training (HF Seq2SeqTrainer)
    num_train_epochs: int = 3
    learning_rate: float = 3.0e-5
    per_device_train_batch_size: int = 16
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    label_smoothing: float = 0.1
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    eval_steps: int = 500
    save_steps: int = 500
    seed: int = 42
    output_subdir: str = "mt"
    baseline_filename: str = "dictionary_mt.json"

    @property
    def output_dir(self) -> Path:
        return model_dir() / self.output_subdir

    @property
    def baseline_path(self) -> Path:
        return self.output_dir / self.baseline_filename


@dataclass
class DocConfig:
    """Document structure layer: formats + the sentinel masking scheme."""
    # sentinel for masked non-translatable spans (ASCII-safe; Windows cp1252 safe).
    sentinel_format: str = "[[PH{}]]"
    mask_code: bool = True            # fenced/inline code
    mask_urls: bool = True            # URLs + emails
    mask_placeholders: bool = True    # {name} {{var}} %s ${VAR} :named
    mask_html_tags: bool = True       # <tag ...>
    mask_numbers: bool = False        # version strings / codes (off: numbers usually fine)
    max_segment_chars: int = 4000


@dataclass
class AgentConfig:
    """Document-translation agent decision thresholds (D1-D5) + optional LLM brain."""
    # D2 — segment + mask
    min_letter_ratio: float = 0.30        # below this letters/non-space -> non-translatable, skip MT
    # D3 — translate sanity
    length_ratio_low: float = 0.4
    length_ratio_high: float = 3.0
    # D4 — verify gate (round-trip back-translation + placeholder retention)
    verify_enabled: bool = True
    verify_min_chrf: float = 0.30         # round-trip chrF (0..1) below this -> flag
    max_retranslate: int = 2              # D4 -> D3 re-translate budget (N)
    # D5 — reassemble + structure validation
    max_repair: int = 1                   # D5 repair budget (M)
    # optional cloud brain (off by default; the agent runs fully on rules)
    llm_fallback_enabled: bool = False
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_api_key_env: str = "DOCTRANS_LLM_API_KEY"


@dataclass
class ServingConfig:
    model_version: str = "v1"
    api_title: str = "Structured Document-Level Translation API"
    api_version: str = "1.0.0"
    log_requests: bool = True
    request_log_subdir: str = "request_logs"
    max_doc_chars: int = 200000

    @property
    def request_log_path(self) -> Path:
        return run_dir() / self.request_log_subdir / "requests.jsonl"


@dataclass
class AppConfig:
    project_title: str = "Structured Document-Level Translation System"
    author: str = "Le Dinh Minh Quan"
    student_id: str = "23127460"
    data: DataConfig = field(default_factory=DataConfig)
    mt: MtConfig = field(default_factory=MtConfig)
    doc: DocConfig = field(default_factory=DocConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SECTIONS = {"data": DataConfig, "mt": MtConfig, "doc": DocConfig,
             "agent": AgentConfig, "serving": ServingConfig}


def _build(cls, raw: Optional[Dict[str, Any]]):
    raw = raw or {}
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: Optional[str | os.PathLike] = None) -> AppConfig:
    raw: Dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    top = {k: raw[k] for k in ("project_title", "author", "student_id") if k in raw}
    sections = {name: _build(cls, raw.get(name)) for name, cls in _SECTIONS.items()}
    return AppConfig(**top, **sections)


def save_config(cfg: AppConfig, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def ensure_dirs() -> Dict[str, Path]:
    dirs = {"artifacts": artifacts_dir(), "data": data_dir(), "models": model_dir(), "runs": run_dir()}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


__all__ = ["DataConfig", "MtConfig", "DocConfig", "AgentConfig", "ServingConfig", "AppConfig",
           "load_config", "save_config", "ensure_dirs",
           "artifacts_dir", "data_dir", "model_dir", "run_dir"]
