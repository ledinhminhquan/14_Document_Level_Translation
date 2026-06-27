"""Parallel-corpus loading (OPUS-100 sentence pairs + news_commentary document context).

* ``load_pairs`` — flat (src, tgt) sentence pairs from OPUS-100 en-fr (the fine-tune corpus).
* ``load_context_articles`` — news_commentary en-fr grouped into *articles* by its monotonic
  per-article ``id`` (a reset/decrease starts a new article), so the previous-sentence
  context for concat-k training is REAL document adjacency.

Both fall back to the built-in seed so everything runs offline. ``datasets`` is lazy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import AppConfig
from ..logging_utils import get_logger
from . import samples

logger = get_logger(__name__)


@dataclass
class Pair:
    id: str
    src: str
    tgt: str

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "src": self.src, "tgt": self.tgt}


def load_pairs(cfg: AppConfig, split: str = "train", limit: Optional[int] = None) -> List[Pair]:
    """Flat sentence pairs from OPUS-100 en-fr; fall back to the seed."""
    dc = cfg.data
    cap = limit or (dc.max_train_samples if split == "train" else dc.max_eval_samples)
    if dc.use_hf:
        try:
            from datasets import load_dataset  # lazy
            ds = load_dataset(dc.mt_dataset, dc.mt_config, split=split, streaming=True)
            out: List[Pair] = []
            for i, r in enumerate(ds):
                if len(out) >= cap:
                    break
                tr = r.get("translation") or {}
                src = str(tr.get(dc.src_lang, "") or "").strip()
                tgt = str(tr.get(dc.tgt_lang, "") or "").strip()
                if src and tgt:
                    out.append(Pair(id=f"{split}{i:06d}", src=src[:1000], tgt=tgt[:1000]))
            if len(out) > 2:
                logger.info("Loaded %d %s pairs from %s", len(out), split, dc.mt_dataset)
                return out
        except Exception as exc:
            logger.warning("Could not load %s (%s); using seed.", dc.mt_dataset, exc)
    return load_seed_pairs()


def load_context_articles(cfg: AppConfig, limit_rows: Optional[int] = None) -> List[List[Pair]]:
    """news_commentary en-fr grouped into articles (by the monotonic per-article `id`).

    Returns a list of articles, each a list of in-order Pairs -> real concat-k context.
    Falls back to one article = the seed pairs.
    """
    dc = cfg.data
    cap = limit_rows or dc.max_train_samples
    if dc.use_hf:
        try:
            from datasets import load_dataset  # lazy
            ds = load_dataset(dc.context_dataset, dc.context_config, split="train", streaming=True)
            articles: List[List[Pair]] = []
            cur: List[Pair] = []
            prev_id = -1
            for i, r in enumerate(ds):
                if i >= cap:
                    break
                tr = r.get("translation") or {}
                src = str(tr.get(dc.src_lang, "") or "").strip()
                tgt = str(tr.get(dc.tgt_lang, "") or "").strip()
                try:
                    rid = int(r.get("id", i))
                except Exception:
                    rid = i
                if rid <= prev_id and cur:        # id reset/decrease -> new article
                    articles.append(cur); cur = []
                prev_id = rid
                if src and tgt:
                    cur.append(Pair(id=f"a{len(articles)}s{rid}", src=src[:1000], tgt=tgt[:1000]))
            if cur:
                articles.append(cur)
            articles = [a for a in articles if len(a) >= 2]
            if articles:
                logger.info("Loaded %d articles (%d pairs) from %s",
                            len(articles), sum(len(a) for a in articles), dc.context_dataset)
                return articles
        except Exception as exc:
            logger.warning("Could not load %s (%s); using seed article.", dc.context_dataset, exc)
    return [load_seed_pairs()]


def load_seed_pairs() -> List[Pair]:
    return [Pair(id=p["id"], src=p["src"], tgt=p["tgt"]) for p in samples.pairs()]


def seed_split(seed: int = 42, eval_frac: float = 0.3):
    import random
    items = load_seed_pairs()
    rng = random.Random(seed)
    rng.shuffle(items)
    n_eval = max(2, int(len(items) * eval_frac))
    return items[n_eval:], items[:n_eval]


def load_eval_pairs(cfg: AppConfig, limit: Optional[int] = None) -> List[Pair]:
    if cfg.data.use_hf:
        for split in ("validation", "test"):
            try:
                out = load_pairs(cfg, split=split, limit=limit or cfg.data.max_eval_samples)
                if len(out) > 2:
                    return out
            except Exception:
                continue
    _, ev = seed_split(cfg.data.seed)
    return ev


__all__ = ["Pair", "load_pairs", "load_context_articles", "load_seed_pairs", "seed_split", "load_eval_pairs"]
