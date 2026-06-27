"""Evaluate the MT core (chrF/BLEU vs baselines) + the DOCUMENT pipeline (chrF + SPS).

Two layers:
* **MT-level:** model (fine-tuned transformer / dictionary offline) vs the dictionary
  baseline vs an identity copy-source floor, on held-out sentence pairs. chrF is the headline.
* **Document-level:** run the full agent on the seed structured documents and report the
  whole-document chrF vs gold + the mean **Structure-Preservation Score (SPS)** +
  placeholder-retention — the structure story (a translation that breaks a table fails).
Runs fully offline (dictionary MT + line/regex parsers + numpy metrics).
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp
from ..data.dataset import load_eval_pairs, seed_split
from ..data import samples
from ..mt.translator import DictionaryTranslator, load_translator
from . import metrics as M

logger = get_logger(__name__)


class _Identity:
    name = "identity"
    version = "id-1.0"

    def translate(self, text: str) -> str:
        return text

    def translate_batch(self, texts):
        return list(texts)


def evaluate(cfg: AppConfig, limit: Optional[int] = None, save: bool = True, load_model: bool = True) -> Dict:
    pairs = load_eval_pairs(cfg, limit=limit or cfg.data.max_eval_samples)
    if len(pairs) <= 2:
        _, pairs = seed_split(cfg.data.seed)
    src = [p.src for p in pairs]
    refs = [p.tgt for p in pairs]

    model = load_translator(cfg.mt, prefer="transformer" if load_model else "dictionary")
    result: Dict = {"n_eval": len(pairs), "pair": f"{cfg.mt.src_lang}-{cfg.mt.tgt_lang}"}
    result["model_name"] = getattr(model, "name", "?")
    result["model"] = M.translation_metrics(model.translate_batch(src), refs)
    result["dictionary"] = M.translation_metrics(DictionaryTranslator().translate_batch(src), refs)
    result["identity"] = M.translation_metrics(_Identity().translate_batch(src), refs)

    # ---- document-level: run the agent on the seed structured docs ----
    try:
        from ..agent.doctrans_agent import DocTransAgent
        agent = DocTransAgent(cfg, load_model=load_model, translator=model)
        doc_chrfs, sps_list, phr_list, demos = [], [], [], []
        for d in samples.docs():
            job = agent.run(d["src"], fmt=d.get("fmt", ""), save=False)
            sd = job.to_dict()
            dc = M.chrf([sd["output"]], [d["tgt"]])
            doc_chrfs.append(dc)
            sps_list.append(sd["structure"].get("sps") or 0.0)
            phr_list.append(sd["placeholder_retention"] if sd["placeholder_retention"] is not None else 1.0)
            demos.append({"id": d["id"], "chrf": round(dc, 2), "sps": sd["structure"].get("sps"),
                          "status": sd["status"], "needs_review": sd["needs_review"]})
        result["document"] = {
            "n_docs": len(doc_chrfs),
            "doc_chrf": round(sum(doc_chrfs) / max(1, len(doc_chrfs)), 2),
            "doc_sps": round(sum(sps_list) / max(1, len(sps_list)), 4),
            "doc_phr": round(sum(phr_list) / max(1, len(phr_list)), 4),
            "demos": demos,
        }
    except Exception as exc:
        logger.info("document eval skipped (%s)", exc)
        result["document"] = {"error": str(exc)}

    m, b, idn = result["model"], result["dictionary"], result["identity"]
    doc = result.get("document", {})
    result["summary"] = {
        "model_name": result["model_name"],
        "model_chrf": m["chrf"], "model_bleu": m["bleu"],
        "dictionary_chrf": b["chrf"], "identity_chrf": idn["chrf"],
        "beats_identity": m["chrf"] >= idn["chrf"],
        "beats_dictionary": m["chrf"] >= b["chrf"],
        "doc_chrf": doc.get("doc_chrf"), "doc_sps": doc.get("doc_sps"), "doc_phr": doc.get("doc_phr"),
    }
    if save:
        out = run_dir() / "eval"
        out.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(result, indent=2, ensure_ascii=False)
        (out / f"eval-{utc_stamp()}.json").write_text(payload, encoding="utf-8")
        (out / "latest.json").write_text(payload, encoding="utf-8")
        logger.info("Eval: %s chrF=%s vs dict=%s vs identity=%s | doc chrF=%s SPS=%s PHR=%s",
                    result["model_name"], m["chrf"], b["chrf"], idn["chrf"],
                    doc.get("doc_chrf"), doc.get("doc_sps"), doc.get("doc_phr"))
    return result


__all__ = ["evaluate"]
