"""The document-translation agent runs offline: D1-D5, structure preserved, placeholders kept."""

from __future__ import annotations

from doctrans.agent.doctrans_agent import DocTransAgent
from doctrans.agent import policy
from doctrans.config import AppConfig
from doctrans.data import samples


def _agent(cfg):
    return DocTransAgent(cfg, load_model=False)


def test_agent_offline_stack(cfg):
    agent = _agent(cfg)
    assert agent.translator.name == "dictionary"
    assert agent.back_translator is None      # no reverse translator offline -> D4 round-trip skips


def test_agent_translates_markdown_all_decisions(cfg):
    agent = _agent(cfg)
    d = samples.docs()[0]
    job = agent.run(d["src"], fmt="markdown", save=False)
    sd = job.to_dict()
    assert sd["status"] in ("completed", "needs_review")
    assert {x["id"] for x in sd["decisions"]} >= {"D1", "D2", "D3", "D4", "D5"}
    assert all(t["ok"] for t in sd["trace"])
    # structure preserved + placeholders kept
    assert sd["structure"].get("struct_match") == 1.0
    assert sd["placeholder_retention"] == 1.0
    # code block + URL preserved verbatim in the output
    assert "pip install system" in sd["output"]
    assert "https://example.com/docs" in sd["output"]


def test_empty_input_fails(cfg):
    assert _agent(cfg).run("", save=False).status.value == "failed"


def test_json_structure_preserved(cfg):
    agent = _agent(cfg)
    job = agent.run('{"greeting": "Hello world", "url": "https://x.com"}', fmt="json", save=False)
    sd = job.to_dict()
    import json as _json
    obj = _json.loads(sd["output"])           # output is valid JSON
    assert obj["url"] == "https://x.com"       # url value untouched
    assert sd["structure"].get("sps") is not None


def test_translate_document_helper(cfg):
    r = _agent(cfg).translate_document(samples.docs()[1]["src"], fmt="markdown")
    assert r["output"] and r["sps"] is not None


def test_policy_gates():
    ac = AppConfig().agent
    assert policy.segment_translatable("Hello world", ac) is True
    assert policy.segment_translatable("12 34 .", ac) is False
    assert policy.translate_sane("hello", "bonjour", ac)["ok"] is True
    assert policy.translate_sane("hello", "", ac)["ok"] is False
    assert policy.verify_gate(1.0, 0.9, ac)["ok"] is True
    assert policy.verify_gate(0.5, 0.9, ac)["ok"] is False     # placeholder loss -> hard fail
    assert policy.structure_gate({"struct_match": 1.0, "markup_validity": 1.0})["ok"] is True
    assert policy.structure_gate({"struct_match": 0.5, "markup_validity": 1.0})["ok"] is False
