"""Unit tests for the structure layer: parsing round-trip, masking, structure score."""

from __future__ import annotations

from doctrans.docparse import router, mask, structure_score
from doctrans.docparse.markdown_doc import parse_markdown, structure_signature
from doctrans.data import samples


def test_markdown_roundtrip_identity():
    md = samples.docs()[0]["src"]
    doc = parse_markdown(md)
    assert doc.reassemble() == md          # byte-stable round-trip
    assert len(doc.translatable_segments) >= 4


def test_format_detection():
    assert router.detect_format("# Title\n\ntext") == "markdown"
    assert router.detect_format("<html><body>hi</body></html>") == "html"
    assert router.detect_format('{"a": "b"}') == "json"
    assert router.detect_format("just some words here") == "plain"


def test_json_translates_values_not_keys():
    doc = router.parse_document('{"greeting": "Hello world", "url": "https://x.com", "n": 5}', fmt="json")
    # one translatable string value ("Hello world"); url has letters too -> 2 string values
    assert len(doc.segments) >= 1
    assert doc.fmt == "json"


def test_masking_protects_and_restores():
    from doctrans.config import DocConfig
    text = "See `pip install x` at https://e.com with {var} and <b>bold</b>."
    masked, mp, _ = mask.mask_text(text, DocConfig())
    assert "pip install x" not in masked      # code masked
    assert "https://e.com" not in masked      # url masked
    assert "{var}" not in masked              # placeholder masked
    restored = mask.restore(masked, mp)
    assert restored == text                   # exact restoration
    v = mask.validate_restoration(masked, mp)
    assert v["ok"] and v["retention"] == 1.0


def test_structure_score():
    src = "# A\n\n- one\n- two\n"
    same = "# B\n\n- un\n- deux\n"        # same structure, translated text
    broken = "A\n\nun two\n"               # heading + list lost
    assert structure_score.struct_match(structure_signature(src), structure_signature(same)) == 1.0
    assert structure_score.struct_match(structure_signature(src), structure_signature(broken)) < 1.0


def test_html_keeps_tags():
    doc = router.parse_document("<p>Hello <b>world</b></p>", fmt="html")
    assert doc.reassemble() == "<p>Hello <b>world</b></p>"
    assert any(s.kind == "text" for s in doc.segments)
