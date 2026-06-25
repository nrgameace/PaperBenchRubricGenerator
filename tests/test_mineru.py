"""Tests for MinerU content conversion and section slicing."""

import pytest

import pb_mineru

_CONTENT = [
    {"type": "text", "text": "Abstract", "text_level": 1},
    {"type": "text", "text": "We propose a method."},
    {"type": "text", "text": "1 Introduction", "text_level": 1},
    {"type": "text", "text": "Prior work..."},
    {"type": "image", "img_caption": "Figure 1: Architecture"},
    {"type": "text", "text": "2 Method", "text_level": 1},
    {"type": "text", "text": "Our method works by..."},
    {"type": "table", "table_body": "| A | B |\n|---|---|\n| 1 | 2 |", "table_caption": "Table 1"},
    {"type": "text", "text": "2.1 Sub-section", "text_level": 2},
    {"type": "text", "text": "Details here."},
    {"type": "text", "text": "3 Experiments", "text_level": 1},
    {"type": "text", "text": "We ran experiments."},
]


def test_blocks_to_text_headings_get_hashes():
    result = pb_mineru.blocks_to_text([{"type": "text", "text": "Introduction", "text_level": 1}])
    assert result == "# Introduction"


def test_blocks_to_text_plain_text_no_hash():
    result = pb_mineru.blocks_to_text([{"type": "text", "text": "Some paragraph."}])
    assert result == "Some paragraph."


def test_blocks_to_text_image_outputs_figure_label():
    result = pb_mineru.blocks_to_text([{"type": "image", "img_caption": "Overview"}])
    assert "[Figure: Overview]" in result


def test_blocks_to_text_image_no_caption():
    result = pb_mineru.blocks_to_text([{"type": "image"}])
    assert "[Figure]" in result


def test_blocks_to_text_table_includes_caption_and_body():
    block = {"type": "table", "table_body": "| A |\n|---|", "table_caption": "Results"}
    result = pb_mineru.blocks_to_text([block])
    assert "Results" in result and "| A |" in result


def test_blocks_to_text_unknown_type_skipped():
    result = pb_mineru.blocks_to_text([{"type": "footnote", "text": "ignored"}])
    assert result == ""


def test_slice_section_returns_matching_section():
    blocks = pb_mineru.slice_section(_CONTENT, "Method")
    texts = [b["text"] for b in blocks if b.get("text")]
    assert "2 Method" in texts
    assert "Our method works by..." in texts
    assert "Details here." in texts
    assert "3 Experiments" not in texts


def test_slice_section_stops_at_next_sibling_heading():
    blocks = pb_mineru.slice_section(_CONTENT, "Introduction")
    texts = [b["text"] for b in blocks if b.get("text")]
    assert "Prior work..." in texts
    assert "2 Method" not in texts


def test_slice_section_falls_back_to_full_list_when_no_match():
    tiny = [{"type": "text", "text": "only thing here"}]
    result = pb_mineru.slice_section(tiny, "nonexistent section xyz")
    assert result == tiny


def test_slice_section_case_insensitive():
    blocks = pb_mineru.slice_section(_CONTENT, "experiments")
    texts = [b["text"] for b in blocks if b.get("text")]
    assert "3 Experiments" in texts
