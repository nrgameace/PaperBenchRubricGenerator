"""Tests for input discovery and loading helpers."""

import json
from pathlib import Path

import pytest

import pb_input


def test_discover_pdf_finds_pdf(tmp_path):
    (tmp_path / "paper.pdf").write_bytes(b"%PDF")
    assert pb_input.discover_pdf(tmp_path).name == "paper.pdf"


def test_discover_pdf_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="No PDF"):
        pb_input.discover_pdf(tmp_path)


def test_discover_mineru_dir_finds_subfolder(tmp_path):
    mineru = tmp_path / "mineru_out"
    mineru.mkdir()
    (mineru / "content_list.json").write_text("[]")
    assert pb_input.discover_mineru_dir(tmp_path) == mineru


def test_discover_mineru_dir_raises_when_missing(tmp_path):
    (tmp_path / "not_mineru").mkdir()
    with pytest.raises(FileNotFoundError, match="No MinerU"):
        pb_input.discover_mineru_dir(tmp_path)


def test_load_content_list_parses_json(tmp_path):
    data = [{"type": "text", "text": "hello"}]
    (tmp_path / "content_list.json").write_text(json.dumps(data))
    assert pb_input.load_content_list(tmp_path) == data


def test_load_content_list_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        pb_input.load_content_list(tmp_path)
