"""End-to-end pipeline test with a fake model and auto-approved reviews (no network)."""

import json
import sys
from types import SimpleNamespace

import pytest

import pb_review
import rubric_gen
from pb_schema import validate_final


class _FakeEmbeddings:
    def create(self, model, input):
        # Deterministic vectors derived from text length so results are stable across runs.
        vectors = [[float(len(text) % 7 + 1), 1.0] for text in input]
        return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in vectors])


class _FakeEmbeddingClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddings()

_BASE = {"root": {"requirements": "Reproduce the paper end to end."},
         "children": [{"requirements": "Set up environment", "expandable": True, "expansion_hint": "deps"},
                      {"requirements": "Report final metric", "expandable": False, "task_category": "Result Analysis"}]}

_EXPANSION = {"children": [{"requirements": "Implement the core method", "expandable": False, "task_category": "Code Development"},
                           {"requirements": "Run the training script", "expandable": False, "task_category": "Code Execution"}]}


def _auto_approve(monkeypatch):
    """Patch review_pass to accept the unedited draft (real validation still runs)."""
    real_review = pb_review.review_pass
    monkeypatch.setattr(rubric_gen, "review_pass",
                        lambda rubric, draft, validate_fn: real_review(rubric, draft, validate_fn, input_fn=lambda _p="": ""))


def _patch_llm(monkeypatch):
    """Patch out all network/model/asset touchpoints with deterministic fakes."""
    monkeypatch.setattr(rubric_gen, "build_client", lambda *a, **k: object())
    monkeypatch.setattr(rubric_gen, "build_embedding_client", lambda *a, **k: _FakeEmbeddingClient())
    monkeypatch.setattr(rubric_gen, "pdf_to_block", lambda _p: {"type": "file"})
    monkeypatch.setattr(rubric_gen, "load_few_shot", lambda: '{"id": "root", "requirements": "example", "weight": 1, "sub_tasks": [], "task_category": "Code Development"}')
    monkeypatch.setattr(rubric_gen, "run_base_llm", lambda *a, **k: _BASE)
    monkeypatch.setattr(rubric_gen, "run_expansion_llm", lambda *a, **k: _EXPANSION)
    def _fake_weight_llm(*args, **kwargs):
        from pb_schema import iter_nodes
        rubric = args[3]
        return {node["id"]: 1 for node in iter_nodes(rubric)}
    monkeypatch.setattr(rubric_gen, "run_weight_llm", _fake_weight_llm)

    def _fake_weight_llm_branch(*args, **kwargs):
        from pb_schema import iter_nodes
        branch_node = args[4]
        return {node["id"]: 1 for node in iter_nodes(branch_node)}
    monkeypatch.setattr(rubric_gen, "run_weight_llm_branch", _fake_weight_llm_branch)
    monkeypatch.setattr(rubric_gen, "run_split_check_llm", lambda *a, **k: {})


def _make_input_dir(tmp_path):
    """Create a minimal valid input directory with a fake PDF and MinerU folder."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "paper.pdf").write_bytes(b"%PDF-1.4 fake")
    mineru = input_dir / "mineru_out"
    mineru.mkdir()
    (mineru / "content_list.json").write_text("[]")
    return input_dir


def test_full_pipeline_produces_valid_final(tmp_path, monkeypatch):
    input_dir = _make_input_dir(tmp_path)
    output_dir = tmp_path / "output"
    _patch_llm(monkeypatch)
    _auto_approve(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["rubric_gen", "--input", str(input_dir), "--output", str(output_dir), "--review"])

    rubric_gen.main()

    final = json.loads((output_dir / "rubric_final.json").read_text())
    validate_final(final)
    assert final["id"] == "root"
    leaf_categories = {n["task_category"] for n in _walk(final) if not n["sub_tasks"]}
    assert leaf_categories <= {"Code Development", "Code Execution", "Result Analysis"}
    assert (output_dir / "rubric_state.json").exists()


def test_full_pipeline_agentic_produces_valid_final(tmp_path, monkeypatch):
    input_dir = _make_input_dir(tmp_path)
    output_dir = tmp_path / "output"
    _patch_llm(monkeypatch)
    # No --review flag: fully agentic, no review_pass calls expected
    monkeypatch.setattr(sys, "argv", ["rubric_gen", "--input", str(input_dir), "--output", str(output_dir)])

    rubric_gen.main()

    final = json.loads((output_dir / "rubric_final.json").read_text())
    validate_final(final)
    assert final["id"] == "root"


def test_rerun_with_final_present_short_circuits(tmp_path, monkeypatch, capsys):
    input_dir = _make_input_dir(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "rubric_final.json").write_text(json.dumps({"id": "root"}))
    (output_dir / "rubric_state.json").write_text(json.dumps({"rubric": {"id": "root"}, "queue": [], "hints": {}}))
    _patch_llm(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["rubric_gen", "--input", str(input_dir), "--output", str(output_dir), "--resume"])

    rubric_gen.main()
    assert "already exists" in capsys.readouterr().out


def test_main_exits_cleanly_when_weight_resolution_exceeds_max_retries(tmp_path, monkeypatch):
    input_dir = _make_input_dir(tmp_path)
    output_dir = tmp_path / "output"
    _patch_llm(monkeypatch)
    # Every weight-assigning call returns nothing, so every node stays invalid no matter how
    # many times the pipeline retries.
    monkeypatch.setattr(rubric_gen, "run_weight_llm_branch", lambda *a, **k: {})
    monkeypatch.setattr(rubric_gen, "run_weight_llm", lambda *a, **k: {})
    monkeypatch.setattr(sys, "argv", ["rubric_gen", "--input", str(input_dir), "--output", str(output_dir)])

    with pytest.raises(SystemExit) as exc_info:
        rubric_gen.main()

    assert "--resume" in str(exc_info.value)


def _walk(node):
    yield node
    for child in node["sub_tasks"]:
        yield from _walk(child)
