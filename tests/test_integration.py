"""End-to-end pipeline test with a fake model and auto-approved reviews (no network)."""

import json
import sys

import pb_review
import rubric_gen
from pb_schema import validate_final

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
    monkeypatch.setattr(rubric_gen, "build_llm", lambda *a, **k: object())
    monkeypatch.setattr(rubric_gen, "pdf_to_block", lambda _p: {"type": "file"})
    monkeypatch.setattr(rubric_gen, "load_few_shot", lambda: '{"id": "root", "requirements": "example", "weight": 1, "sub_tasks": [], "task_category": "Code Development"}')
    monkeypatch.setattr(rubric_gen, "run_base_llm", lambda *a, **k: _BASE)
    monkeypatch.setattr(rubric_gen, "run_expansion_llm", lambda *a, **k: _EXPANSION)
    monkeypatch.setattr(rubric_gen, "run_weight_llm", lambda *a, **k: {})  # all weights default to 1


def test_full_pipeline_produces_valid_final(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    _patch_llm(monkeypatch)
    _auto_approve(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["rubric_gen", str(pdf)])

    rubric_gen.main()

    final = json.loads((tmp_path / "rubric_final.json").read_text())
    validate_final(final)  # must load into the genuine TaskNode without error
    assert final["id"] == "root"
    leaf_categories = {n["task_category"] for n in _walk(final) if not n["sub_tasks"]}
    assert leaf_categories <= {"Code Development", "Code Execution", "Result Analysis"}
    assert (tmp_path / "rubric_state.json").exists()


def test_rerun_with_final_present_short_circuits(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "rubric_final.json").write_text(json.dumps({"id": "root"}))
    (tmp_path / "rubric_state.json").write_text(json.dumps({"rubric": {"id": "root"}, "queue": [], "hints": {}}))
    _patch_llm(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["rubric_gen", str(pdf), "--resume"])

    rubric_gen.main()
    assert "already exists" in capsys.readouterr().out


def _walk(node):
    """Yield every node in the tree."""
    yield node
    for child in node["sub_tasks"]:
        yield from _walk(child)
