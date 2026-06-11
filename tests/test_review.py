"""Tests for the human-review edit/validate/retry loop with simulated input."""

import json

import pytest

from pb_review import RerunPass, review_pass


def _ok(_rubric):
    """Validator that always passes."""
    return None


def _always_fail(_rubric):
    """Validator that always raises."""
    raise ValueError("bad schema")


def _scripted_input(responses):
    """Return an input() stand-in that yields successive scripted responses."""
    iterator = iter(responses)
    return lambda _prompt="": next(iterator)


def test_review_pass_accepts_unedited_valid_draft(tmp_path):
    draft = tmp_path / "draft.json"
    rubric = {"id": "root", "sub_tasks": []}
    result = review_pass(rubric, draft, _ok, input_fn=_scripted_input([""]))
    assert result == rubric
    assert json.loads(draft.read_text())  # draft was written


def test_review_pass_picks_up_edits(tmp_path):
    draft = tmp_path / "draft.json"

    def edit_then_enter(_prompt=""):
        draft.write_text(json.dumps({"id": "edited"}))
        return ""

    result = review_pass({"id": "root"}, draft, _ok, input_fn=edit_then_enter)
    assert result == {"id": "edited"}


def test_review_pass_reedit_loop_then_success(tmp_path):
    draft = tmp_path / "draft.json"
    calls = {"n": 0}
    validators = [ValueError("nope"), None]

    def validate(_rubric):
        result = validators[min(calls["n"], len(validators) - 1)]
        calls["n"] += 1
        if isinstance(result, Exception):
            raise result

    # press Enter, choose 'e' to re-edit, press Enter again -> second validation passes
    result = review_pass({"id": "root"}, draft, validate, input_fn=_scripted_input(["", "e", ""]))
    assert result == {"id": "root"}


def test_review_pass_rerun_choice_raises(tmp_path):
    draft = tmp_path / "draft.json"
    with pytest.raises(RerunPass):
        review_pass({"id": "root"}, draft, _always_fail, input_fn=_scripted_input(["", "r"]))


def test_review_pass_handles_invalid_json_then_recovers(tmp_path):
    draft = tmp_path / "draft.json"
    state = {"n": 0}

    def maybe_break_json(_prompt=""):
        if state["n"] == 0:
            draft.write_text("{ not json")
        else:
            draft.write_text(json.dumps({"id": "fixed"}))
        state["n"] += 1
        return ""

    result = review_pass({"id": "root"}, draft, _ok, input_fn=maybe_break_json)
    assert result == {"id": "fixed"}
