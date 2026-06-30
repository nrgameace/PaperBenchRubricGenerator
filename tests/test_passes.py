"""Tests for parsing, node normalization, merging, and weight application (no network)."""

import copy
from types import SimpleNamespace

import pytest

import pb_passes
from pb_schema import find_node, validate_final, validate_partial


def test_parse_json_plain():
    assert pb_passes.parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_with_code_fence():
    assert pb_passes.parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_with_surrounding_prose():
    assert pb_passes.parse_json_response('Here you go:\n{"a": [1, 2]}\nDone.') == {"a": [1, 2]}


def test_parse_json_invalid_raises():
    with pytest.raises((ValueError,)):
        pb_passes.parse_json_response("not json at all")


def test_clean_id_normalizes_to_kebab_case():
    assert pb_passes.clean_id("Vim Block Forward SSM") == "vim-block-forward-ssm"
    assert pb_passes.clean_id("already-kebab-1") == "already-kebab-1"
    assert pb_passes.clean_id("") == "node"
    assert pb_passes.clean_id("@@@") == "node"


def test_slugify_fallback():
    assert pb_passes.slugify("Set up the Environment!") == "set-up-the-environment"
    assert pb_passes.slugify("") == "node"
    assert pb_passes.slugify("one two three four five six seven", max_words=3) == "one-two-three"


def test_ensure_unique_id_preserves_clean_id_then_suffixes():
    used = set()
    assert pb_passes.ensure_unique_id("vim-block-forward-ssm", used) == "vim-block-forward-ssm"
    assert pb_passes.ensure_unique_id("vim-block-forward-ssm", used) == "vim-block-forward-ssm-2"
    assert used == {"vim-block-forward-ssm", "vim-block-forward-ssm-2"}


def test_choose_id_prefers_model_id_over_requirements():
    used = set()
    assert pb_passes.choose_id({"id": "Vim Block Forward SSM", "requirements": "something else"}, used) == "vim-block-forward-ssm"


def test_choose_id_falls_back_to_requirements_slug_when_id_missing():
    used = set()
    assert pb_passes.choose_id({"requirements": "Set up environment"}, used) == "set-up-environment"


def test_normalize_child_uses_model_id():
    used = set()
    node, hint = pb_passes.normalize_child(
        {"id": "build-vim-block", "requirements": "  Build the Vim block  ", "expandable": True, "expansion_hint": "covers x"}, used)
    assert node["requirements"] == "Build the Vim block"
    assert node["id"] == "build-vim-block"
    assert node["sub_tasks"] == [] and node["weight"] == 0 and node["task_category"] is None
    assert hint == "covers x"


def test_normalize_child_leaf():
    node, hint = pb_passes.normalize_child({"requirements": "do x", "expandable": False, "task_category": "Code Execution"}, set())
    assert hint is None
    assert node["task_category"] == "Code Execution"


def test_normalize_child_expandable_without_hint_gets_default():
    _, hint = pb_passes.normalize_child({"requirements": "area", "expandable": True}, set())
    assert hint and isinstance(hint, str)


def test_apply_base_uses_model_ids():
    parsed = {"root": {"requirements": "reproduce paper"},
              "children": [{"id": "env-setup", "requirements": "setup", "expandable": True, "expansion_hint": "env"},
                           {"id": "final-metric", "requirements": "atomic", "expandable": False, "task_category": "Code Development"}]}
    rubric, queue, hints = pb_passes.apply_base(parsed)
    assert rubric["id"] == "root" and rubric["requirements"] == "reproduce paper"
    assert [child["id"] for child in rubric["sub_tasks"]] == ["env-setup", "final-metric"]
    assert queue == ["env-setup"] and "env-setup" in hints
    validate_partial(rubric, queue)


def test_apply_base_dedupes_colliding_model_ids():
    parsed = {"root": {"requirements": "r"},
              "children": [{"id": "build-model", "requirements": "a", "expandable": False, "task_category": "Code Development"},
                           {"id": "build-model", "requirements": "b", "expandable": False, "task_category": "Code Development"}]}
    rubric, _, _ = pb_passes.apply_base(parsed)
    assert [child["id"] for child in rubric["sub_tasks"]] == ["build-model", "build-model-2"]


def test_apply_expansion_attaches_children_and_returns_pending():
    rubric, queue, hints = pb_passes.apply_base(
        {"root": {"requirements": "r"}, "children": [{"requirements": "setup", "expandable": True, "expansion_hint": "env"}]})
    node_id = queue[0]
    parsed = {"children": [{"requirements": "install deps", "expandable": False, "task_category": "Code Development"},
                           {"requirements": "subarea", "expandable": True, "expansion_hint": "more"}]}
    new_pending = pb_passes.apply_expansion(rubric, node_id, parsed, hints)
    target = find_node(rubric, node_id)
    assert len(target["sub_tasks"]) == 2 and target["task_category"] is None
    assert len(new_pending) == 1 and node_id not in hints
    validate_partial(rubric, new_pending)


def test_apply_expansion_accepts_bare_list():
    rubric, queue, hints = pb_passes.apply_base(
        {"root": {"requirements": "r"}, "children": [{"requirements": "setup", "expandable": True, "expansion_hint": "env"}]})
    node_id = queue[0]
    pending = pb_passes.apply_expansion(rubric, node_id, [{"requirements": "x", "expandable": False, "task_category": "Code Development"}], hints)
    assert pending == []
    assert len(find_node(rubric, node_id)["sub_tasks"]) == 1


def test_apply_expansion_missing_node_raises():
    with pytest.raises(ValueError):
        pb_passes.apply_expansion({"id": "root", "sub_tasks": []}, "ghost", {"children": []}, {})


def _two_leaf_rubric():
    return {
        "id": "root", "requirements": "r", "weight": 0, "task_category": None,
        "finegrained_task_category": None,
        "sub_tasks": [
            {"id": "a", "requirements": "do a", "weight": 0, "sub_tasks": [],
             "task_category": "Code Development", "finegrained_task_category": None},
            {"id": "b", "requirements": "do b", "weight": 0, "sub_tasks": [],
             "task_category": "Code Development", "finegrained_task_category": None},
        ],
    }


def test_apply_weights_sets_valid_weights():
    rubric = _two_leaf_rubric()
    pb_passes.apply_weights(rubric, {"root": 1, "a": 3, "b": 2})
    assert find_node(rubric, "root")["weight"] == 1
    assert find_node(rubric, "a")["weight"] == 3
    assert find_node(rubric, "b")["weight"] == 2
    validate_final(rubric)


def test_apply_weights_raises_on_negative_weight():
    with pytest.raises(ValueError, match="invalid weight"):
        pb_passes.apply_weights(_two_leaf_rubric(), {"root": -5, "a": 1, "b": 1})


def test_apply_weights_raises_on_missing_weight():
    with pytest.raises(ValueError, match="invalid weight"):
        pb_passes.apply_weights(_two_leaf_rubric(), {"root": 1, "a": 3})


def test_apply_weights_raises_on_boolean_weight():
    with pytest.raises(ValueError, match="invalid weight"):
        pb_passes.apply_weights(_two_leaf_rubric(), {"root": True, "a": 1, "b": 1})


# ── find_invalid_weights tests ────────────────────────────────────────────────

def test_find_invalid_weights_empty_when_all_valid():
    assert pb_passes.find_invalid_weights(_two_leaf_rubric(), {"root": 1, "a": 3, "b": 2}) == []


def test_find_invalid_weights_detects_missing():
    invalid = pb_passes.find_invalid_weights(_two_leaf_rubric(), {"root": 1, "a": 3})
    assert [(nid, raw) for nid, _, raw in invalid] == [("b", None)]


def test_find_invalid_weights_detects_negative():
    invalid = pb_passes.find_invalid_weights(_two_leaf_rubric(), {"root": 1, "a": -3, "b": 2})
    assert [nid for nid, _, _ in invalid] == ["a"]
    assert invalid[0][2] == -3


def test_find_invalid_weights_detects_boolean():
    invalid = pb_passes.find_invalid_weights(_two_leaf_rubric(), {"root": 1, "a": True, "b": 2})
    assert [nid for nid, _, _ in invalid] == ["a"]


def test_find_invalid_weights_detects_non_numeric_string():
    invalid = pb_passes.find_invalid_weights(_two_leaf_rubric(), {"root": 1, "a": "heavy", "b": 2})
    assert [nid for nid, _, _ in invalid] == ["a"]


def test_find_invalid_weights_does_not_mutate_rubric():
    rubric = _two_leaf_rubric()
    before = copy.deepcopy(rubric)
    pb_passes.find_invalid_weights(rubric, {})
    assert rubric == before


def test_find_invalid_weights_includes_requirements_in_tuple():
    invalid = pb_passes.find_invalid_weights(_two_leaf_rubric(), {"root": 1, "a": 3})
    assert invalid[0][1] == "do b"


# ── run_weight_llm new param tests ────────────────────────────────────────────

def _weight_instruction(client):
    """Extract the instruction text from the first run_weight_llm call."""
    return client.messages.calls[0]["messages"][0]["content"][0]["text"]


def test_run_weight_llm_targeted_node_ids_narrows_instruction():
    client = _FakeClient('{"weights": {"a": 3}}')
    pb_passes.run_weight_llm(client, [], "text", _two_leaf_rubric(), "model", node_ids=["a"])
    instruction = _weight_instruction(client)
    assert "a" in instruction
    assert "EVERY node" not in instruction


def test_run_weight_llm_feedback_appended():
    client = _FakeClient('{"weights": {"root": 1, "a": 3, "b": 2}}')
    pb_passes.run_weight_llm(client, [], "text", _two_leaf_rubric(), "model", feedback="recheck table 3")
    instruction = _weight_instruction(client)
    assert "USER FEEDBACK" in instruction
    assert "recheck table 3" in instruction


def test_run_weight_llm_no_feedback_block_when_none():
    client = _FakeClient('{"weights": {"root": 1, "a": 3, "b": 2}}')
    pb_passes.run_weight_llm(client, [], "text", _two_leaf_rubric(), "model")
    assert "USER FEEDBACK" not in _weight_instruction(client)


class _FakeBlock:
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]
        self.usage = SimpleNamespace(
            input_tokens=10, output_tokens=5,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )


class _FakeMessages:
    def __init__(self, content):
        self._content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._content)


class _FakeClient:
    def __init__(self, content):
        self.messages = _FakeMessages(content)


def test_run_base_llm_with_fake_client():
    client = _FakeClient(
        '```json\n{"root": {"requirements": "repro"}, '
        '"children": [{"requirements": "s", "expandable": true, "expansion_hint": "h"}]}\n```'
    )
    parsed = pb_passes.run_base_llm(
        client,
        [{"type": "text", "text": "sys"}],
        {"type": "document"},
        "This is the structured paper text.",
        "claude-opus-4-8",
    )
    rubric, queue, _ = pb_passes.apply_base(parsed)
    assert rubric["requirements"] == "repro" and len(queue) == 1
    assert client.messages.calls


def _target_rubric():
    return {
        "id": "root", "requirements": "r", "weight": 0, "task_category": None,
        "finegrained_task_category": None,
        "sub_tasks": [{"id": "target", "requirements": "expand me", "weight": 0,
                       "sub_tasks": [], "task_category": None, "finegrained_task_category": None}]
    }


def test_run_expansion_llm_appends_feedback_when_provided():
    client = _FakeClient('{"children": []}')
    pb_passes.run_expansion_llm(
        client,
        [{"type": "text", "text": "sys"}],
        "section text",
        _target_rubric(),
        "target",
        "expansion hint",
        "claude-sonnet-4-6",
        feedback="check table 3",
    )
    instruction = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert "USER FEEDBACK" in instruction
    assert "check table 3" in instruction


def test_run_expansion_llm_no_feedback_block_when_empty():
    client = _FakeClient('{"children": []}')
    pb_passes.run_expansion_llm(
        client,
        [{"type": "text", "text": "sys"}],
        "section text",
        _target_rubric(),
        "target",
        "expansion hint",
        "claude-sonnet-4-6",
        feedback="",
    )
    instruction = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert "USER FEEDBACK" not in instruction


def test_invoke_llm_records_usage_when_tracker_provided():
    from pb_cost import CostTracker
    tracker = CostTracker()
    client = _FakeClient('{"a": 1}')
    pb_passes.invoke_llm(client, [], [{"role": "user", "content": []}], "claude-opus-4-8", tracker=tracker)
    assert tracker.totals_for("claude-opus-4-8")["input"] == 10
    assert tracker.totals_for("claude-opus-4-8")["output"] == 5


def test_invoke_llm_no_tracker_still_returns_text():
    client = _FakeClient("hello")
    result = pb_passes.invoke_llm(client, [], [{"role": "user", "content": []}], "claude-opus-4-8")
    assert result == "hello"


def test_invoke_llm_wraps_errors():
    class _BoomMessages:
        def create(self, **kwargs):
            raise RuntimeError("network down")

    class _BoomClient:
        messages = _BoomMessages()

    with pytest.raises(RuntimeError, match="Anthropic API call failed"):
        pb_passes.invoke_llm(_BoomClient(), [], [], "claude-opus-4-8")
