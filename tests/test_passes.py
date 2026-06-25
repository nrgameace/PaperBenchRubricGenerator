"""Tests for parsing, node normalization, merging, and weight application (no network)."""

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


def test_apply_weights_sets_integers_and_defaults():
    rubric = {"id": "root", "requirements": "r", "weight": 0, "task_category": None, "finegrained_task_category": None,
              "sub_tasks": [{"id": "a", "requirements": "x", "weight": 0, "sub_tasks": [],
                             "task_category": "Code Development", "finegrained_task_category": None},
                            {"id": "b", "requirements": "y", "weight": 0, "sub_tasks": [],
                             "task_category": "Code Development", "finegrained_task_category": None}]}
    pb_passes.apply_weights(rubric, {"root": 1, "a": 3})  # "b" omitted -> default 1
    assert find_node(rubric, "root")["weight"] == 1
    assert find_node(rubric, "a")["weight"] == 3
    assert find_node(rubric, "b")["weight"] == 1
    validate_final(rubric)


def test_apply_weights_rejects_negative_and_bool():
    rubric = {"id": "root", "requirements": "r", "weight": 0, "task_category": None, "finegrained_task_category": None,
              "sub_tasks": [{"id": "a", "requirements": "x", "weight": 0, "sub_tasks": [],
                             "task_category": "Code Development", "finegrained_task_category": None}]}
    pb_passes.apply_weights(rubric, {"root": -5, "a": True})
    assert find_node(rubric, "root")["weight"] == 1
    assert find_node(rubric, "a")["weight"] == 1


class _FakeBlock:
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


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
        "claude-opus-4-8",
    )
    rubric, queue, _ = pb_passes.apply_base(parsed)
    assert rubric["requirements"] == "repro" and len(queue) == 1
    assert client.messages.calls


def test_invoke_llm_wraps_errors():
    class _BoomMessages:
        def create(self, **kwargs):
            raise RuntimeError("network down")

    class _BoomClient:
        messages = _BoomMessages()

    with pytest.raises(RuntimeError, match="Anthropic API call failed"):
        pb_passes.invoke_llm(_BoomClient(), [], [], "claude-opus-4-8")
