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


def test_parse_json_invalid_error_includes_raw_response_preview():
    with pytest.raises(ValueError, match="the model went off the rails"):
        pb_passes.parse_json_response("the model went off the rails and said something weird")


def test_parse_json_malformed_after_extraction_includes_raw_response_preview():
    with pytest.raises(ValueError, match=r"nothing parses here"):
        pb_passes.parse_json_response('prose with a { that never closes and nothing parses here')


def test_parse_json_ignores_stray_brackets_in_reasoning_prose_before_real_json():
    """Regression test: a model that writes a reasoning scratchpad (citing ranges like
    "[0.4, 0.6]") before its actual JSON answer must not have that stray array mistaken
    for the payload — this is the exact failure pattern seen in production split-check
    responses that included prose analysis ahead of the JSON."""
    text = (
        'Looking at the alpha ablation, values in the range [0.4, 0.6] preserve accuracy.\n\n'
        '{"splits": {}, "duplicates": {"leaf-a": "leaf-b"}}'
    )
    assert pb_passes.parse_json_response(text) == {"splits": {}, "duplicates": {"leaf-a": "leaf-b"}}


def test_parse_json_prefers_outer_object_over_its_own_nested_sub_objects():
    text = '{"splits": {"leaf-a": [{"id": "x", "requirements": "r"}]}, "duplicates": {}}'
    parsed = pb_passes.parse_json_response(text)
    assert set(parsed.keys()) == {"splits", "duplicates"}
    assert parsed["splits"]["leaf-a"][0]["id"] == "x"


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


_ENV_SETUP_2_RAW = {
    "id": "env-setup-2",
    "requirements": (
        "All 10 environments are configured and runnable: ant_big_maze, ant_hardest_maze, "
        "arm_binpick_hard, arm_push_easy, arm_push_hard, humanoid, humanoid_big_maze, "
        "humanoid_u_maze, ant_u4_maze, ant_u5_maze."
    ),
    "expandable": False,
    "task_category": "Code Execution",
}


def test_normalize_child_overrides_dense_leaf_that_enumerates_items():
    node, hint = pb_passes.normalize_child(_ENV_SETUP_2_RAW, set())
    assert hint is not None
    assert node["task_category"] is None
    assert node["sub_tasks"] == []
    assert node["finegrained_task_category"] is None


def test_normalize_child_override_hint_mentions_detected_count():
    _, hint = pb_passes.normalize_child(_ENV_SETUP_2_RAW, set())
    assert "10" in hint
    assert "children" in hint


def test_normalize_child_leaf_stays_leaf_below_enumeration_threshold():
    node, hint = pb_passes.normalize_child({"requirements": "do x", "expandable": False, "task_category": "Code Execution"}, set())
    assert hint is None
    assert node["task_category"] == "Code Execution"


def test_normalize_child_expandable_flag_takes_precedence_over_enumeration_check():
    raw = dict(_ENV_SETUP_2_RAW, expandable=True, expansion_hint="model's own hint")
    _, hint = pb_passes.normalize_child(raw, set())
    assert hint == "model's own hint"


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


def test_apply_expansion_queues_forced_expandable_child_from_enumeration_override():
    rubric, queue, hints = pb_passes.apply_base(
        {"root": {"requirements": "r"}, "children": [{"requirements": "setup", "expandable": True, "expansion_hint": "env"}]})
    node_id = queue[0]
    parsed = {"children": [_ENV_SETUP_2_RAW]}
    new_pending = pb_passes.apply_expansion(rubric, node_id, parsed, hints)
    assert "env-setup-2" in new_pending
    assert "env-setup-2" in hints
    assert find_node(rubric, "env-setup-2")["task_category"] is None
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


def test_apply_expansion_within_max_depth_is_unaffected():
    rubric, queue, hints = pb_passes.apply_base(
        {"root": {"requirements": "r"}, "children": [{"requirements": "setup", "expandable": True, "expansion_hint": "env"}]})
    node_id = queue[0]  # depth 1
    parsed = {"children": [{"requirements": "subarea", "expandable": True, "expansion_hint": "more"}]}
    errors = []
    new_pending = pb_passes.apply_expansion(rubric, node_id, parsed, hints, errors=errors, max_depth=2)
    assert len(new_pending) == 1
    assert errors == []


def test_apply_expansion_forces_leaf_and_records_error_past_max_depth():
    rubric, queue, hints = pb_passes.apply_base(
        {"root": {"requirements": "r"}, "children": [{"requirements": "setup", "expandable": True, "expansion_hint": "env"}]})
    node_id = queue[0]  # depth 1; children would be depth 2
    parsed = {"children": [{"id": "too-deep", "requirements": "subarea", "expandable": True, "expansion_hint": "more"}]}
    errors = []
    new_pending = pb_passes.apply_expansion(rubric, node_id, parsed, hints, errors=errors, max_depth=1)
    target = find_node(rubric, node_id)
    child = target["sub_tasks"][0]
    assert child["id"] == "too-deep"
    assert child["task_category"] is not None
    assert child["sub_tasks"] == []
    assert new_pending == []
    assert "too-deep" not in hints
    assert len(errors) == 1
    assert "too-deep" in errors[0]
    assert "maximum depth of 1" in errors[0]
    validate_final(rubric)


def test_apply_expansion_depth_guardrail_is_noop_without_errors_list():
    rubric, queue, hints = pb_passes.apply_base(
        {"root": {"requirements": "r"}, "children": [{"requirements": "setup", "expandable": True, "expansion_hint": "env"}]})
    node_id = queue[0]
    parsed = {"children": [{"id": "too-deep", "requirements": "subarea", "expandable": True, "expansion_hint": "more"}]}
    new_pending = pb_passes.apply_expansion(rubric, node_id, parsed, hints, max_depth=1)
    assert new_pending == []
    validate_final(rubric)


# ── branch_size / force_branch_cap_leaves tests ──────────────────────────────

def _nested_branch_rubric():
    return {
        "id": "root", "requirements": "r", "weight": 0, "task_category": None,
        "finegrained_task_category": None,
        "sub_tasks": [
            {
                "id": "branch-a", "requirements": "branch a", "weight": 0,
                "task_category": None, "finegrained_task_category": None,
                "sub_tasks": [
                    {"id": "child-1", "requirements": "c1", "weight": 0, "sub_tasks": [],
                     "task_category": None, "finegrained_task_category": None},
                    {"id": "child-2", "requirements": "c2", "weight": 0,
                     "task_category": None, "finegrained_task_category": None,
                     "sub_tasks": [
                         {"id": "grandchild-1", "requirements": "g1", "weight": 0, "sub_tasks": [],
                          "task_category": None, "finegrained_task_category": None},
                     ]},
                ],
            },
            {"id": "branch-b", "requirements": "branch b", "weight": 0, "sub_tasks": [],
             "task_category": "Code Development", "finegrained_task_category": None},
        ],
    }


def test_branch_size_counts_full_subtree():
    rubric = _nested_branch_rubric()
    assert pb_passes.branch_size(rubric, "branch-a") == 4  # branch-a, child-1, child-2, grandchild-1


def test_branch_size_returns_1_for_leaf():
    rubric = _nested_branch_rubric()
    assert pb_passes.branch_size(rubric, "branch-b") == 1


def test_branch_size_raises_for_missing_branch():
    rubric = _nested_branch_rubric()
    with pytest.raises(ValueError):
        pb_passes.branch_size(rubric, "ghost")


def test_force_branch_cap_leaves_forces_all_ids_to_fallback_category():
    rubric = _nested_branch_rubric()
    hints = {"child-1": "hint1", "grandchild-1": "hint-g1"}
    pb_passes.force_branch_cap_leaves(rubric, ["child-1", "grandchild-1"], "branch-a", hints)
    assert find_node(rubric, "child-1")["task_category"] == pb_passes._DEPTH_FALLBACK_CATEGORY
    assert find_node(rubric, "grandchild-1")["task_category"] == pb_passes._DEPTH_FALLBACK_CATEGORY


def test_force_branch_cap_leaves_pops_hints_for_forced_nodes():
    rubric = _nested_branch_rubric()
    hints = {"child-1": "hint1", "grandchild-1": "hint-g1"}
    pb_passes.force_branch_cap_leaves(rubric, ["child-1", "grandchild-1"], "branch-a", hints)
    assert hints == {}


def test_force_branch_cap_leaves_logs_one_error_line_per_node_with_branch_id():
    rubric = _nested_branch_rubric()
    hints = {"child-1": "hint1", "grandchild-1": "hint-g1"}
    errors = []
    pb_passes.force_branch_cap_leaves(rubric, ["child-1", "grandchild-1"], "branch-a", hints, errors=errors)
    assert len(errors) == 2
    assert errors[0] == f"child-1: Branch 'branch-a' hit the {pb_passes.MAX_BRANCH_NODES}-node cap; forced to leaf."
    assert errors[1] == f"grandchild-1: Branch 'branch-a' hit the {pb_passes.MAX_BRANCH_NODES}-node cap; forced to leaf."


def test_force_branch_cap_leaves_no_error_mutation_when_errors_none():
    rubric = _nested_branch_rubric()
    hints = {"child-1": "hint1"}
    pb_passes.force_branch_cap_leaves(rubric, ["child-1"], "branch-a", hints)  # should not raise


def test_force_branch_cap_leaves_adds_branch_id_to_capped_branches_set():
    rubric = _nested_branch_rubric()
    hints = {"child-1": "hint1", "grandchild-1": "hint-g1"}
    capped = set()
    pb_passes.force_branch_cap_leaves(rubric, ["child-1", "grandchild-1"], "branch-a", hints, capped_branches=capped)
    assert capped == {"branch-a"}


def test_force_branch_cap_leaves_no_capped_branches_mutation_when_none():
    rubric = _nested_branch_rubric()
    hints = {"child-1": "hint1"}
    pb_passes.force_branch_cap_leaves(rubric, ["child-1"], "branch-a", hints)  # should not raise


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


# ── run_weight_llm_branch tests ──────────────────────────────────────────────

def _branch_rubric():
    return {
        "id": "root", "requirements": "r", "weight": 0, "task_category": None,
        "finegrained_task_category": None,
        "sub_tasks": [
            {
                "id": "section-a", "requirements": "section a", "weight": 0,
                "task_category": None, "finegrained_task_category": None,
                "sub_tasks": [
                    {"id": "leaf-a1", "requirements": "do a1", "weight": 0, "sub_tasks": [],
                     "task_category": "Code Development", "finegrained_task_category": None},
                    {"id": "leaf-a2", "requirements": "do a2", "weight": 0, "sub_tasks": [],
                     "task_category": "Code Execution", "finegrained_task_category": None},
                ],
            },
            {
                "id": "section-b", "requirements": "section b", "weight": 0,
                "task_category": None, "finegrained_task_category": None,
                "sub_tasks": [
                    {"id": "leaf-b1", "requirements": "do b1", "weight": 0, "sub_tasks": [],
                     "task_category": "Code Development", "finegrained_task_category": None},
                ],
            },
        ],
    }


def test_run_weight_llm_branch_targets_subtree_ids():
    rubric = _branch_rubric()
    branch_node = rubric["sub_tasks"][0]
    client = _FakeClient('{"weights": {"section-a": 3, "leaf-a1": 2, "leaf-a2": 1}}')
    pb_passes.run_weight_llm_branch(client, [], "text", rubric, branch_node, "model")
    instruction = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert "section-a" in instruction
    assert "leaf-a1" in instruction
    assert "leaf-a2" in instruction
    assert "EVERY node" not in instruction


def test_run_weight_llm_branch_includes_full_rubric_for_context():
    rubric = _branch_rubric()
    branch_node = rubric["sub_tasks"][0]
    client = _FakeClient('{"weights": {"section-a": 3, "leaf-a1": 2, "leaf-a2": 1}}')
    pb_passes.run_weight_llm_branch(client, [], "text", rubric, branch_node, "model")
    instruction = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert "section-b" in instruction


def test_run_weight_llm_branch_returns_partial_weights():
    rubric = _branch_rubric()
    branch_node = rubric["sub_tasks"][0]
    client = _FakeClient('{"weights": {"section-a": 3, "leaf-a1": 2, "leaf-a2": 1}}')
    result = pb_passes.run_weight_llm_branch(client, [], "text", rubric, branch_node, "model")
    assert result == {"section-a": 3, "leaf-a1": 2, "leaf-a2": 1}


# ── run_weight_llm_global tests ───────────────────────────────────────────────

def test_run_weight_llm_global_sends_current_weights_as_context():
    rubric = _two_leaf_rubric()
    current_weights = {"root": 1, "a": 3, "b": 2}
    client = _FakeClient('{"weights": {"root": 1, "a": 4, "b": 2}}')
    pb_passes.run_weight_llm_global(client, [], "text", rubric, current_weights, "model")
    instruction = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert '"a": 3' in instruction or "locally" in instruction.lower() or "calibrate" in instruction.lower()


def test_run_weight_llm_global_with_feedback_appended():
    rubric = _two_leaf_rubric()
    client = _FakeClient('{"weights": {"root": 1, "a": 3, "b": 2}}')
    pb_passes.run_weight_llm_global(client, [], "text", rubric, {}, "model", feedback="recheck table 3")
    instruction = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert "USER FEEDBACK" in instruction
    assert "recheck table 3" in instruction


def test_run_weight_llm_global_without_feedback_no_feedback_block():
    rubric = _two_leaf_rubric()
    client = _FakeClient('{"weights": {"root": 1, "a": 3, "b": 2}}')
    pb_passes.run_weight_llm_global(client, [], "text", rubric, {}, "model")
    instruction = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert "USER FEEDBACK" not in instruction


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
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_FakeBlock(text)]
        self.stop_reason = stop_reason
        self.usage = SimpleNamespace(
            input_tokens=10, output_tokens=5,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )


class _FakeStreamManager:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get_final_message(self):
        return self._response


class _FakeMessages:
    def __init__(self, content, stop_reason="end_turn"):
        self._content = content
        self._stop_reason = stop_reason
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeStreamManager(_FakeResponse(self._content, stop_reason=self._stop_reason))


class _FakeClient:
    def __init__(self, content, stop_reason="end_turn"):
        self.messages = _FakeMessages(content, stop_reason=stop_reason)


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


def _dense_target_rubric():
    return {
        "id": "root", "requirements": "r", "weight": 0, "task_category": None,
        "finegrained_task_category": None,
        "sub_tasks": [{"id": "target", "requirements": _ENV_SETUP_2_RAW["requirements"], "weight": 0,
                       "sub_tasks": [], "task_category": None, "finegrained_task_category": None}]
    }


def test_run_expansion_llm_injects_enumeration_guardrail_for_dense_target():
    client = _FakeClient('{"children": []}')
    pb_passes.run_expansion_llm(
        client,
        [{"type": "text", "text": "sys"}],
        "section text",
        _dense_target_rubric(),
        "target",
        "expansion hint",
        "claude-sonnet-4-6",
    )
    instruction = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert "ENUMERATION GUARDRAIL" in instruction
    assert "10" in instruction


def test_run_expansion_llm_omits_enumeration_guardrail_for_single_item_target():
    client = _FakeClient('{"children": []}')
    pb_passes.run_expansion_llm(
        client,
        [{"type": "text", "text": "sys"}],
        "section text",
        _target_rubric(),
        "target",
        "expansion hint",
        "claude-sonnet-4-6",
    )
    instruction = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert "ENUMERATION GUARDRAIL" not in instruction


def test_run_expansion_llm_enumeration_guardrail_appears_before_feedback_block():
    client = _FakeClient('{"children": []}')
    pb_passes.run_expansion_llm(
        client,
        [{"type": "text", "text": "sys"}],
        "section text",
        _dense_target_rubric(),
        "target",
        "expansion hint",
        "claude-sonnet-4-6",
        feedback="check table 3",
    )
    instruction = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert instruction.index("ENUMERATION GUARDRAIL") < instruction.index("USER FEEDBACK")


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


def test_invoke_llm_prints_current_usage_when_tracker_provided(capsys):
    from pb_cost import CostTracker
    tracker = CostTracker()
    client = _FakeClient('{"a": 1}')
    pb_passes.invoke_llm(client, [], [{"role": "user", "content": []}], "claude-opus-4-8", tracker=tracker)
    out = capsys.readouterr().out
    assert f"Current usage: ${tracker.total_cost():.4f}" in out


def test_invoke_llm_no_print_when_tracker_is_none(capsys):
    client = _FakeClient('{"a": 1}')
    pb_passes.invoke_llm(client, [], [{"role": "user", "content": []}], "claude-opus-4-8")
    out = capsys.readouterr().out
    assert "Current usage" not in out


def test_invoke_llm_wraps_errors():
    class _BoomMessages:
        def stream(self, **kwargs):
            raise RuntimeError("network down")

    class _BoomClient:
        messages = _BoomMessages()

    with pytest.raises(RuntimeError, match="Anthropic API call failed"):
        pb_passes.invoke_llm(_BoomClient(), [], [], "claude-opus-4-8")


def test_invoke_llm_raises_clear_error_when_response_truncated_at_max_tokens():
    client = _FakeClient('{"a": 1', stop_reason="max_tokens")
    with pytest.raises(RuntimeError, match="truncated at max_tokens=8000"):
        pb_passes.invoke_llm(client, [], [{"role": "user", "content": []}], "claude-sonnet-4-6")


def test_invoke_llm_does_not_raise_when_stop_reason_is_end_turn():
    client = _FakeClient('{"a": 1}', stop_reason="end_turn")
    result = pb_passes.invoke_llm(client, [], [{"role": "user", "content": []}], "claude-sonnet-4-6")
    assert result == '{"a": 1}'


# ── run_split_check_llm tests ─────────────────────────────────────────────────

def _split_check_branch():
    """A branch with one Result Analysis leaf, one Evaluation/Metrics leaf, one non-matching leaf."""
    return {
        "id": "branch-a", "requirements": "branch a", "weight": 0,
        "task_category": None, "finegrained_task_category": None,
        "sub_tasks": [
            {"id": "result-leaf", "requirements": "small, medium, and large model variants all match",
             "weight": 0, "sub_tasks": [], "task_category": "Result Analysis", "finegrained_task_category": None},
            {"id": "eval-leaf", "requirements": "accuracy on the benchmark", "weight": 0, "sub_tasks": [],
             "task_category": None, "finegrained_task_category": "Evaluation, Metrics & Benchmarking"},
            {"id": "code-leaf", "requirements": "implement the model", "weight": 0, "sub_tasks": [],
             "task_category": "Code Development", "finegrained_task_category": None},
        ],
    }


def _split_check_rubric():
    return {"id": "root", "requirements": "r", "weight": 0, "task_category": None,
            "finegrained_task_category": None, "sub_tasks": [_split_check_branch()]}


def test_run_split_check_llm_only_includes_matching_category_leaves_in_prompt():
    rubric = _split_check_rubric()
    branch = rubric["sub_tasks"][0]
    client = _FakeClient('{"splits": {}}')
    pb_passes.run_split_check_llm(client, [], "section text", rubric, branch, "model")
    instruction = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert "result-leaf" in instruction
    assert "eval-leaf" in instruction
    assert "code-leaf" not in instruction


def test_run_split_check_llm_includes_section_text():
    rubric = _split_check_rubric()
    branch = rubric["sub_tasks"][0]
    client = _FakeClient('{"splits": {}}')
    pb_passes.run_split_check_llm(client, [], "unique section text marker", rubric, branch, "model")
    instruction = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert "unique section text marker" in instruction


def test_run_split_check_llm_skips_llm_call_when_no_matching_leaves():
    rubric = {"id": "root", "requirements": "r", "weight": 0, "task_category": None,
              "finegrained_task_category": None, "sub_tasks": [
                  {"id": "branch-b", "requirements": "branch b", "weight": 0,
                   "task_category": None, "finegrained_task_category": None, "sub_tasks": [
                       {"id": "code-leaf", "requirements": "implement it", "weight": 0, "sub_tasks": [],
                        "task_category": "Code Development", "finegrained_task_category": None},
                   ]},
              ]}
    branch = rubric["sub_tasks"][0]
    client = _FakeClient('{"splits": {}}')
    result = pb_passes.run_split_check_llm(client, [], "text", rubric, branch, "model")
    assert result == {"splits": {}, "duplicates": {}}
    assert client.messages.calls == []


def test_run_split_check_llm_returns_splits_and_duplicates_dict():
    rubric = _split_check_rubric()
    branch = rubric["sub_tasks"][0]
    client = _FakeClient(
        '{"splits": {"result-leaf": [{"id": "a", "requirements": "x"}]}, '
        '"duplicates": {"eval-leaf": "result-leaf"}}'
    )
    result = pb_passes.run_split_check_llm(client, [], "text", rubric, branch, "model")
    assert result == {"splits": {"result-leaf": [{"id": "a", "requirements": "x"}]},
                       "duplicates": {"eval-leaf": "result-leaf"}}


def test_run_split_check_llm_defaults_missing_duplicates_key_to_empty_dict():
    rubric = _split_check_rubric()
    branch = rubric["sub_tasks"][0]
    client = _FakeClient('{"splits": {"result-leaf": [{"id": "a", "requirements": "x"}]}}')
    result = pb_passes.run_split_check_llm(client, [], "text", rubric, branch, "model")
    assert result["duplicates"] == {}


def test_run_split_check_llm_prompt_mentions_duplicates():
    rubric = _split_check_rubric()
    branch = rubric["sub_tasks"][0]
    client = _FakeClient('{"splits": {}, "duplicates": {}}')
    pb_passes.run_split_check_llm(client, [], "text", rubric, branch, "model")
    instruction = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert "duplicates" in instruction.lower()


def test_run_split_check_llm_includes_all_leaves_when_flagged():
    rubric = _split_check_rubric()
    branch = rubric["sub_tasks"][0]
    client = _FakeClient('{"splits": {}, "duplicates": {}}')
    pb_passes.run_split_check_llm(client, [], "text", rubric, branch, "model", include_all_leaves=True)
    instruction = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert "code-leaf" in instruction


def test_run_split_check_llm_include_all_leaves_defaults_false():
    rubric = _split_check_rubric()
    branch = rubric["sub_tasks"][0]
    client = _FakeClient('{"splits": {}, "duplicates": {}}')
    pb_passes.run_split_check_llm(client, [], "text", rubric, branch, "model")
    instruction = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert "code-leaf" not in instruction


def test_split_check_candidates_include_all_returns_every_leaf_regardless_of_category():
    branch = _split_check_branch()
    candidates = pb_passes._split_check_candidates(branch, include_all=True)
    assert {n["id"] for n in candidates} == {"result-leaf", "eval-leaf", "code-leaf"}


def test_run_split_check_llm_scales_max_tokens_with_candidate_count():
    rubric = _split_check_rubric()
    branch = rubric["sub_tasks"][0]
    client = _FakeClient('{"splits": {}, "duplicates": {}}')
    pb_passes.run_split_check_llm(client, [], "text", rubric, branch, "model")
    assert client.messages.calls[0]["max_tokens"] == 8000 + 400 * 2


def test_run_split_check_llm_scales_max_tokens_higher_with_include_all_leaves():
    rubric = _split_check_rubric()
    branch = rubric["sub_tasks"][0]
    client = _FakeClient('{"splits": {}, "duplicates": {}}')
    pb_passes.run_split_check_llm(client, [], "text", rubric, branch, "model", include_all_leaves=True)
    assert client.messages.calls[0]["max_tokens"] == 8000 + 400 * 3


def test_run_split_check_llm_caps_max_tokens_at_32000():
    branch = {
        "id": "huge-branch", "requirements": "r", "weight": 0,
        "task_category": None, "finegrained_task_category": None,
        "sub_tasks": [
            {"id": f"leaf-{i}", "requirements": "x", "weight": 0, "sub_tasks": [],
             "task_category": "Code Development", "finegrained_task_category": None}
            for i in range(100)
        ],
    }
    rubric = {"id": "root", "requirements": "r", "weight": 0, "task_category": None,
              "finegrained_task_category": None, "sub_tasks": [branch]}
    client = _FakeClient('{"splits": {}, "duplicates": {}}')
    pb_passes.run_split_check_llm(client, [], "text", rubric, branch, "model", include_all_leaves=True)
    assert client.messages.calls[0]["max_tokens"] == 32000


def test_split_check_candidates_default_still_filters_by_category():
    branch = _split_check_branch()
    candidates = pb_passes._split_check_candidates(branch)
    assert {n["id"] for n in candidates} == {"result-leaf", "eval-leaf"}


# ── apply_dedup tests ─────────────────────────────────────────────────────────

def _dedup_rubric():
    return {
        "id": "root", "requirements": "r", "weight": 0, "task_category": None,
        "finegrained_task_category": None,
        "sub_tasks": [
            {
                "id": "branch-a", "requirements": "branch a", "weight": 0,
                "task_category": None, "finegrained_task_category": None,
                "sub_tasks": [
                    {"id": "leaf-1", "requirements": "claim one", "weight": 0, "sub_tasks": [],
                     "task_category": "Result Analysis", "finegrained_task_category": None},
                    {"id": "leaf-2", "requirements": "claim one restated", "weight": 0, "sub_tasks": [],
                     "task_category": "Result Analysis", "finegrained_task_category": None},
                ],
            },
            {
                "id": "branch-b", "requirements": "branch b", "weight": 0,
                "task_category": None, "finegrained_task_category": None,
                "sub_tasks": [
                    {"id": "only-child", "requirements": "sole claim", "weight": 0, "sub_tasks": [],
                     "task_category": "Result Analysis", "finegrained_task_category": None},
                ],
            },
        ],
    }


def test_apply_dedup_removes_leaf_from_parent_sub_tasks():
    rubric = _dedup_rubric()
    pb_passes.apply_dedup(rubric, "leaf-2", "leaf-1")
    assert find_node(rubric, "leaf-2") is None
    assert find_node(rubric, "leaf-1") is not None


def test_apply_dedup_raises_for_leaf_with_no_parent():
    rubric = _dedup_rubric()
    with pytest.raises(ValueError):
        pb_passes.apply_dedup(rubric, "root", "leaf-1")
    with pytest.raises(ValueError):
        pb_passes.apply_dedup(rubric, "ghost", "leaf-1")


def test_apply_dedup_logs_removal_with_duplicate_of_id():
    rubric = _dedup_rubric()
    errors = []
    pb_passes.apply_dedup(rubric, "leaf-2", "leaf-1", errors=errors)
    assert errors[0] == "leaf-2: split-check removed as duplicate of 'leaf-1'."


def test_apply_dedup_no_error_mutation_when_errors_none():
    rubric = _dedup_rubric()
    pb_passes.apply_dedup(rubric, "leaf-2", "leaf-1")  # should not raise


def test_apply_dedup_forces_parent_to_leaf_when_last_child_removed():
    rubric = _dedup_rubric()
    errors = []
    pb_passes.apply_dedup(rubric, "only-child", "leaf-1", errors=errors)
    parent = find_node(rubric, "branch-b")
    assert parent["task_category"] == pb_passes._DEPTH_FALLBACK_CATEGORY
    assert parent["sub_tasks"] == []
    assert len(errors) == 2
    assert "branch-b" in errors[1]
    validate_final(rubric)


# ── apply_split tests ─────────────────────────────────────────────────────────

def _split_target_rubric():
    return {
        "id": "root", "requirements": "r", "weight": 0, "task_category": None,
        "finegrained_task_category": None,
        "sub_tasks": [
            {"id": "bundled-leaf", "requirements": "small, medium, and large model variants all match",
             "weight": 0, "sub_tasks": [], "task_category": "Result Analysis", "finegrained_task_category": None},
        ],
    }


def test_apply_split_replaces_leaf_with_children():
    rubric = _split_target_rubric()
    children = [
        {"id": "small-variant", "requirements": "small variant matches", "expandable": False, "task_category": "Result Analysis"},
        {"id": "medium-variant", "requirements": "medium variant matches", "expandable": False, "task_category": "Result Analysis"},
        {"id": "large-variant", "requirements": "large variant matches", "expandable": False, "task_category": "Result Analysis"},
    ]
    new_children = pb_passes.apply_split(rubric, "bundled-leaf", children)
    leaf = find_node(rubric, "bundled-leaf")
    assert leaf["task_category"] is None
    assert leaf["finegrained_task_category"] is None
    assert [c["id"] for c in leaf["sub_tasks"]] == ["small-variant", "medium-variant", "large-variant"]
    assert len(new_children) == 3


def test_apply_split_raises_for_missing_leaf():
    rubric = _split_target_rubric()
    with pytest.raises(ValueError):
        pb_passes.apply_split(rubric, "ghost", [])


def test_apply_split_deduplicates_child_ids_against_rest_of_rubric():
    rubric = _split_target_rubric()
    rubric["sub_tasks"].append(
        {"id": "existing-id", "requirements": "already here", "weight": 0, "sub_tasks": [],
         "task_category": "Code Development", "finegrained_task_category": None}
    )
    children = [{"id": "existing-id", "requirements": "collides", "expandable": False, "task_category": "Result Analysis"}]
    pb_passes.apply_split(rubric, "bundled-leaf", children)
    leaf = find_node(rubric, "bundled-leaf")
    assert leaf["sub_tasks"][0]["id"] != "existing-id"


def test_apply_split_logs_warning_when_child_still_enumerates():
    rubric = _split_target_rubric()
    children = [
        {"id": "still-bundled", "requirements": "including a, b, c, d",
         "expandable": False, "task_category": "Result Analysis"},
    ]
    errors = []
    pb_passes.apply_split(rubric, "bundled-leaf", children, errors=errors)
    leaf = find_node(rubric, "bundled-leaf")
    assert leaf["sub_tasks"][0]["task_category"] == pb_passes._DEPTH_FALLBACK_CATEGORY
    assert len(errors) == 1
    assert "still-bundled" in errors[0]
    assert "under-split" in errors[0]


def test_apply_split_no_error_list_mutation_when_errors_none():
    rubric = _split_target_rubric()
    children = [
        {"id": "still-bundled", "requirements": "including a, b, c, d",
         "expandable": False, "task_category": "Result Analysis"},
    ]
    # Should not raise even though errors=None (default)
    pb_passes.apply_split(rubric, "bundled-leaf", children)


def test_apply_split_accepts_atomic_children_without_warning():
    rubric = _split_target_rubric()
    children = [
        {"id": "atomic-a", "requirements": "atomic claim a", "expandable": False, "task_category": "Result Analysis"},
        {"id": "atomic-b", "requirements": "atomic claim b", "expandable": False, "task_category": "Result Analysis"},
    ]
    errors = []
    pb_passes.apply_split(rubric, "bundled-leaf", children, errors=errors)
    assert errors == []
