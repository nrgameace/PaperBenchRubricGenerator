"""Tests for the orchestrator's pure helpers (queue reconciliation, hint pruning)."""

import copy
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import rubric_gen
from pb_schema import find_node


def _tree():
    """A small rubric with one expanded and two still-pending nodes."""
    return {"id": "root", "requirements": "r", "weight": 0, "task_category": None,
            "finegrained_task_category": None, "sub_tasks": [
                {"id": "expanded", "requirements": "x", "weight": 0, "task_category": None,
                 "finegrained_task_category": None, "sub_tasks": [
                     {"id": "leaf", "requirements": "y", "weight": 0, "sub_tasks": [],
                      "task_category": "Code Development", "finegrained_task_category": None}]},
                {"id": "pending", "requirements": "z", "weight": 0, "sub_tasks": [],
                 "task_category": None, "finegrained_task_category": None},
                {"id": "became_leaf", "requirements": "w", "weight": 0, "sub_tasks": [],
                 "task_category": "Code Execution", "finegrained_task_category": None}]}


def test_reconcile_queue_drops_expanded_categorized_and_missing():
    tree = _tree()
    reconciled = rubric_gen.reconcile_queue(tree, ["expanded", "pending", "became_leaf", "ghost"])
    assert reconciled == ["pending"]


def test_reconcile_queue_preserves_order():
    tree = {"id": "root", "requirements": "r", "weight": 0, "task_category": None,
            "finegrained_task_category": None, "sub_tasks": [
                {"id": "a", "requirements": "x", "weight": 0, "sub_tasks": [], "task_category": None, "finegrained_task_category": None},
                {"id": "b", "requirements": "y", "weight": 0, "sub_tasks": [], "task_category": None, "finegrained_task_category": None}]}
    assert rubric_gen.reconcile_queue(tree, ["b", "a"]) == ["b", "a"]


def test_prune_hints_keeps_only_queued():
    state = {"queue": ["pending"], "hints": {"pending": "h1", "stale": "h2"}}
    rubric_gen.prune_hints(state)
    assert state["hints"] == {"pending": "h1"}


# ── _expand_subtree tests ─────────────────────────────────────────────────────

def _rubric_with_expandable_node():
    """Rubric with root having one unexpanded top-level node."""
    return {
        "id": "root", "requirements": "r", "weight": 0,
        "task_category": None, "finegrained_task_category": None,
        "sub_tasks": [{
            "id": "section-a", "requirements": "section a", "weight": 0,
            "task_category": None, "finegrained_task_category": None, "sub_tasks": []
        }]
    }


def test_expand_subtree_expands_all_levels():
    rubric = _rubric_with_expandable_node()
    hints = {"section-a": "hint for section a"}

    # section-a generates expandable child-1 and leaf child-2
    # child-1 generates leaf child-1-a
    expansion_responses = {
        "section-a": {"children": [
            {"id": "child-1", "requirements": "child one", "expandable": True, "expansion_hint": "child 1 hint"},
            {"id": "child-2", "requirements": "child two", "expandable": False, "task_category": "Code Development"},
        ]},
        "child-1": {"children": [
            {"id": "child-1-a", "requirements": "leaf", "expandable": False, "task_category": "Code Execution"},
        ]},
    }
    expanded_order = []

    def fake_run_expansion_llm(client, system_blocks, section_text, rb, node_id, hint, model, feedback="", tracker=None):
        expanded_order.append(node_id)
        return expansion_responses[node_id]

    with patch("rubric_gen.run_expansion_llm", side_effect=fake_run_expansion_llm), \
         patch("rubric_gen.blocks_to_text", return_value="text"), \
         patch("rubric_gen.slice_section", return_value=[]):
        rubric_gen._expand_subtree(None, [], [], rubric, "section-a", hints, "model")

    assert expanded_order == ["section-a", "child-1"]
    assert len(find_node(rubric, "section-a")["sub_tasks"]) == 2
    assert len(find_node(rubric, "child-1")["sub_tasks"]) == 1
    assert find_node(rubric, "child-1-a")["task_category"] == "Code Execution"


def test_expand_subtree_forwards_feedback_to_all_calls():
    rubric = _rubric_with_expandable_node()
    hints = {"section-a": "hint"}
    feedback_seen = []

    def fake_run_expansion_llm(client, system_blocks, section_text, rb, node_id, hint, model, feedback="", tracker=None):
        feedback_seen.append(feedback)
        return {"children": [
            {"id": f"leaf-{node_id}", "requirements": "x",
             "expandable": False, "task_category": "Code Development"}
        ]}

    with patch("rubric_gen.run_expansion_llm", side_effect=fake_run_expansion_llm), \
         patch("rubric_gen.blocks_to_text", return_value=""), \
         patch("rubric_gen.slice_section", return_value=[]):
        rubric_gen._expand_subtree(None, [], [], rubric, "section-a", hints, "model", feedback="be more detailed")

    assert feedback_seen == ["be more detailed"]


def test_expand_subtree_no_feedback_passes_empty_string():
    rubric = _rubric_with_expandable_node()
    hints = {"section-a": "hint"}
    feedback_seen = []

    def fake_run_expansion_llm(client, system_blocks, section_text, rb, node_id, hint, model, feedback="", tracker=None):
        feedback_seen.append(feedback)
        return {"children": [
            {"id": "leaf", "requirements": "x", "expandable": False, "task_category": "Code Development"}
        ]}

    with patch("rubric_gen.run_expansion_llm", side_effect=fake_run_expansion_llm), \
         patch("rubric_gen.blocks_to_text", return_value=""), \
         patch("rubric_gen.slice_section", return_value=[]):
        rubric_gen._expand_subtree(None, [], [], rubric, "section-a", hints, "model")

    assert feedback_seen == [""]


# ── run_expansion_phase review-frequency test ─────────────────────────────────

def _two_node_state():
    """State with two top-level expandable nodes."""
    return {
        "rubric": {
            "id": "root", "requirements": "r", "weight": 0,
            "task_category": None, "finegrained_task_category": None,
            "sub_tasks": [
                {"id": "node-a", "requirements": "a", "weight": 0,
                 "task_category": None, "finegrained_task_category": None, "sub_tasks": []},
                {"id": "node-b", "requirements": "b", "weight": 0,
                 "task_category": None, "finegrained_task_category": None, "sub_tasks": []},
            ]
        },
        "queue": ["node-a", "node-b"],
        "hints": {"node-a": "hint a", "node-b": "hint b"},
    }


def test_expansion_phase_reviews_once_per_top_level_node(tmp_path):
    state = _two_node_state()
    review_count = {"n": 0}

    def fake_review(rubric, draft_path, validate_fn, input_fn=None):
        review_count["n"] += 1
        return rubric

    with patch("rubric_gen._expand_subtree"), \
         patch("rubric_gen.review_pass", side_effect=fake_review), \
         patch("rubric_gen.pretty_print_nodes"), \
         patch("rubric_gen.commit"), \
         patch("rubric_gen.reconcile_queue", side_effect=lambda _rubric, q: q):
        rubric_gen.run_expansion_phase(None, [], [], state, "model", tmp_path)

    assert review_count["n"] == 2


def test_expansion_phase_passes_feedback_on_rerun(tmp_path):
    from pb_review import RerunPass
    state = _two_node_state()
    # Only testing node-a: first review gives feedback, second approves
    review_calls = {"n": 0}

    def fake_review(rubric, draft_path, validate_fn, input_fn=None):
        review_calls["n"] += 1
        if review_calls["n"] == 1:
            raise RerunPass("needs more detail")
        return rubric

    subtree_calls = []

    def fake_expand_subtree(client, system_blocks, content_list, rubric, node_id, hints, model, feedback="", tracker=None):
        subtree_calls.append((node_id, feedback))

    with patch("rubric_gen._expand_subtree", side_effect=fake_expand_subtree), \
         patch("rubric_gen.review_pass", side_effect=fake_review), \
         patch("rubric_gen.pretty_print_nodes"), \
         patch("rubric_gen.commit"), \
         patch("rubric_gen.reconcile_queue", return_value=[]):
        rubric_gen.run_expansion_phase(None, [], [], state, "model", tmp_path)

    # node-a expanded twice: once with empty feedback, once with "needs more detail"
    node_a_calls = [(nid, fb) for nid, fb in subtree_calls if nid == "node-a"]
    assert node_a_calls == [("node-a", ""), ("node-a", "needs more detail")]


# ── _resolve_invalid_weights tests ───────────────────────────────────────────

def _weighted_rubric():
    return {
        "id": "root", "requirements": "r", "weight": 0,
        "task_category": None, "finegrained_task_category": None,
        "sub_tasks": [
            {"id": "a", "requirements": "do a", "weight": 0, "sub_tasks": [],
             "task_category": "Code Development", "finegrained_task_category": None},
        ],
    }


def test_resolve_invalid_weights_returns_immediately_when_all_valid():
    weights = {"root": 1, "a": 2}
    with patch("rubric_gen.find_invalid_weights", return_value=[]):
        result = rubric_gen._resolve_invalid_weights(None, [], "", _weighted_rubric(), "model", weights)
    assert result == weights


def test_resolve_invalid_weights_applies_manual_override_and_terminates():
    invalid_once = [("a", "do a", None)]
    with patch("rubric_gen.find_invalid_weights", side_effect=[invalid_once, []]), \
         patch("rubric_gen.collect_weight_corrections", return_value=({"a": 5}, [])):
        result = rubric_gen._resolve_invalid_weights(None, [], "", _weighted_rubric(), "model", {"root": 1})
    assert result["a"] == 5


def test_resolve_invalid_weights_calls_llm_for_regen_ids_and_merges():
    invalid_once = [("a", "do a", None)]
    with patch("rubric_gen.find_invalid_weights", side_effect=[invalid_once, []]), \
         patch("rubric_gen.collect_weight_corrections", return_value=({}, ["a"])), \
         patch("rubric_gen.run_weight_llm", return_value={"a": 3}) as mock_llm:
        result = rubric_gen._resolve_invalid_weights(None, [], "text", _weighted_rubric(), "model", {"root": 1})
    assert result["a"] == 3
    assert mock_llm.called
    call_kwargs = mock_llm.call_args[1]
    assert call_kwargs.get("node_ids") == ["a"]


def test_resolve_invalid_weights_loops_on_persistent_invalidity():
    invalid = [("a", "do a", None)]
    with patch("rubric_gen.find_invalid_weights", side_effect=[invalid, invalid, []]), \
         patch("rubric_gen.collect_weight_corrections", return_value=({}, ["a"])), \
         patch("rubric_gen.run_weight_llm", return_value={"a": 4}):
        result = rubric_gen._resolve_invalid_weights(None, [], "", _weighted_rubric(), "model", {})
    assert result["a"] == 4


# ── run_weight_phase feedback bug fix test ───────────────────────────────────

def _minimal_weighted_state():
    return {
        "rubric": {
            "id": "root", "requirements": "r", "weight": 0,
            "task_category": None, "finegrained_task_category": None,
            "sub_tasks": [
                {"id": "leaf", "requirements": "do x", "weight": 0, "sub_tasks": [],
                 "task_category": "Code Development", "finegrained_task_category": None},
            ],
        },
        "queue": [],
        "hints": {},
    }


def test_run_weight_phase_passes_feedback_to_llm_on_retry(tmp_path):
    from pb_review import RerunPass
    state = _minimal_weighted_state()
    review_calls = {"n": 0}
    llm_feedback_args = []

    def fake_review(rubric, draft_path, validate_fn):
        review_calls["n"] += 1
        if review_calls["n"] == 1:
            raise RerunPass("check table 5")
        return rubric

    def fake_run_weight_llm(*args, feedback=None, **kwargs):
        llm_feedback_args.append(feedback)
        return {"root": 1, "leaf": 2}

    with patch("rubric_gen.run_weight_llm", side_effect=fake_run_weight_llm), \
         patch("rubric_gen._resolve_invalid_weights", side_effect=lambda *a, **k: a[5]), \
         patch("rubric_gen.apply_weights"), \
         patch("rubric_gen.review_pass", side_effect=fake_review), \
         patch("rubric_gen.pretty_print_nodes"), \
         patch("rubric_gen.commit"), \
         patch("rubric_gen.blocks_to_text", return_value=""):
        rubric_gen.run_weight_phase(None, [], [], state, "model", tmp_path)

    assert llm_feedback_args[0] is None
    assert llm_feedback_args[1] == "check table 5"


# ── --review flag / agentic mode tests ───────────────────────────────────────

def test_parse_args_review_flag_true_when_present(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["rubric_gen", "--input", "in", "--output", "out", "--review"])
    args = rubric_gen.parse_args()
    assert args.review is True


def test_parse_args_review_flag_false_when_omitted(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["rubric_gen", "--input", "in", "--output", "out"])
    args = rubric_gen.parse_args()
    assert args.review is False


def test_resolve_invalid_weights_agentic_auto_queues_all_for_regen():
    invalid = [("a", "do a", None), ("b", "do b", -1)]
    with patch("rubric_gen.find_invalid_weights", side_effect=[invalid, []]), \
         patch("rubric_gen.collect_weight_corrections") as mock_collect, \
         patch("rubric_gen.run_weight_llm", return_value={"a": 1, "b": 2}):
        result = rubric_gen._resolve_invalid_weights(
            None, [], "", _weighted_rubric(), "model", {}, human_review=False
        )
    mock_collect.assert_not_called()
    assert result["a"] == 1 and result["b"] == 2


def test_run_base_phase_agentic_skips_review(tmp_path):
    state = {"rubric": {}, "queue": [], "hints": {}}
    fake_rubric = {"id": "root", "sub_tasks": [], "requirements": "r"}

    with patch("rubric_gen.run_base_llm", return_value={"root": {"requirements": "r"}, "children": []}), \
         patch("rubric_gen.apply_base", return_value=(fake_rubric, [], {})), \
         patch("rubric_gen.review_pass") as mock_review, \
         patch("rubric_gen.pretty_print_nodes"), \
         patch("rubric_gen.commit"), \
         patch("rubric_gen.reconcile_queue", return_value=[]), \
         patch("rubric_gen.blocks_to_text", return_value=""):
        rubric_gen.run_base_phase(None, [], None, [], state, "model", tmp_path, human_review=False)

    mock_review.assert_not_called()


def test_run_expansion_phase_agentic_skips_review(tmp_path):
    state = _two_node_state()

    with patch("rubric_gen._expand_subtree"), \
         patch("rubric_gen.review_pass") as mock_review, \
         patch("rubric_gen.pretty_print_nodes"), \
         patch("rubric_gen.commit"), \
         patch("rubric_gen.reconcile_queue", side_effect=lambda _rubric, q: q):
        rubric_gen.run_expansion_phase(None, [], [], state, "model", tmp_path, human_review=False)

    mock_review.assert_not_called()


def test_run_weight_phase_agentic_skips_review(tmp_path):
    state = _minimal_weighted_state()

    with patch("rubric_gen.run_weight_llm", return_value={"root": 1, "leaf": 2}), \
         patch("rubric_gen._resolve_invalid_weights", side_effect=lambda *a, **k: a[5]), \
         patch("rubric_gen.apply_weights"), \
         patch("rubric_gen.review_pass") as mock_review, \
         patch("rubric_gen.pretty_print_nodes"), \
         patch("rubric_gen.commit"), \
         patch("rubric_gen.blocks_to_text", return_value=""):
        rubric_gen.run_weight_phase(None, [], [], state, "model", tmp_path, human_review=False)

    mock_review.assert_not_called()
