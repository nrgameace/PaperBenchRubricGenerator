"""Tests for the orchestrator's pure helpers (queue reconciliation, hint pruning)."""

import rubric_gen


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
