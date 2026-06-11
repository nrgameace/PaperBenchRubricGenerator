"""Tests for schema access, traversal, and validation against the genuine TaskNode."""

import copy

import pytest

import pb_schema


def _leaf(node_id, category="Code Development"):
    """Build a valid leaf node dict."""
    return {"id": node_id, "requirements": "do x", "weight": 1, "sub_tasks": [],
            "task_category": category, "finegrained_task_category": None}


def _internal(node_id, children):
    """Build an internal node dict with the given children."""
    return {"id": node_id, "requirements": "area", "weight": 1, "sub_tasks": list(children),
            "task_category": None, "finegrained_task_category": None}


def test_iter_nodes_and_find_and_all_ids():
    tree = _internal("root", [_leaf("a"), _internal("b", [_leaf("c")])])
    assert pb_schema.all_ids(tree) == ["root", "a", "b", "c"]
    assert pb_schema.find_node(tree, "c")["id"] == "c"
    assert pb_schema.find_node(tree, "missing") is None


def test_validate_final_accepts_valid_tree():
    tree = _internal("root", [_leaf("a"), _leaf("b")])
    pb_schema.validate_final(tree)


def test_validate_final_rejects_leaf_without_category():
    tree = _internal("root", [{"id": "a", "requirements": "x", "weight": 1, "sub_tasks": [],
                               "task_category": None, "finegrained_task_category": None}])
    with pytest.raises(ValueError):
        pb_schema.validate_final(tree)


def test_validate_final_rejects_internal_with_category():
    bad = {"id": "root", "requirements": "r", "weight": 1, "task_category": "Code Development",
           "finegrained_task_category": None, "sub_tasks": [_leaf("a")]}
    with pytest.raises(ValueError):
        pb_schema.validate_final(bad)


def test_validate_final_rejects_duplicate_ids():
    tree = _internal("root", [_leaf("dup"), _leaf("dup")])
    with pytest.raises(ValueError, match="Duplicate"):
        pb_schema.validate_final(tree)


def test_validate_final_rejects_missing_required_field():
    tree = {"id": "root", "weight": 1, "sub_tasks": [], "task_category": "Code Development"}
    with pytest.raises(ValueError):
        pb_schema.validate_final(tree)


def test_validate_partial_tolerates_pending_leaf():
    pending = {"id": "root", "requirements": "r", "weight": 0, "task_category": None,
               "finegrained_task_category": None,
               "sub_tasks": [{"id": "p", "requirements": "todo", "weight": 0, "sub_tasks": [],
                              "task_category": None, "finegrained_task_category": None}]}
    pb_schema.validate_partial(pending, pending_ids={"p"})


def test_validate_partial_does_not_mutate_input():
    pending = {"id": "root", "requirements": "r", "weight": 0, "task_category": None,
               "finegrained_task_category": None,
               "sub_tasks": [{"id": "p", "requirements": "todo", "weight": 0, "sub_tasks": [],
                              "task_category": None, "finegrained_task_category": None}]}
    before = copy.deepcopy(pending)
    pb_schema.validate_partial(pending, pending_ids={"p"})
    assert pending == before


def test_validate_partial_still_flags_non_pending_uncategorized_leaf():
    pending = {"id": "root", "requirements": "r", "weight": 0, "task_category": None,
               "finegrained_task_category": None,
               "sub_tasks": [{"id": "real", "requirements": "x", "weight": 0, "sub_tasks": [],
                              "task_category": None, "finegrained_task_category": None}]}
    with pytest.raises(ValueError):
        pb_schema.validate_partial(pending, pending_ids=set())
