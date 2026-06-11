"""Tests for the self-contained TaskNode: validation, traversal, and transforms."""

import pytest

from task_node import TaskNode, reduce_to_category, zero_weight_by_category


def _leaf(node_id, category="Code Development", weight=1, finegrained=None):
    """Build a valid leaf TaskNode."""
    return TaskNode(id=node_id, requirements="do x", weight=weight, sub_tasks=[],
                    task_category=category, finegrained_task_category=finegrained)


def _internal(node_id, children, weight=1):
    """Build a valid internal TaskNode."""
    return TaskNode(id=node_id, requirements="area", weight=weight, sub_tasks=list(children))


def test_leaf_requires_task_category():
    with pytest.raises(ValueError, match="doesn't have a task category"):
        TaskNode(id="a", requirements="x", weight=1, sub_tasks=[])


def test_internal_rejects_task_category():
    with pytest.raises(ValueError, match="cannot have a task category"):
        TaskNode(id="root", requirements="r", weight=1, sub_tasks=[_leaf("a")], task_category="Code Development")


def test_rejects_negative_weight():
    with pytest.raises(ValueError, match="non-negative"):
        _leaf("a", weight=-1)


def test_rejects_non_numeric_weight():
    with pytest.raises(ValueError, match="must be a number"):
        TaskNode(id="a", requirements="x", weight="heavy", sub_tasks=[], task_category="Code Development")


def test_rejects_invalid_task_category():
    with pytest.raises(ValueError, match="Invalid task category"):
        _leaf("a", category="Nonsense")


def test_rejects_invalid_finegrained_category():
    with pytest.raises(ValueError, match="Invalid finegrained"):
        _leaf("a", finegrained="Nonsense")


def test_from_dict_roundtrips_to_dict():
    data = {"id": "root", "requirements": "r", "weight": 1,
            "sub_tasks": [{"id": "a", "requirements": "x", "weight": 2, "sub_tasks": [],
                           "task_category": "Code Development", "finegrained_task_category": None}],
            "task_category": None, "finegrained_task_category": None}
    assert TaskNode.from_dict(data).to_dict() == data


def test_from_dict_missing_field_raises():
    with pytest.raises(ValueError, match="Missing required field"):
        TaskNode.from_dict({"id": "root", "weight": 1, "sub_tasks": [], "task_category": "Code Development"})


def test_find_and_contains_and_get_parent():
    tree = _internal("root", [_leaf("a"), _internal("b", [_leaf("c")])])
    assert tree.find("c").id == "c"
    assert tree.contains("c") and not tree.contains("missing")
    assert tree.get_parent("c").id == "b"
    with pytest.raises(ValueError):
        tree.find("missing")


def test_get_parent_of_root_raises():
    with pytest.raises(ValueError, match="root node has no parent"):
        _leaf("root").get_parent("root")


def test_replace_and_delete():
    tree = _internal("root", [_leaf("a"), _leaf("b")])
    replaced = tree.replace("a", _leaf("a", category="Code Execution"))
    assert replaced.find("a").task_category == "Code Execution"
    deleted = tree.delete("a")
    assert [child.id for child in deleted.sub_tasks] == ["b"]


def test_replace_missing_raises():
    with pytest.raises(ValueError, match="not found"):
        _internal("root", [_leaf("a")]).replace("ghost", _leaf("ghost"))


def test_add_sub_task_clears_task_category():
    leaf = _leaf("root")
    expanded = leaf.add_sub_task(_leaf("child"))
    assert expanded.task_category is None and not expanded.is_leaf()


def test_set_sub_tasks_clears_category_when_nonempty():
    leaf = _leaf("root")  # starts as a categorized leaf
    node = leaf.set_sub_tasks([_leaf("a")])
    assert not node.is_leaf() and node.task_category is None


def test_get_leaf_nodes_and_descendants():
    tree = _internal("root", [_leaf("a"), _internal("b", [_leaf("c"), _leaf("d")])])
    assert [n.id for n in tree.get_leaf_nodes()] == ["a", "c", "d"]
    assert [n.id for n in tree.get_descendants_depth_first()] == ["a", "b", "c", "d"]


def test_get_descendants_with_duplicate_ids():
    tree = _internal("root", [_leaf("dup"), _internal("b", [_leaf("dup")])])
    assert {n.id for n in tree.get_descendants_with_duplicate_ids()} == {"dup"}


def test_get_prior_nodes():
    tree = _internal("A", [_internal("B", [_leaf("D"), _leaf("E")]), _internal("C", [_leaf("F"), _leaf("G")])])
    prior = tree.find("G").get_prior_nodes(tree)
    assert [n.id for n in prior] == ["A", "B", "C", "F"]


def test_prune_to_depth_noop_when_within_depth():
    tree = _internal("root", [_internal("b", [_leaf("c")])])
    assert tree.prune_to_depth(5).to_dict() == tree.to_dict()


def test_duplicate_with_new_ids_changes_every_id():
    tree = _internal("root", [_leaf("a"), _leaf("b")])
    duplicated = tree.duplicate_with_new_ids()
    original_ids = {"root", "a", "b"}
    new_ids = {n.id for n in [duplicated] + duplicated.get_descendants_depth_first()}
    assert new_ids.isdisjoint(original_ids)


def test_code_only_drops_non_code_leaves():
    tree = _internal("root", [_leaf("keep", "Code Development"), _leaf("drop", "Result Analysis")])
    reduced = tree.code_only()
    assert [n.id for n in reduced.sub_tasks] == ["keep"]


def test_code_only_returns_none_when_no_code():
    assert _internal("root", [_leaf("a", "Result Analysis")]).code_only() is None


def test_resources_provided_zeroes_acquisition_weight():
    tree = _internal("root", [_leaf("data", "Code Development", weight=5, finegrained="Dataset and Model Acquisition"),
                              _leaf("other", "Code Development", weight=5)])
    out = tree.resources_provided()
    assert out.find("data").weight == 0 and out.find("other").weight == 5


def test_zero_weight_by_category_requires_exactly_one_arg():
    with pytest.raises(ValueError, match="exactly one"):
        zero_weight_by_category(_leaf("a"))


def test_reduce_to_category_keeps_matching_subtree():
    tree = _internal("root", [_internal("b", [_leaf("c", "Code Development"), _leaf("d", "Result Analysis")])])
    reduced = reduce_to_category(tree, "Code Development")
    assert [n.id for n in reduced.find("b").sub_tasks] == ["c"]
