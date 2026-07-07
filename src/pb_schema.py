"""Rubric traversal and validation against the self-contained PaperBench TaskNode."""

import copy

from task_node import VALID_FINEGRAINED_TASK_CATEGORIES, TaskNode

LEAF_CATEGORIES = ("Code Development", "Code Execution", "Result Analysis")
FINEGRAINED_CATEGORIES = tuple(VALID_FINEGRAINED_TASK_CATEGORIES)
_PLACEHOLDER_CATEGORY = "Code Development"


def iter_nodes(node: dict):
    """Yield every node dict in the tree depth-first, including the root."""
    yield node
    for child in node.get("sub_tasks", []) or []:
        yield from iter_nodes(child)


def find_node(node: dict, node_id: str):
    """Return the node dict with ``node_id`` or None if it is absent."""
    for candidate in iter_nodes(node):
        if candidate.get("id") == node_id:
            return candidate
    return None


def all_ids(node: dict) -> list:
    """Return every node id present in the tree."""
    return [candidate.get("id") for candidate in iter_nodes(node)]


def find_parent(node: dict, node_id: str):
    """Return the parent dict of ``node_id`` within node's subtree, or None if node_id is
    the root of this subtree or absent."""
    for candidate in iter_nodes(node):
        for child in candidate.get("sub_tasks", []) or []:
            if child.get("id") == node_id:
                return candidate
    return None


def node_depth(node: dict, node_id: str, _current_depth: int = 0) -> int | None:
    """Return the depth of ``node_id`` from ``node`` (root=0), or None if absent."""
    if node.get("id") == node_id:
        return _current_depth
    for child in node.get("sub_tasks", []) or []:
        found = node_depth(child, node_id, _current_depth + 1)
        if found is not None:
            return found
    return None


def _materialize_pending(node: dict, pending_ids: set) -> dict:
    """Deep-copy the tree, stamping queued-but-unexpanded leaf nodes with a placeholder category."""
    clone = copy.deepcopy(node)
    for candidate in iter_nodes(clone):
        is_leaf = not candidate.get("sub_tasks")
        needs_stamp = candidate.get("id") in pending_ids and not candidate.get("task_category")
        if is_leaf and needs_stamp:
            candidate["task_category"] = _PLACEHOLDER_CATEGORY
    return clone


def validate_partial(rubric: dict, pending_ids=None) -> None:
    """Validate an in-progress rubric, tolerating queued-but-unexpanded nodes. Raises ValueError."""
    materialized = _materialize_pending(rubric, set(pending_ids or []))
    TaskNode.from_dict(materialized)


def validate_final(rubric: dict) -> None:
    """Validate a completed rubric and reject duplicate ids. Raises ValueError on any problem."""
    TaskNode.from_dict(rubric)
    ids = all_ids(rubric)
    duplicates = sorted({node_id for node_id in ids if ids.count(node_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate node ids found: {duplicates}")
