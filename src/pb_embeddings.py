"""Deterministic embedding-based branch mass computation and weight rescaling.

Replaces the LLM-driven global weight recalibration: branch importance is derived from
(a) how many distinct claims a branch's leaves embed into (via clustering), and
(b) how much of the paper's actual content that branch covers (page/table/figure span,
from the base pass's section_map), not from raw leaf/weight counts an LLM guesses at.
"""

import json
import math
from pathlib import Path

import openai

from pb_schema import iter_nodes

EMBEDDING_MODEL = "text-embedding-3-small"
BRANCH_CLUSTER_THRESHOLD = 0.87
DUPLICATE_CLUSTER_THRESHOLD = 0.92


def build_embedding_client() -> openai.OpenAI:
    """Construct the OpenAI client used for embeddings only."""
    return openai.OpenAI()


def embed_texts(client, texts: list, model: str = EMBEDDING_MODEL) -> list:
    """Return one embedding vector per input text, in order, via a single batched call."""
    if not texts:
        return []
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def extract_leaves(rubric: dict) -> list:
    """Return {id, requirements, weight, branch_id} for every leaf, tagged with the
    top-level branch (direct child of root) it descends from."""
    leaves = []
    for branch in rubric.get("sub_tasks", []) or []:
        branch_id = branch["id"]
        for node in iter_nodes(branch):
            if not node.get("sub_tasks"):
                leaves.append({
                    "id": node["id"],
                    "requirements": node["requirements"],
                    "weight": node.get("weight", 0),
                    "branch_id": branch_id,
                })
    return leaves


def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cluster_by_threshold(ids: list, vectors: list, threshold: float) -> list:
    """Greedy single-link clustering: an id joins the first existing cluster with
    cosine similarity >= threshold to any member, else starts a new cluster."""
    clusters, cluster_vectors = [], []
    for node_id, vector in zip(ids, vectors):
        for cluster, vectors_in_cluster in zip(clusters, cluster_vectors):
            if any(cosine_similarity(vector, other) >= threshold for other in vectors_in_cluster):
                cluster.append(node_id)
                vectors_in_cluster.append(vector)
                break
        else:
            clusters.append([node_id])
            cluster_vectors.append([vector])
    return clusters


def branch_mass(leaves: list, vectors_by_id: dict, branch_id: str, threshold: float = BRANCH_CLUSTER_THRESHOLD) -> int:
    """Return the distinct-claim cluster count for one branch's leaves (its 'mass')."""
    branch_leaves = [leaf for leaf in leaves if leaf["branch_id"] == branch_id]
    ids = [leaf["id"] for leaf in branch_leaves]
    vectors = [vectors_by_id[node_id] for node_id in ids]
    return len(cluster_by_threshold(ids, vectors, threshold))


def compute_all_branch_masses(rubric: dict, vectors_by_id: dict, threshold: float = BRANCH_CLUSTER_THRESHOLD) -> dict:
    """Return {branch_id: mass} for every top-level branch."""
    leaves = extract_leaves(rubric)
    branch_ids = [branch["id"] for branch in rubric.get("sub_tasks", []) or []]
    return {branch_id: branch_mass(leaves, vectors_by_id, branch_id, threshold) for branch_id in branch_ids}


def _section_score(entry) -> float | None:
    """Return a branch's raw page+table+figure span score, or None if malformed."""
    if not isinstance(entry, dict):
        return None
    pages, tables, figures = entry.get("pages"), entry.get("tables"), entry.get("figures")
    is_numeric = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    if not (isinstance(pages, (list, tuple)) and len(pages) == 2 and all(is_numeric(p) for p in pages)):
        return None
    if not (is_numeric(tables) and is_numeric(figures)):
        return None
    return max(pages[1] - pages[0], 0) + tables + figures


def derive_target_proportions(section_map: dict, branch_ids: list) -> dict:
    """Convert a base-pass section_map into normalized target proportions per branch id.

    A branch missing from section_map, or with a malformed entry, falls back to a uniform
    target (1 / len(branch_ids)) for that branch only, so one bad entry never fails the
    whole rescale.
    """
    section_map = section_map or {}
    scores = {branch_id: _section_score(section_map.get(branch_id)) for branch_id in branch_ids}
    uniform_share = 1 / len(branch_ids)
    fallback_ids = [branch_id for branch_id, score in scores.items() if score is None]
    valid_ids = [branch_id for branch_id, score in scores.items() if score is not None]

    targets = {branch_id: uniform_share for branch_id in fallback_ids}
    remaining_share = 1 - len(fallback_ids) * uniform_share
    valid_total = sum(scores[branch_id] for branch_id in valid_ids)
    for branch_id in valid_ids:
        if valid_total > 0:
            targets[branch_id] = remaining_share * (scores[branch_id] / valid_total)
        else:
            targets[branch_id] = uniform_share
    return targets


def rescale_branch_weights(leaves: list, masses: dict, targets: dict) -> dict:
    """Return {leaf_id: new_weight} for every leaf, rescaled per its branch's factor.

    factor = (targets[branch] * total_mass) / masses[branch]; every leaf's current weight
    is multiplied by factor and rounded to an int, floored at 1.
    """
    total_mass = sum(masses.values())
    weights = {}
    for leaf in leaves:
        mass = masses.get(leaf["branch_id"], 0)
        target = targets.get(leaf["branch_id"], 0)
        factor = (target * total_mass) / mass if mass > 0 else 1
        weights[leaf["id"]] = max(1, round(leaf["weight"] * factor))
    return weights


def cluster_cross_branch_duplicates(leaves: list, vectors_by_id: dict, threshold: float = DUPLICATE_CLUSTER_THRESHOLD) -> list:
    """Cluster all leaves ignoring branch boundaries; return only clusters spanning >1 branch."""
    ids = [leaf["id"] for leaf in leaves]
    vectors = [vectors_by_id[node_id] for node_id in ids]
    branch_by_id = {leaf["id"]: leaf["branch_id"] for leaf in leaves}
    clusters = cluster_by_threshold(ids, vectors, threshold)
    return [cluster for cluster in clusters if len({branch_by_id[node_id] for node_id in cluster}) > 1]


def build_duplicate_report(clusters: list, leaves: list) -> list:
    """Convert id-clusters into JSON-serializable entries with branch ids and requirements."""
    leaves_by_id = {leaf["id"]: leaf for leaf in leaves}
    report = []
    for cluster in clusters:
        report.append({
            "leaf_ids": cluster,
            "branch_ids": sorted({leaves_by_id[node_id]["branch_id"] for node_id in cluster}),
            "requirements": {node_id: leaves_by_id[node_id]["requirements"] for node_id in cluster},
        })
    return report


def write_flagged_duplicates(report: list, output_dir) -> None:
    """Write flagged_duplicates.json alongside rubric_final.json, only when non-empty."""
    if not report:
        return
    path = Path(output_dir) / "flagged_duplicates.json"
    path.write_text(json.dumps(report, indent=2))


def rescale_global_weights(rubric: dict, client, section_map: dict) -> tuple:
    """Extract leaves, embed once, compute branch masses, derive targets, rescale weights,
    and flag cross-branch duplicates.

    Returns (new_weights: dict[leaf_id, int], duplicate_report: list). Internal-node and
    root weights are untouched — the caller overlays the returned leaf weights onto the
    existing weights dict.
    """
    leaves = extract_leaves(rubric)
    vectors = embed_texts(client, [leaf["requirements"] for leaf in leaves])
    vectors_by_id = {leaf["id"]: vector for leaf, vector in zip(leaves, vectors)}

    branch_ids = [branch["id"] for branch in rubric.get("sub_tasks", []) or []]
    masses = {branch_id: branch_mass(leaves, vectors_by_id, branch_id) for branch_id in branch_ids}
    targets = derive_target_proportions(section_map, branch_ids)

    new_weights = rescale_branch_weights(leaves, masses, targets)
    duplicate_clusters = cluster_cross_branch_duplicates(leaves, vectors_by_id)
    duplicate_report = build_duplicate_report(duplicate_clusters, leaves)
    return new_weights, duplicate_report
