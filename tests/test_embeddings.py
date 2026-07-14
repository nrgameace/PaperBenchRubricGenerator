"""Tests for deterministic embedding-based branch mass computation and weight rescaling."""

from types import SimpleNamespace

import pytest

import pb_embeddings


# ── fake OpenAI embeddings client ─────────────────────────────────────────────

class _FakeEmbeddingResponse:
    def __init__(self, vectors):
        self.data = [SimpleNamespace(embedding=v) for v in vectors]


class _FakeEmbeddings:
    def __init__(self, vectors):
        self.calls = []
        self._vectors = vectors

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeEmbeddingResponse(self._vectors)


class _FakeEmbeddingClient:
    def __init__(self, vectors):
        self.embeddings = _FakeEmbeddings(vectors)


def _branch_rubric():
    return {
        "id": "root", "requirements": "r", "weight": 1, "task_category": None,
        "finegrained_task_category": None,
        "sub_tasks": [
            {
                "id": "branch-a", "requirements": "branch a", "weight": 0,
                "task_category": None, "finegrained_task_category": None,
                "sub_tasks": [
                    {"id": "a1", "requirements": "claim one", "weight": 3, "sub_tasks": [],
                     "task_category": "Code Development", "finegrained_task_category": None},
                    {"id": "a2", "requirements": "claim one restated", "weight": 2, "sub_tasks": [],
                     "task_category": "Code Development", "finegrained_task_category": None},
                ],
            },
            {
                "id": "branch-b", "requirements": "branch b", "weight": 0,
                "task_category": None, "finegrained_task_category": None,
                "sub_tasks": [
                    {"id": "b1", "requirements": "unrelated claim", "weight": 4, "sub_tasks": [],
                     "task_category": "Code Development", "finegrained_task_category": None},
                ],
            },
        ],
    }


# ── embed_texts ────────────────────────────────────────────────────────────

def test_embed_texts_returns_vectors_in_order():
    client = _FakeEmbeddingClient([[1.0, 0.0], [0.0, 1.0]])
    vectors = pb_embeddings.embed_texts(client, ["a", "b"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert len(client.embeddings.calls) == 1
    assert client.embeddings.calls[0]["input"] == ["a", "b"]


def test_embed_texts_empty_list_makes_no_call():
    client = _FakeEmbeddingClient([])
    assert pb_embeddings.embed_texts(client, []) == []
    assert client.embeddings.calls == []


# ── extract_leaves ─────────────────────────────────────────────────────────

def test_extract_leaves_tags_branch_id():
    leaves = pb_embeddings.extract_leaves(_branch_rubric())
    by_id = {leaf["id"]: leaf for leaf in leaves}
    assert set(by_id) == {"a1", "a2", "b1"}
    assert by_id["a1"]["branch_id"] == "branch-a"
    assert by_id["a2"]["branch_id"] == "branch-a"
    assert by_id["b1"]["branch_id"] == "branch-b"
    assert by_id["a1"]["requirements"] == "claim one"
    assert by_id["a1"]["weight"] == 3


def test_extract_leaves_excludes_internal_nodes():
    leaves = pb_embeddings.extract_leaves(_branch_rubric())
    ids = {leaf["id"] for leaf in leaves}
    assert "branch-a" not in ids
    assert "root" not in ids


# ── cosine_similarity ──────────────────────────────────────────────────────

def test_cosine_similarity_identical_vectors_is_one():
    assert pb_embeddings.cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert pb_embeddings.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one():
    assert pb_embeddings.cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


# ── cluster_by_threshold ───────────────────────────────────────────────────

def test_cluster_by_threshold_groups_similar_vectors():
    ids = ["x", "y", "z"]
    vectors = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    clusters = pb_embeddings.cluster_by_threshold(ids, vectors, threshold=0.9)
    assert sorted(sorted(c) for c in clusters) == [["x", "y"], ["z"]]


def test_cluster_by_threshold_all_distinct_below_threshold():
    ids = ["x", "y"]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    clusters = pb_embeddings.cluster_by_threshold(ids, vectors, threshold=0.9)
    assert sorted(sorted(c) for c in clusters) == [["x"], ["y"]]


def test_cluster_by_threshold_empty_input():
    assert pb_embeddings.cluster_by_threshold([], [], threshold=0.9) == []


# ── branch_mass / compute_all_branch_masses ───────────────────────────────

def test_branch_mass_counts_distinct_clusters_not_leaf_count():
    """Regression test for the SmoothQuant bug: a branch with many leaves that all
    restate the same claim must have LOW mass, not high mass from raw leaf count."""
    leaves = pb_embeddings.extract_leaves(_branch_rubric())
    vectors_by_id = {"a1": [1.0, 0.0], "a2": [0.99, 0.01], "b1": [0.0, 1.0]}
    assert pb_embeddings.branch_mass(leaves, vectors_by_id, "branch-a", threshold=0.9) == 1
    assert pb_embeddings.branch_mass(leaves, vectors_by_id, "branch-b", threshold=0.9) == 1


def test_compute_all_branch_masses_returns_mass_per_branch():
    rubric = _branch_rubric()
    vectors_by_id = {"a1": [1.0, 0.0], "a2": [0.99, 0.01], "b1": [0.0, 1.0]}
    masses = pb_embeddings.compute_all_branch_masses(rubric, vectors_by_id, threshold=0.9)
    assert masses == {"branch-a": 1, "branch-b": 1}


# ── derive_target_proportions ──────────────────────────────────────────────

def test_derive_target_proportions_uses_page_table_figure_span():
    section_map = {
        "branch-a": {"pages": [1, 5], "tables": 1, "figures": 0},
        "branch-b": {"pages": [5, 6], "tables": 0, "figures": 0},
    }
    targets = pb_embeddings.derive_target_proportions(section_map, ["branch-a", "branch-b"])
    # branch-a raw = (5-1)+1+0 = 5; branch-b raw = (6-5)+0+0 = 1; total = 6
    assert targets["branch-a"] == pytest.approx(5 / 6)
    assert targets["branch-b"] == pytest.approx(1 / 6)
    assert sum(targets.values()) == pytest.approx(1.0)


def test_derive_target_proportions_falls_back_to_uniform_for_missing_branch():
    section_map = {"branch-a": {"pages": [1, 5], "tables": 0, "figures": 0}}
    targets = pb_embeddings.derive_target_proportions(section_map, ["branch-a", "branch-b"])
    assert targets["branch-b"] == pytest.approx(0.5)
    assert sum(targets.values()) == pytest.approx(1.0)


def test_derive_target_proportions_falls_back_to_uniform_for_malformed_entry():
    section_map = {
        "branch-a": {"pages": [1, 5], "tables": 0, "figures": 0},
        "branch-b": {"pages": "not-a-list", "tables": 0, "figures": 0},
    }
    targets = pb_embeddings.derive_target_proportions(section_map, ["branch-a", "branch-b"])
    assert targets["branch-b"] == pytest.approx(0.5)
    assert sum(targets.values()) == pytest.approx(1.0)


def test_derive_target_proportions_empty_section_map_is_uniform():
    targets = pb_embeddings.derive_target_proportions({}, ["branch-a", "branch-b", "branch-c"])
    assert targets == {"branch-a": pytest.approx(1 / 3), "branch-b": pytest.approx(1 / 3),
                        "branch-c": pytest.approx(1 / 3)}


# ── rescale_branch_weights ─────────────────────────────────────────────────

def test_rescale_branch_weights_applies_factor_per_branch():
    leaves = [
        {"id": "a1", "branch_id": "branch-a", "weight": 3},
        {"id": "a2", "branch_id": "branch-a", "weight": 2},
        {"id": "b1", "branch_id": "branch-b", "weight": 4},
    ]
    masses = {"branch-a": 1, "branch-b": 1}
    targets = {"branch-a": 0.8, "branch-b": 0.2}
    weights = pb_embeddings.rescale_branch_weights(leaves, masses, targets)
    # total_mass = 2; factor_a = (0.8*2)/1 = 1.6; factor_b = (0.2*2)/1 = 0.4
    assert weights["a1"] == round(3 * 1.6)
    assert weights["a2"] == round(2 * 1.6)
    assert weights["b1"] == max(1, round(4 * 0.4))


def test_rescale_branch_weights_floors_at_one():
    leaves = [{"id": "a1", "branch_id": "branch-a", "weight": 1}]
    masses = {"branch-a": 5}
    targets = {"branch-a": 0.01}
    weights = pb_embeddings.rescale_branch_weights(leaves, masses, targets)
    assert weights["a1"] >= 1


def test_rescale_branch_weights_handles_zero_mass_defensively():
    leaves = [{"id": "a1", "branch_id": "branch-a", "weight": 3}]
    masses = {"branch-a": 0}
    targets = {"branch-a": 1.0}
    weights = pb_embeddings.rescale_branch_weights(leaves, masses, targets)
    assert weights["a1"] == 3  # factor defaults to 1 (no rescale) when mass is 0


# ── cluster_cross_branch_duplicates ────────────────────────────────────────

def test_cluster_cross_branch_duplicates_only_returns_multi_branch_clusters():
    leaves = [
        {"id": "a1", "branch_id": "branch-a"},
        {"id": "b1", "branch_id": "branch-b"},
        {"id": "a2", "branch_id": "branch-a"},
    ]
    vectors_by_id = {"a1": [1.0, 0.0], "b1": [0.999, 0.001], "a2": [0.0, 1.0]}
    clusters = pb_embeddings.cluster_cross_branch_duplicates(leaves, vectors_by_id, threshold=0.95)
    assert clusters == [["a1", "b1"]]


def test_cluster_cross_branch_duplicates_excludes_single_branch_clusters():
    leaves = [
        {"id": "a1", "branch_id": "branch-a"},
        {"id": "a2", "branch_id": "branch-a"},
    ]
    vectors_by_id = {"a1": [1.0, 0.0], "a2": [0.999, 0.001]}
    clusters = pb_embeddings.cluster_cross_branch_duplicates(leaves, vectors_by_id, threshold=0.95)
    assert clusters == []


# ── build_duplicate_report / write_flagged_duplicates ──────────────────────

def test_build_duplicate_report_includes_branch_ids_and_requirements():
    leaves = [
        {"id": "a1", "branch_id": "branch-a", "requirements": "claim x", "weight": 1},
        {"id": "b1", "branch_id": "branch-b", "requirements": "claim x restated", "weight": 1},
    ]
    report = pb_embeddings.build_duplicate_report([["a1", "b1"]], leaves)
    assert report == [{
        "leaf_ids": ["a1", "b1"],
        "branch_ids": ["branch-a", "branch-b"],
        "requirements": {"a1": "claim x", "b1": "claim x restated"},
    }]


def test_write_flagged_duplicates_skips_when_empty(tmp_path):
    pb_embeddings.write_flagged_duplicates([], tmp_path)
    assert not (tmp_path / "flagged_duplicates.json").exists()


def test_write_flagged_duplicates_writes_json_when_nonempty(tmp_path):
    report = [{"leaf_ids": ["a1", "b1"], "branch_ids": ["branch-a", "branch-b"],
               "requirements": {"a1": "x", "b1": "y"}}]
    pb_embeddings.write_flagged_duplicates(report, tmp_path)
    path = tmp_path / "flagged_duplicates.json"
    assert path.exists()
    import json
    assert json.loads(path.read_text()) == report


# ── rescale_global_weights (end-to-end) ────────────────────────────────────

def test_rescale_global_weights_end_to_end():
    rubric = _branch_rubric()
    client = _FakeEmbeddingClient([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    section_map = {
        "branch-a": {"pages": [1, 3], "tables": 0, "figures": 0},
        "branch-b": {"pages": [3, 4], "tables": 0, "figures": 0},
    }
    weights, duplicate_report = pb_embeddings.rescale_global_weights(rubric, client, section_map)
    assert set(weights) == {"a1", "a2", "b1"}
    assert all(w >= 1 for w in weights.values())
    assert isinstance(duplicate_report, list)
    assert len(client.embeddings.calls) == 1  # single batched embedding call


def test_rescale_global_weights_flags_cross_branch_duplicates():
    rubric = _branch_rubric()
    # a1 and b1 embed near-identically -> flagged as a cross-branch duplicate.
    client = _FakeEmbeddingClient([[1.0, 0.0], [0.0, 1.0], [0.999, 0.001]])
    weights, duplicate_report = pb_embeddings.rescale_global_weights(rubric, client, {})
    flagged_ids = {leaf_id for entry in duplicate_report for leaf_id in entry["leaf_ids"]}
    assert "a1" in flagged_ids and "b1" in flagged_ids
