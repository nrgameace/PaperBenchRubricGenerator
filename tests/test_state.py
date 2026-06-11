"""Tests for state persistence and phase derivation."""

import pb_state


def test_empty_state_shape():
    state = pb_state.empty_state()
    assert state == {"rubric": None, "queue": [], "hints": {}}


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "rubric_state.json"
    rubric = {"id": "root", "sub_tasks": []}
    pb_state.save_state(path, rubric, ["a", "b"], {"a": "hint"})
    loaded = pb_state.load_state(path)
    assert loaded == {"rubric": rubric, "queue": ["a", "b"], "hints": {"a": "hint"}}


def test_load_missing_returns_none(tmp_path):
    assert pb_state.load_state(tmp_path / "nope.json") is None


def test_save_state_is_atomic_no_leftover_tmp(tmp_path):
    path = tmp_path / "rubric_state.json"
    pb_state.save_state(path, {"id": "root"}, [], {})
    assert path.exists()
    assert not (tmp_path / "rubric_state.json.tmp").exists()


def test_determine_phase_final_exists_wins():
    assert pb_state.determine_phase({"rubric": {"id": "r"}, "queue": ["x"]}, True) == pb_state.PHASE_DONE


def test_determine_phase_base_when_no_rubric():
    assert pb_state.determine_phase(None, False) == pb_state.PHASE_BASE
    assert pb_state.determine_phase({"rubric": None, "queue": []}, False) == pb_state.PHASE_BASE


def test_determine_phase_expansion_when_queue_nonempty():
    assert pb_state.determine_phase({"rubric": {"id": "r"}, "queue": ["x"]}, False) == pb_state.PHASE_EXPANSION


def test_determine_phase_weight_when_queue_empty():
    assert pb_state.determine_phase({"rubric": {"id": "r"}, "queue": []}, False) == pb_state.PHASE_WEIGHT
