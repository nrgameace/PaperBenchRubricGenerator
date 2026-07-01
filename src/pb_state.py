"""State persistence and phase derivation for resumable rubric generation."""

import json
from pathlib import Path

PHASE_DONE = "done"
PHASE_WEIGHT = "weight"
PHASE_EXPANSION = "expansion"
PHASE_BASE = "base"


def empty_state() -> dict:
    """Return a fresh, empty state structure."""
    return {"rubric": None, "queue": [], "hints": {}, "errors": []}


def load_state(path):
    """Load the state file if it exists, else return None."""
    state_path = Path(path)
    if not state_path.exists():
        return None
    with open(state_path, "r", encoding="utf-8") as handle:
        state = json.load(handle)
    state.setdefault("errors", [])
    return state


def save_state(path, rubric, queue, hints, errors=None) -> None:
    """Persist the rubric, expansion queue, hints, and guardrail errors atomically via a temp file."""
    payload = {"rubric": rubric, "queue": list(queue), "hints": dict(hints), "errors": list(errors or [])}
    state_path = Path(path)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    tmp_path.replace(state_path)


def determine_phase(state, final_exists: bool) -> str:
    """Decide which phase to (re)enter from the state and whether the final file exists."""
    if final_exists:
        return PHASE_DONE
    if not state or not state.get("rubric"):
        return PHASE_BASE
    if state.get("queue"):
        return PHASE_EXPANSION
    return PHASE_WEIGHT
