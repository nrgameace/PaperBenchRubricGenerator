#!/usr/bin/env python3
"""Generate a PaperBench rubric from a paper PDF via human-reviewed LLM passes.

Pipeline:
  1. Base node pass    — root + top-level nodes (each reviewed, then saved).
  2. Expansion passes  — breadth-first expansion of every non-leaf node.
  3. Weight pass       — integer weights per sibling group across the whole tree.

State is checkpointed to rubric_state.json after every approved pass so a run can be
resumed with --resume. The final, validated rubric is written to rubric_final.json.
"""

import argparse
import copy
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pb_passes import (apply_base, apply_expansion, apply_weights, build_llm, build_system_message,
                       pdf_to_block, run_base_llm, run_expansion_llm, run_weight_llm)
from pb_review import RerunPass, pretty_print_nodes, review_pass
from pb_schema import all_ids, find_node, validate_final, validate_partial
from pb_state import (PHASE_BASE, PHASE_DONE, PHASE_EXPANSION, PHASE_WEIGHT, determine_phase,
                      empty_state, load_state, save_state)

HERE = Path(__file__).resolve().parent
MODEL = "claude-opus-4-8"
MAX_TOKENS = 8000
FEW_SHOT_RUBRIC_PATH = Path(os.environ.get("FEW_SHOT_RUBRIC_PATH", HERE / "examples" / "example_rubric.json"))
STATE_FILE = Path("rubric_state.json")
DRAFT_FILE = Path("rubric_draft.json")
FINAL_FILE = Path("rubric_final.json")


def parse_args():
    """Parse the PDF path and the optional --resume flag."""
    parser = argparse.ArgumentParser(description="Generate a PaperBench rubric from a paper PDF.")
    parser.add_argument("pdf_path", help="Path to the paper PDF.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing rubric_state.json.")
    return parser.parse_args()


def load_few_shot() -> str:
    """Load the example rubric used purely as a format/depth exemplar."""
    if not FEW_SHOT_RUBRIC_PATH.exists():
        raise SystemExit(f"Few-shot rubric not found: {FEW_SHOT_RUBRIC_PATH}\n"
                         f"Set the FEW_SHOT_RUBRIC_PATH environment variable to an existing example rubric.")
    with open(FEW_SHOT_RUBRIC_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


def reconcile_queue(rubric: dict, queue) -> list:
    """Drop queue ids that no longer need expansion after a human edit, preserving BFS order."""
    result = []
    for node_id in queue:
        node = find_node(rubric, node_id)
        if node is None or node.get("sub_tasks") or node.get("task_category"):
            continue
        result.append(node_id)
    return result


def prune_hints(state: dict) -> None:
    """Keep only hints whose node is still queued for expansion."""
    queued = set(state["queue"])
    state["hints"] = {key: value for key, value in state["hints"].items() if key in queued}


def commit(state: dict) -> None:
    """Persist the current state to the checkpoint file."""
    save_state(STATE_FILE, state["rubric"], state["queue"], state["hints"])


def run_base_phase(llm, system_message, pdf_block, state) -> None:
    """Generate, review, and save the base nodes."""
    print("\n>>> BASE NODE PASS: generating root and top-level nodes...")
    while True:
        rubric, queue, hints = apply_base(run_base_llm(llm, system_message, pdf_block))
        pretty_print_nodes("Generated base nodes", rubric["sub_tasks"])
        try:
            approved = review_pass(rubric, DRAFT_FILE, lambda candidate: validate_partial(candidate, queue))
        except RerunPass:
            print("Re-running base pass...")
            continue
        state["rubric"], state["hints"] = approved, hints
        state["queue"] = reconcile_queue(approved, queue)
        prune_hints(state)
        commit(state)
        return


def run_expansion_phase(llm, system_message, pdf_block, state) -> None:
    """Breadth-first expand every queued node, reviewing and saving each pass."""
    while state["queue"]:
        node_id = state["queue"][0]
        target = find_node(state["rubric"], node_id)
        hint = state["hints"].get(node_id, "Expand this node into its sub-tasks based on the paper.")
        print(f"\n>>> EXPANSION PASS: '{node_id}' — {target['requirements'][:70]}")
        _expand_one(llm, system_message, pdf_block, state, node_id, hint)


def _expand_one(llm, system_message, pdf_block, state, node_id, hint) -> None:
    """Run, review, and save a single node's expansion pass."""
    while True:
        parsed = run_expansion_llm(llm, system_message, pdf_block, state["rubric"], node_id, hint)
        candidate = copy.deepcopy(state["rubric"])
        candidate_hints = dict(state["hints"])
        new_pending = apply_expansion(candidate, node_id, parsed, candidate_hints)
        remaining = [q for q in state["queue"] if q != node_id] + new_pending
        pretty_print_nodes(f"Children of '{node_id}'", find_node(candidate, node_id)["sub_tasks"])
        try:
            approved = review_pass(candidate, DRAFT_FILE, lambda c: validate_partial(c, remaining))
        except RerunPass:
            print("Re-running expansion pass...")
            continue
        state["rubric"], state["hints"] = approved, candidate_hints
        state["queue"] = reconcile_queue(approved, remaining)
        prune_hints(state)
        commit(state)
        return


def run_weight_phase(llm, system_message, pdf_block, state) -> dict:
    """Assign, review, and save integer weights across the whole tree."""
    print("\n>>> WEIGHT PASS: assigning integer weights to every node...")
    while True:
        candidate = copy.deepcopy(state["rubric"])
        apply_weights(candidate, run_weight_llm(llm, system_message, pdf_block, state["rubric"]))
        pretty_print_nodes("Weighted rubric", candidate)
        try:
            approved = review_pass(candidate, DRAFT_FILE, validate_final)
        except RerunPass:
            print("Re-running weight pass...")
            continue
        state["rubric"], state["queue"] = approved, []
        commit(state)
        return approved


def finalize(rubric: dict) -> None:
    """Validate the completed rubric and write rubric_final.json."""
    validate_final(rubric)
    with open(FINAL_FILE, "w", encoding="utf-8") as handle:
        json.dump(rubric, handle, indent=2, ensure_ascii=False)
    print(f"\nFinal rubric written to {FINAL_FILE} ({len(all_ids(rubric))} nodes) and validated.")


def main() -> None:
    """Drive the rubric-generation pipeline, resuming if requested."""
    args = parse_args()
    if not Path(args.pdf_path).exists():
        raise SystemExit(f"PDF not found: {args.pdf_path}")

    state = load_state(STATE_FILE) if args.resume else None
    if state is None:
        if not args.resume and STATE_FILE.exists():
            print(f"Note: {STATE_FILE} exists but --resume was not passed; starting fresh and overwriting it.")
        state = empty_state()

    phase = determine_phase(state, FINAL_FILE.exists() and args.resume)
    if phase == PHASE_DONE:
        print(f"{FINAL_FILE} already exists; nothing to do. Delete it or omit --resume to start over.")
        return

    system_message = build_system_message(load_few_shot())
    pdf_block = pdf_to_block(args.pdf_path)
    llm = build_llm(MODEL, MAX_TOKENS)

    if phase == PHASE_BASE:
        run_base_phase(llm, system_message, pdf_block, state)
        phase = PHASE_EXPANSION
    if phase == PHASE_EXPANSION:
        run_expansion_phase(llm, system_message, pdf_block, state)
        phase = PHASE_WEIGHT
    if phase == PHASE_WEIGHT:
        finalize(run_weight_phase(llm, system_message, pdf_block, state))


if __name__ == "__main__":
    main()
