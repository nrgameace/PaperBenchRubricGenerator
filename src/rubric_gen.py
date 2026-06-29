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

from pb_cost import CostTracker
from pb_input import discover_mineru_dir, discover_pdf, load_content_list
from pb_mineru import blocks_to_text, slice_section
from pb_passes import (apply_base, apply_expansion, apply_weights, build_client, build_system_blocks,
                       pdf_to_block, run_base_llm, run_expansion_llm, run_weight_llm)
from pb_review import RerunPass, pretty_print_nodes, review_pass
from pb_schema import all_ids, find_node, validate_final, validate_partial
from pb_state import (PHASE_BASE, PHASE_DONE, PHASE_EXPANSION, PHASE_WEIGHT, determine_phase,
                      empty_state, load_state, save_state)

HERE = Path(__file__).resolve().parent
OPUS = "claude-opus-4-8"
SONNET = "claude-sonnet-4-6"
FEW_SHOT_RUBRIC_PATH = Path(os.environ.get("FEW_SHOT_RUBRIC_PATH", HERE.parent / "examples" / "example_rubric.json"))


def parse_args():
    """Parse --input and --output directory flags plus the optional --resume flag."""
    parser = argparse.ArgumentParser(description="Generate a PaperBench rubric from a paper PDF.")
    parser.add_argument("--input", required=True, help="Path to the input directory (contains PDF and MinerU folder).")
    parser.add_argument("--output", required=True, help="Path to the output directory (receives state and final rubric).")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing rubric_state.json in the output dir.")
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


def commit(state: dict, output_dir: Path) -> None:
    """Persist the current state to the checkpoint file in output_dir."""
    save_state(output_dir / "rubric_state.json", state["rubric"], state["queue"], state["hints"])


def run_base_phase(client, system_blocks, pdf_block, content_list, state, model, output_dir, tracker=None) -> None:
    """Generate, review, and save the base nodes."""
    print("\n>>> BASE NODE PASS: generating root and top-level nodes...")
    content_list_text = blocks_to_text(content_list)
    feedback = ""
    while True:
        rubric, queue, hints = apply_base(run_base_llm(client, system_blocks, pdf_block, content_list_text, model, tracker=tracker))
        pretty_print_nodes("Generated base nodes", rubric["sub_tasks"])
        try:
            approved = review_pass(rubric, output_dir / "rubric_draft.json", lambda candidate: validate_partial(candidate, queue))
        except RerunPass as e:
            feedback = e.feedback
            print("Re-running base pass...")
            continue
        state["rubric"], state["hints"] = approved, hints
        state["queue"] = reconcile_queue(approved, queue)
        prune_hints(state)
        commit(state, output_dir)
        return


def _expand_subtree(client, system_blocks, content_list, rubric, node_id, hints, model, feedback: str = "", tracker=None) -> None:
    """Fully expand node_id and all its expandable descendants in-place (BFS, no review pause)."""
    local_queue = [node_id]
    while local_queue:
        current_id = local_queue.pop(0)
        hint = hints.get(current_id, "Expand this node into its sub-tasks based on the paper.")
        section_text = blocks_to_text(slice_section(content_list, hint))
        print(f"  Expanding '{current_id}'...")
        parsed = run_expansion_llm(client, system_blocks, section_text, rubric, current_id, hint, model, feedback=feedback, tracker=tracker)
        new_pending = apply_expansion(rubric, current_id, parsed, hints)
        local_queue.extend(new_pending)


def run_expansion_phase(client, system_blocks, content_list, state, model, output_dir, tracker=None) -> None:
    """Expand each top-level node's full subtree, then review once before moving to the next."""
    while state["queue"]:
        node_id = state["queue"][0]
        target = find_node(state["rubric"], node_id)
        print(f"\n>>> EXPANDING SUBTREE: '{node_id}' — {target['requirements'][:70]}")
        feedback = ""
        while True:
            candidate = copy.deepcopy(state["rubric"])
            candidate_hints = dict(state["hints"])
            _expand_subtree(client, system_blocks, content_list, candidate, node_id, candidate_hints, model, feedback, tracker=tracker)
            remaining_queue = state["queue"][1:]
            pretty_print_nodes(f"Subtree '{node_id}' (fully expanded)", [find_node(candidate, node_id)])
            try:
                approved = review_pass(candidate, output_dir / "rubric_draft.json",
                                       lambda c: validate_partial(c, remaining_queue))
            except RerunPass as e:
                feedback = e.feedback
                print(f"Re-running subtree expansion with feedback...")
                continue
            state["rubric"] = approved
            state["hints"] = candidate_hints
            state["queue"] = reconcile_queue(approved, remaining_queue)
            prune_hints(state)
            commit(state, output_dir)
            break


def run_weight_phase(client, system_blocks, content_list, state, model, output_dir, tracker=None) -> dict:
    """Assign, review, and save integer weights across the whole tree."""
    print("\n>>> WEIGHT PASS: assigning integer weights to every node...")
    content_list_text = blocks_to_text(content_list)
    feedback = ""
    while True:
        candidate = copy.deepcopy(state["rubric"])
        apply_weights(candidate, run_weight_llm(client, system_blocks, content_list_text, state["rubric"], model, tracker=tracker))
        pretty_print_nodes("Weighted rubric", candidate)
        try:
            approved = review_pass(candidate, output_dir / "rubric_draft.json", validate_final)
        except RerunPass as e:
            feedback = e.feedback
            print("Re-running weight pass...")
            continue
        state["rubric"], state["queue"] = approved, []
        commit(state, output_dir)
        return approved


def finalize(rubric: dict, output_dir: Path) -> None:
    """Validate the completed rubric and write rubric_final.json to output_dir."""
    validate_final(rubric)
    final_file = output_dir / "rubric_final.json"
    with open(final_file, "w", encoding="utf-8") as handle:
        json.dump(rubric, handle, indent=2, ensure_ascii=False)
    print(f"\nFinal rubric written to {final_file} ({len(all_ids(rubric))} nodes) and validated.")


def main() -> None:
    """Drive the rubric-generation pipeline, resuming if requested."""
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = discover_pdf(input_dir)
    state_file = output_dir / "rubric_state.json"
    final_file = output_dir / "rubric_final.json"

    state = load_state(state_file) if args.resume else None
    if state is None:
        if not args.resume and state_file.exists():
            print(f"Note: {state_file} exists but --resume was not passed; starting fresh and overwriting it.")
        state = empty_state()

    phase = determine_phase(state, final_file.exists() and args.resume)
    if phase == PHASE_DONE:
        print(f"{final_file} already exists; nothing to do. Delete it or omit --resume to start over.")
        return

    client = build_client()
    system_blocks = build_system_blocks(load_few_shot())
    pdf_block = pdf_to_block(pdf_path)
    mineru_dir = discover_mineru_dir(input_dir)
    content_list = load_content_list(mineru_dir)
    tracker = CostTracker()

    if phase == PHASE_BASE:
        run_base_phase(client, system_blocks, pdf_block, content_list, state, OPUS, output_dir, tracker)
        phase = PHASE_EXPANSION
    if phase == PHASE_EXPANSION:
        run_expansion_phase(client, system_blocks, content_list, state, SONNET, output_dir, tracker)
        phase = PHASE_WEIGHT
    if phase == PHASE_WEIGHT:
        finalize(run_weight_phase(client, system_blocks, content_list, state, SONNET, output_dir, tracker), output_dir)
    tracker.print_report()


if __name__ == "__main__":
    main()
