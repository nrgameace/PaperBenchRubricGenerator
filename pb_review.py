"""Human-in-the-loop review: pretty-print, edit a draft file, validate, and retry."""

import json


class RerunPass(Exception):
    """Raised when the user chooses to re-run the LLM pass instead of editing the draft."""


def pretty_print_nodes(title: str, nodes) -> None:
    """Print newly generated nodes for human inspection."""
    print(f"\n=== {title} ===")
    print(json.dumps(nodes, indent=2, ensure_ascii=False))
    print("=== end ===\n")


def _write_draft(path, rubric) -> None:
    """Write the current rubric to the editable draft file."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rubric, handle, indent=2, ensure_ascii=False)


def _read_draft(path):
    """Read the (possibly user-edited) draft file back as JSON."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _prompt_retry_choice(prompt_fn) -> str:
    """Ask whether to re-run the LLM pass ('r') or edit the file again ('e')."""
    while True:
        choice = prompt_fn("Validation failed. [r]e-run the LLM pass or [e]dit the file again? ").strip().lower()
        if choice in ("r", "e"):
            return choice
        print("Please enter 'r' or 'e'.")


def review_pass(rubric, draft_path, validate_fn, input_fn=input) -> dict:
    """Write the draft, wait for edits, validate, and loop; return approved rubric or raise RerunPass."""
    _write_draft(draft_path, rubric)
    while True:
        input_fn(f"Edit {draft_path} as needed, then press Enter to continue...")
        try:
            edited = _read_draft(draft_path)
        except json.JSONDecodeError as exc:
            print(f"Draft is not valid JSON: {exc}")
            continue
        try:
            validate_fn(edited)
        except Exception as exc:
            print(f"Schema validation failed: {exc}")
            if _prompt_retry_choice(input_fn) == "r":
                raise RerunPass()
            continue
        return edited
