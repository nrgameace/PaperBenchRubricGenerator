"""Human-in-the-loop review: pretty-print, edit a draft file, validate, and retry."""

import json


class RerunPass(Exception):
    """Raised when the user provides feedback to re-run the LLM pass with extra context."""

    def __init__(self, feedback: str = ""):
        self.feedback = feedback
        super().__init__(feedback)


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


def review_pass(rubric, draft_path, validate_fn, input_fn=input) -> dict:
    """Write the draft, wait for human input, validate, and loop.

    Empty Enter → read the draft (which may have been edited), validate, return if valid.
    Non-empty typed text → raise RerunPass(feedback) to regenerate with that feedback.
    """
    _write_draft(draft_path, rubric)
    while True:
        response = input_fn(
            f"Edit {draft_path} as needed, then press Enter to approve"
            " (or type feedback to regenerate): "
        ).strip()
        if response:
            raise RerunPass(response)
        try:
            edited = _read_draft(draft_path)
        except json.JSONDecodeError as exc:
            print(f"Draft is not valid JSON: {exc}")
            continue
        try:
            validate_fn(edited)
        except Exception as exc:
            print(f"Schema validation failed: {exc}")
            continue
        return edited
