"""Cross-model coverage judge for the rubric pipeline's optional JUDGE checkpoint.

Uses OpenAI (o3-mini), not Anthropic, so its failure modes are uncorrelated with the
Claude models that generated the rubric being judged. Reads a branch's source section
text plus its generated leaves and judges whether anything the paper states is not
represented by any leaf — catching under-expansion (missed claims, dropped
figure/table sub-items), which the split-check pass never catches since split-check
only looks for over-bundling.

Fully self-contained: no dependency on pb_passes.SYSTEM_PREAMBLE or the shared Claude
system_blocks, since this is a different provider running a narrow, unrelated task.
"""

from types import SimpleNamespace

from pb_passes import parse_json_response

_RESULT_ANALYSIS_CATEGORIES = ("Evaluation, Metrics & Benchmarking", "Result Analysis")


def _usage_shim(usage) -> SimpleNamespace:
    """Adapt an OpenAI ChatCompletion usage object into the attribute names
    pb_cost.CostTracker.record reads via getattr (input_tokens/output_tokens/
    cache_creation_input_tokens/cache_read_input_tokens), so record() itself never
    needs to know about a second provider's usage schema."""
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    return SimpleNamespace(
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=cached,
    )


def _invoke_judge_llm(client, instruction: str, model: str, max_completion_tokens: int = 4000, tracker=None) -> str:
    """Sole OpenAI call chokepoint for the judge pass. Mirrors pb_passes.invoke_llm's
    defensive shape (RuntimeError on API failure, truncation check, robust text
    extraction, cost tracking) but is independent of it — different provider, different
    SDK surface, no shared system_blocks."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": instruction}],
            max_completion_tokens=max_completion_tokens,
            reasoning_effort="medium",
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI API call failed: {exc}") from exc
    if tracker is not None:
        tracker.record(model, _usage_shim(response.usage))
        print(f"  Current usage: ${tracker.total_cost():.4f}")
    choice = response.choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        raise RuntimeError(
            f"OpenAI API response for {model} was truncated at max_completion_tokens="
            f"{max_completion_tokens} before completing its JSON output. Re-run with a "
            "higher max_completion_tokens for this call."
        )
    content = getattr(choice.message, "content", None)
    if not content:
        raise RuntimeError(f"OpenAI API response for {model} contained no message content.")
    return content


def _category_guidance(leaves: list) -> str:
    """Pick category-aware framing for what 'missing' should mean, based on the
    majority task_category/finegrained_task_category across this branch's leaves."""
    result_analysis_count = sum(
        1 for n in leaves
        if n.get("finegrained_task_category") in _RESULT_ANALYSIS_CATEGORIES
        or n.get("task_category") == "Result Analysis"
    )
    if result_analysis_count > len(leaves) / 2:
        return (
            "This branch is Result-Analysis/Evaluation-heavy: a 'missing' claim should mean an "
            "uncovered reported value — a number, metric, or table row stated in the section "
            "that no leaf's requirements text captures."
        )
    return (
        "This branch is Code-Development-heavy: a 'missing' claim should mean an uncovered "
        "named step, component, hyperparameter, or architectural piece the section states is "
        "required that no leaf's requirements text captures."
    )


def run_coverage_judge(client, section_text: str, branch_errors_context: str, leaves: list,
                       model: str = "o3-mini", tracker=None) -> dict:
    """One call per top-level branch. Judges whether section_text states anything not
    represented by any of leaves' requirements text.

    Skips the API call entirely (returns {"verdict": "sufficient", "missing": []}) when
    the branch has zero leaves — that's the expansion/split-check guardrails' problem,
    not this pass's.

    The returned verdict is derived from the filtered "missing" list, not trusted
    verbatim from the model's own "verdict" field: entries missing a claim or evidence
    field are dropped silently (grounding requirement — a "missing" item must cite
    evidence from section_text, not general paper knowledge) rather than triggering a
    retry round-trip, consistent with never looping on a malformed sub-field.
    """
    if not leaves:
        return {"verdict": "sufficient", "missing": []}

    max_completion_tokens = min(4000 + 200 * len(leaves), 16000)
    leaf_lines = "\n".join(f'- "{n["id"]}": {n["requirements"]}' for n in leaves)
    reorg_block = (
        f"\n\nKNOWN REORGANIZATION NOTES (do not flag these as missing — they were "
        f"intentionally restructured by an earlier pass):\n{branch_errors_context}"
        if branch_errors_context else ""
    )
    instruction = (
        "You are an independent auditor checking a grading rubric for completeness against a "
        "paper section. Do not apply any figure-analysis, self-check-scratchpad, or other "
        "rubric-generation conventions — this is a standalone coverage-check task with its own "
        "rules below.\n\n"
        "Only judge claims stated in THIS section. Do not flag anything not explicitly stated "
        "in the text below, even if it would be true of the paper as a whole — no scope creep "
        "beyond this section.\n\n"
        f"{_category_guidance(leaves)}\n\n"
        f"EXISTING RUBRIC LEAVES FOR THIS BRANCH:\n{leaf_lines}\n\n"
        f"RELEVANT PAPER SECTION:\n{section_text}"
        f"{reorg_block}\n\n"
        "Respond with the JSON object ONLY, no text before or after it: "
        '{"verdict": "sufficient" | "insufficient", "missing": [{"claim": "<specific paper '
        'statement not covered by any leaf>", "evidence": "<verbatim or near-verbatim quote '
        'from the section text above>"}]}. missing must be empty when verdict is "sufficient". '
        "Every missing entry must cite evidence from the section text above, not general paper "
        "knowledge."
    )
    raw = _invoke_judge_llm(client, instruction, model, max_completion_tokens=max_completion_tokens, tracker=tracker)
    parsed = parse_json_response(raw)
    if not isinstance(parsed, dict):
        return {"verdict": "sufficient", "missing": []}
    missing = [
        m for m in (parsed.get("missing") or [])
        if isinstance(m, dict) and m.get("claim") and m.get("evidence")
    ]
    return {"verdict": "insufficient" if missing else "sufficient", "missing": missing}


def format_missing_as_feedback(missing: list) -> str:
    """Turn a judge's "missing" list into the same kind of free-text feedback string
    --review mode's typed RerunPass feedback already feeds into _expand_subtree, so
    re-expansion reuses the existing feedback-forwarding code path instead of a new one.
    """
    if not missing:
        return ""
    lines = "\n".join(f'- {m["claim"]} (evidence: "{m["evidence"]}")' for m in missing)
    return (
        "The following claims from the paper section are not represented by any existing "
        f"leaf and must be covered when re-expanding this branch:\n{lines}"
    )
