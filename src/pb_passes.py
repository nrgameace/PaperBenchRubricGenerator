"""LLM passes: prompt construction, Anthropic SDK invocation, JSON parsing, node merging.

All network access is funneled through ``invoke_llm`` so the merge/parse helpers stay
pure and unit-testable. Generated nodes are always grounded in the supplied paper; the
few-shot rubric in the system prompt is a format reference only (see SYSTEM_PREAMBLE).
"""

import base64
import json
import re
from pathlib import Path

import anthropic

from pb_enumeration import MIN_ENUMERATED_ITEMS_TO_SPLIT, build_enumeration_hint, count_enumerated_items
from pb_schema import FINEGRAINED_CATEGORIES, LEAF_CATEGORIES, all_ids, find_node, find_parent, iter_nodes, node_depth

MAX_EXPANSION_DEPTH = 7
_DEPTH_FALLBACK_CATEGORY = "Code Development"
MAX_BRANCH_NODES = 40

SYSTEM_PREAMBLE = f"""You are an expert ML research engineer building a PaperBench grading \
rubric for ONE specific paper, supplied as a PDF in the user message.

A rubric is a tree of TaskNodes describing everything required to reproduce the paper from \
scratch. Each node has:
- "requirements": a precise, self-contained statement of one thing that must be true in a \
correct reproduction.
- "sub_tasks": a list of child nodes.
- "task_category": for LEAF nodes only (no sub_tasks), exactly one of \
{list(LEAF_CATEGORIES)}.
- "finegrained_task_category": optional, leaf nodes only, one of {list(FINEGRAINED_CATEGORIES)}.
- "id": a descriptive kebab-case id for every node, following the Node ID Format Rule below.
(weights are assigned automatically in a separate step — do not produce a "weight".)

Content rules:
- Do NOT INCLUDE experiments that just appear in the appendix and aren't referrenced or used at all in the main section of the paper. If an experiment or figure
is referenced in the main paper, you can include that experiment, however, if the only place it exsists is in the appendix, then do not include it in the rubric.

Category meanings:
- "Code Development": the submission contains a correct implementation of this requirement.
- "Code Execution": running the reproduce.sh script successfully executes this.
- "Result Analysis": the reproduce.sh execution produced evidence agreeing with these results.

Structural rules:
- LEAF nodes MUST have a task_category. Internal nodes (with children) MUST NOT.
- Requirements must be specific, verifiable, and tied to concrete details of the paper.

Node ID Format Rule:
- Node IDs must be short, human-readable, and descriptive. Use kebab-case based on the 
content, e.g. "vim-block-forward-ssm" not "the-forward-direction-branch-has-1". 
IDs must be unique across the entire rubric.

Leaf Node Granularity Rule:
- Each leaf node must describe exactly ONE verifiable thing. An expert familiar with 
the paper should be able to verify it in under 15 minutes by reading the submission. 
If a requirement contains "and" connecting two distinct verifiable things, split it 
into two separate leaf nodes.

Framework and Algorithm Naming Rule:
- When referencing external frameworks, algorithms, or libraries by name, verify the 
exact name against the paper text. Do not substitute or paraphrase framework names. 
For example if the paper says "Cascade Mask R-CNN" do not write "ViTDet".

Result Match Specificity Rule:
- Result Analysis nodes must cite the specific table or figure number they correspond 
to, and include the exact metric values reported in the paper. Write "Table 1 reports 
Vim-Ti top-1 accuracy of 76.1%" not "results match the paper's reported values".

Self Check Instruction:
After generating the rubric, perform a self-check pass:
1. For every leaf node, ask: "Does this correspond to exactly one verifiable thing?"
2. For every Result Analysis node, ask: "Did I cross-verify this number against both 
the figure and the table it came from?"
3. For every framework or algorithm name, ask: "Is this the exact name used in the paper?"
4. Flag any discrepancies found in ALL CAPS within the requirements field.

Weight Guidance:
- Weights reflect importance to the paper's core contributions, not implementation 
difficulty. The main claimed contribution should have the highest weight among siblings. 
Negative results and appendix-referenced experiments should have lower weights than 
core positive results.

Table Rules:
For every table in the paper, enumerate ALL rows explicitly. 
Do not selectively include only the rows that compare the 
proposed method against its primary baseline. Every row in 
a results table that is discussed in the paper text must have 
a corresponding Result Analysis node.

GROUNDING — this is critical:
- Every node, requirement, and implementation step MUST come from THIS paper — its methods, \
datasets, models, experiments, metrics, and reported results.
- An example rubric is included below ONLY to illustrate JSON shape, nesting style, category \
usage, and rough depth. It describes a DIFFERENT paper. Do NOT copy, paraphrase, or carry over \
any of its nodes, requirements, or topics.
- If something is not described in this paper, do not invent a node for it. Prefer omission \
over invention.
- If something is added to the rubric it must be fully vetted. Text MUST BE CROSS VERIFIED with any Figures & Tables in the same section or referring to the same topic in the paper to ensure
that they are both referncing the same information. If there is ANY discrepancy FLAG IT and include it in a node in ALL CAPS.


Figure and diagram analysis instructions:
For every figure in the paper, treat it as a separate analysis task before writing any rubric nodes. Do not rely on the caption or surrounding text alone. Instead, work through the figure systematically:

Identify every distinct component shown (boxes, arrows, operators, data paths, labels)
For each component, note its type (operation, tensor, quantization boundary, data format, connection)
Trace every data path from input to output, noting where format changes occur (e.g. FP16 to INT8, or vice versa)
Cross-verify what you read from the figure against the caption, the nearest paragraph, and any equations that reference it. Flag any discrepancy in ALL CAPS.
Only after completing steps 1-4, write the rubric nodes for that figure

For architecture and precision-mapping figures specifically (e.g. block diagrams, quantization flow diagrams), you must enumerate every operator shown and assign it an explicit data format (INT8 or FP16). Do not group operators with vague language like "lightweight ops stay in FP16" — name each one individually and state its format.
Each figure should produce at least one rubric node per distinct verifiable claim visible in the figure. A single node covering an entire figure is almost always a sign the figure was not fully analyzed.
Take as many tokens as needed for this analysis. Accuracy on figures is more important than brevity. Show your figure analysis as a reasoning scratchpad before writing the nodes, using the format:
FIGURE [N] ANALYSIS:
Components: ...
Data paths: ...
Format boundaries: ...
Cross-verification: ...
Discrepancies: ...
Then write the nodes.

Always respond with VALID JSON ONLY — no prose, no markdown code fences."""

_EXAMPLE_HEADER = "===== EXAMPLE RUBRIC (FORMAT / DEPTH REFERENCE ONLY — DIFFERENT PAPER, DO NOT REUSE ITS CONTENT) ====="
_EXAMPLE_FOOTER = "===== END EXAMPLE RUBRIC ====="

_CHILD_SHAPE = """Each child object has this shape (give every node a descriptive kebab-case "id" per the Node ID Format Rule):
{"id": "kebab-case-id", "requirements": "...", "expandable": true,  "expansion_hint": "<one sentence on its future sub-tasks>", "task_category": null, "finegrained_task_category": null}
or, for an atomic leaf requirement:
{"id": "kebab-case-id", "requirements": "...", "expandable": false, "expansion_hint": null, "task_category": "Code Development", "finegrained_task_category": null}"""


def build_client() -> anthropic.Anthropic:
    """Construct the Anthropic client."""
    return anthropic.Anthropic()


def build_system_blocks(few_shot_json: str) -> list:
    """Build the cached system block: grounding rules plus the few-shot format exemplar."""
    text = f"{SYSTEM_PREAMBLE}\n\n{_EXAMPLE_HEADER}\n{few_shot_json}\n{_EXAMPLE_FOOTER}"
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]


def pdf_to_block(pdf_path) -> dict:
    """Read a PDF and return a cached base64 document content block."""
    data = Path(pdf_path).read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": encoded},
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }


def _human_message(pdf_block: dict, instruction: str) -> dict:
    """Return a user-role message containing the PDF block and a text instruction."""
    return {"role": "user", "content": [pdf_block, {"type": "text", "text": instruction}]}


def _text_message(instruction: str) -> dict:
    """Return a user-role message with a text instruction only (no PDF)."""
    return {"role": "user", "content": [{"type": "text", "text": instruction}]}


def invoke_llm(client, system_blocks, messages, model, max_tokens=8000, tracker=None) -> str:
    """Invoke the model and return its text. Raises RuntimeError on API failure.

    Uses the streaming API rather than a blocking ``create`` call: the Anthropic SDK refuses
    a non-streaming request whenever ``max_tokens`` is large enough that the response could
    plausibly take longer than 10 minutes (see run_split_check_llm's scaled max_tokens), and
    streaming is the SDK's documented way to make those calls safely regardless of size.
    """
    try:
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=messages,
        ) as stream:
            response = stream.get_final_message()
    except Exception as exc:
        raise RuntimeError(f"Anthropic API call failed: {exc}") from exc
    if tracker is not None:
        tracker.record(model, response.usage)
        print(f"  Current usage: ${tracker.total_cost():.4f}")
    if getattr(response, "stop_reason", None) == "max_tokens":
        raise RuntimeError(
            f"Anthropic API response for {model} was truncated at max_tokens={max_tokens} "
            "before completing its JSON output. Re-run with a higher max_tokens for this call."
        )
    return response.content[0].text


def _strip_code_fences(text: str) -> str:
    """Remove surrounding markdown code fences from a model response, if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("```", 2)
    inner = body[1] if len(body) >= 2 else stripped
    if inner.startswith("json"):
        inner = inner[4:]
    return inner.strip().rstrip("`").strip()


def _extract_json_span(text: str) -> str:
    """Return the longest balanced JSON object/array found in text.

    Scans every '{'/'[' as a candidate start and keeps the longest span that decodes to
    complete, valid JSON. A naive first-open-to-last-close scan is fooled by stray bracket
    characters in prose that precedes the real JSON (e.g. a model writing out a reasoning
    scratchpad with math notation like "[0.4, 0.6]" or citation numbers before its actual
    answer) — those parse as tiny valid JSON arrays in isolation and can span past the real
    payload's closing bracket. Picking the longest successful decode instead reliably finds
    the intended payload, and naturally prefers an outer object over its own nested
    sub-objects (the outer span is always longer).
    """
    decoder = json.JSONDecoder()
    best_span = None
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if best_span is None or (end - i) > (best_span[1] - best_span[0]):
            best_span = (i, end)
    if best_span is None:
        raise ValueError("No JSON object or array found in model response.")
    start, end = best_span
    return text[start:end]


def parse_json_response(text: str):
    """Parse a model response into JSON, tolerating code fences and surrounding prose.

    Raises ValueError with a preview of the raw response on failure, so a malformed model
    response is diagnosable instead of surfacing as an opaque JSONDecodeError with no context
    on what the model actually returned.
    """
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_extract_json_span(cleaned))
    except (json.JSONDecodeError, ValueError) as exc:
        preview = cleaned[:2000]
        raise ValueError(f"Model response was not valid JSON ({len(cleaned)} chars). Response: {preview!r}") from exc


def clean_id(text: str) -> str:
    """Normalize an id to kebab-case (lowercase, alphanumeric words joined by hyphens)."""
    cleaned = "-".join(re.findall(r"[a-z0-9]+", (text or "").lower()))
    return cleaned or "node"


def slugify(text: str, max_words: int = 8, max_len: int = 60) -> str:
    """Derive a short kebab-case id from requirements text (fallback when the model omits an id)."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    base = "-".join(words[:max_words])[:max_len].rstrip("-")
    return base or "node"


def ensure_unique_id(base: str, used_ids: set) -> str:
    """Keep the model's clean id as-is when unique; append -2, -3, ... only on collision."""
    candidate = base if base not in used_ids else next(f"{base}-{i}" for i in range(2, 10_000) if f"{base}-{i}" not in used_ids)
    used_ids.add(candidate)
    return candidate


def choose_id(raw: dict, used_ids: set) -> str:
    """Use the model-provided id (per the Node ID Format Rule), falling back to a requirements slug."""
    base = clean_id(raw["id"]) if raw.get("id") else slugify(raw.get("requirements", ""))
    return ensure_unique_id(base, used_ids)


def normalize_child(raw: dict, used_ids: set) -> tuple:
    """Convert a generated child object into a (node_dict, hint_or_None) pair, honoring its id.

    A node the model marked non-expandable is still forced into a pending/expandable
    state when its own requirements text enumerates >= MIN_ENUMERATED_ITEMS_TO_SPLIT
    named sub-items (comma-separated lists, "et al." citations, or Table/Figure
    references) — this catches dense leaves like "all 10 environments are ...: a, b,
    c, ..." that should split into one child per item instead of remaining a single
    node.
    """
    node = {
        "id": choose_id(raw, used_ids),
        "requirements": (raw.get("requirements") or "").strip(),
        "weight": 0,
        "sub_tasks": [],
        "task_category": None,
        "finegrained_task_category": None,
    }
    if raw.get("expandable"):
        hint = (raw.get("expansion_hint") or "").strip() or "Expand this node into its sub-tasks based on the paper."
        return node, hint
    enum_count = count_enumerated_items(node["requirements"])
    if enum_count >= MIN_ENUMERATED_ITEMS_TO_SPLIT:
        return node, build_enumeration_hint(raw.get("expansion_hint"), enum_count)
    node["task_category"] = raw.get("task_category")
    node["finegrained_task_category"] = raw.get("finegrained_task_category")
    return node, None


def apply_base(parsed: dict) -> tuple:
    """Build the initial rubric, expansion queue, hints, and section_map from a base-pass response."""
    root_requirements = (parsed.get("root", {}).get("requirements") or "Reproduce the paper from scratch.").strip()
    rubric = {
        "id": "root",
        "requirements": root_requirements,
        "weight": 0,
        "sub_tasks": [],
        "task_category": None,
        "finegrained_task_category": None,
    }
    queue, hints, used_ids = [], {}, {"root"}
    for raw in parsed.get("children", []):
        node, hint = normalize_child(raw, used_ids)
        rubric["sub_tasks"].append(node)
        if hint is not None:
            queue.append(node["id"])
            hints[node["id"]] = hint
    section_map = parsed.get("section_map") or {}
    return rubric, queue, hints, section_map


def _children_from_parsed(parsed):
    """Extract the list of child objects from an expansion response."""
    if isinstance(parsed, list):
        return parsed
    return parsed.get("children", [])


def apply_expansion(rubric: dict, node_id: str, parsed, hints: dict, errors: list = None, max_depth: int = MAX_EXPANSION_DEPTH) -> list:
    """Attach generated children to ``node_id`` in place; return the new pending ids.

    Children that would sit past ``max_depth`` are forced into leaves instead of being
    queued for further expansion, since expansion cost is unbounded otherwise. Each such
    guardrail hit is appended to ``errors`` (if provided) as ``"<id>: ..."``.
    """
    target = find_node(rubric, node_id)
    if target is None:
        raise ValueError(f"Node '{node_id}' not found in rubric.")
    child_depth = node_depth(rubric, node_id) + 1
    target["sub_tasks"] = []
    new_pending = []
    used_ids = set(all_ids(rubric))
    for raw in _children_from_parsed(parsed):
        node, hint = normalize_child(raw, used_ids)
        target["sub_tasks"].append(node)
        if hint is not None and child_depth > max_depth:
            node["task_category"] = _DEPTH_FALLBACK_CATEGORY
            if errors is not None:
                errors.append(f"{node['id']}: Model attempted to expand past the maximum depth of {max_depth} nodes.")
        elif hint is not None:
            new_pending.append(node["id"])
            hints[node["id"]] = hint
    if target["sub_tasks"]:
        target["task_category"] = None
    hints.pop(node_id, None)
    return new_pending


def branch_size(rubric: dict, branch_id: str) -> int:
    """Return the number of nodes currently in branch_id's subtree (including branch_id itself)."""
    branch_root = find_node(rubric, branch_id)
    if branch_root is None:
        raise ValueError(f"Node '{branch_id}' not found in rubric.")
    return len(list(iter_nodes(branch_root)))


def force_branch_cap_leaves(rubric: dict, node_ids: list, branch_id: str, hints: dict,
                             errors: list = None, capped_branches: set = None,
                             max_branch_nodes: int = MAX_BRANCH_NODES) -> None:
    """Force each still-pending node_id (a barren stub already attached to the tree) into a leaf
    because branch_id hit the node cap, rather than expanding it further.

    The cap is checked at BFS-dequeue granularity by the caller (mirroring the depth guardrail's
    style), so a single expansion call that attaches many children at once can push branch_size
    past max_branch_nodes before the next check catches it — expected, not an off-by-one.
    """
    for node_id in node_ids:
        node = find_node(rubric, node_id)
        if node is None:
            continue
        node["task_category"] = _DEPTH_FALLBACK_CATEGORY
        node["sub_tasks"] = []
        hints.pop(node_id, None)
        if errors is not None:
            errors.append(f"{node_id}: Branch '{branch_id}' hit the {max_branch_nodes}-node cap; forced to leaf.")
    if capped_branches is not None:
        capped_branches.add(branch_id)


def apply_weights(rubric: dict, weights: dict) -> None:
    """Set an integer weight on every node from an id->weight mapping.

    Raises ValueError for any node whose weight is missing or invalid.
    Callers must resolve all invalids (via find_invalid_weights) before calling this.
    """
    weights = weights or {}
    for node in iter_nodes(rubric):
        node_id = node["id"]
        value = weights.get(node_id)
        if (
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < 0
        ):
            raise ValueError(f"Node '{node_id}' has an invalid weight: {value!r}.")
        node["weight"] = int(value)


def find_invalid_weights(rubric: dict, raw_weights: dict) -> list:
    """Return (node_id, requirements, raw_value) tuples for nodes with invalid weights.

    Invalid means missing, boolean, non-numeric, or negative. Does not mutate the rubric.
    """
    raw_weights = raw_weights or {}
    invalid = []
    for node in iter_nodes(rubric):
        node_id = node["id"]
        value = raw_weights.get(node_id)
        is_valid = (
            value is not None
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value >= 0
        )
        if not is_valid:
            invalid.append((node_id, node.get("requirements", ""), value))
    return invalid


def run_base_llm(client, system_blocks, pdf_block, content_list_text, model, tracker=None) -> dict:
    """Run the base node pass; sends both the full MinerU text and the PDF for figure analysis."""
    instruction = (
        "TASK: Generate the BASE of the rubric for the attached paper.\n\n"
        "The structured paper content is provided below as parsed text. "
        "Use the attached PDF for figure and diagram analysis.\n\n"
        f"PAPER CONTENT (MinerU structured parse):\n{content_list_text}\n\n"
        "Produce the root node's requirements and the TOP-LEVEL child nodes — the major areas of "
        "work needed to reproduce THIS paper. Most top-level children are expandable (they receive "
        "their own sub-tasks later); give each a one-sentence expansion_hint that names the paper "
        "section or topic it corresponds to (this hint is used to slice the relevant section for expansion).\n\n"
        "Also return a section_map giving, for each top-level child id, its approximate page span "
        "and the number of tables/figures it covers in the paper — use the MinerU structured parse's "
        "page markers and table/figure blocks to determine this.\n\n"
        "Respond with JSON ONLY in exactly this shape:\n"
        '{"root": {"requirements": "<one sentence describing full reproduction of THIS paper>"}, '
        '"children": [ <child objects> ], '
        '"section_map": {"<top-level-child-id>": {"pages": [<start_page_int>, <end_page_int>], '
        '"tables": <int>, "figures": <int>}}}\n\n' + _CHILD_SHAPE
    )
    return parse_json_response(
        invoke_llm(client, system_blocks, [_human_message(pdf_block, instruction)], model, tracker=tracker)
    )


def run_expansion_llm(client, system_blocks, section_text, rubric: dict, node_id: str, hint: str, model, feedback: str = "", tracker=None) -> dict:
    """Run an expansion pass for one node; sends only the relevant section text (no PDF).

    Injects an ENUMERATION GUARDRAIL instruction when the node's own requirements
    enumerate >= MIN_ENUMERATED_ITEMS_TO_SPLIT named sub-items, telling the model to
    produce at least that many children rather than collapsing them.
    """
    target = find_node(rubric, node_id)
    instruction = (
        "TASK: Expand ONE node of the in-progress rubric into its DIRECT children, grounded in the "
        "paper. Do not repeat or modify any other node.\n\n"
        f"RELEVANT PAPER SECTION:\n{section_text}\n\n"
        f"NODE TO EXPAND:\n- requirements: {target['requirements']}\n- expansion hint: {hint}\n\n"
        f"FULL RUBRIC SO FAR (context only):\n{json.dumps(rubric, indent=2, ensure_ascii=False)}\n\n"
        'Respond with JSON ONLY: {"children": [ <child objects> ]}\n\n' + _CHILD_SHAPE
    )
    enum_count = count_enumerated_items(target["requirements"])
    if enum_count >= MIN_ENUMERATED_ITEMS_TO_SPLIT:
        instruction += (
            f"\n\nENUMERATION GUARDRAIL: this node's requirements name at least {enum_count} "
            "distinct sub-items (baselines, model variants, table rows, or ablation settings). "
            f"Produce at least {enum_count} children, one per named item — do not collapse them "
            "into fewer, denser nodes."
        )
    if feedback:
        instruction += f"\n\nUSER FEEDBACK (incorporate this when generating children):\n{feedback}"
    return parse_json_response(
        invoke_llm(client, system_blocks, [_text_message(instruction)], model, tracker=tracker)
    )


def run_weight_llm(client, system_blocks, content_list_text, rubric: dict, model, tracker=None, feedback=None, node_ids=None) -> dict:
    """Run the weight pass; sends the full MinerU text only (no PDF).

    When node_ids is set, the instruction targets only those ids (full rubric still sent for context).
    When feedback is provided, a USER FEEDBACK block is appended.
    """
    if node_ids:
        id_list = ", ".join(f'"{nid}"' for nid in node_ids)
        task_line = (
            f"TASK: Assign integer WEIGHTS to these specific node IDs only: [{id_list}]. "
            "The full rubric is provided for context but only return weights for those IDs.\n\n"
        )
        response_shape = (
            'Respond with JSON ONLY mapping those node ids to integers: '
            '{"weights": {' + ", ".join(f'"{nid}": <int>' for nid in node_ids) + '}}'
        )
    else:
        task_line = (
            "TASK: Assign integer WEIGHTS to every node in the completed rubric, reflecting each item's "
            "importance to reproducing the attached paper.\n\n"
        )
        response_shape = 'Respond with JSON ONLY mapping EVERY node id to an integer: {"weights": {"<id>": <int>, ...}}'
    instruction = (
        task_line
        + f"PAPER CONTENT (MinerU structured parse):\n{content_list_text}\n\n"
        "Within each group of sibling nodes, assign non-negative integers for relative importance "
        "(they need NOT sum to any fixed value; the benchmark normalizes within each group). More "
        "important or more effort-intensive siblings get larger integers. Use 1 for the root.\n\n"
        f"FULL RUBRIC (ids included):\n{json.dumps(rubric, indent=2, ensure_ascii=False)}\n\n"
        + response_shape
    )
    if feedback:
        instruction += f"\n\nUSER FEEDBACK (incorporate this when assigning weights):\n{feedback}"
    parsed = parse_json_response(
        invoke_llm(client, system_blocks, [_text_message(instruction)], model, tracker=tracker)
    )
    return parsed.get("weights", parsed) if isinstance(parsed, dict) else {}


def run_weight_llm_branch(client, system_blocks, content_list_text, rubric: dict, branch_node: dict, model, tracker=None) -> dict:
    """Assign weights to every node in branch_node's subtree (branch_node + all descendants).

    Full rubric is sent for context but only weights for the subtree are requested.
    """
    branch_ids = [node["id"] for node in iter_nodes(branch_node)]
    id_list = ", ".join(f'"{nid}"' for nid in branch_ids)
    response_shape = (
        'Respond with JSON ONLY mapping those node ids to integers: '
        '{"weights": {' + ", ".join(f'"{nid}": <int>' for nid in branch_ids) + '}}'
    )
    instruction = (
        f"TASK: Assign integer WEIGHTS to every node in the subtree rooted at \"{branch_node['id']}\" "
        f"(node IDs: [{id_list}]). The full rubric is provided for context but only return weights for those IDs.\n\n"
        f"PAPER CONTENT (MinerU structured parse):\n{content_list_text}\n\n"
        "Within each group of sibling nodes, assign non-negative integers for relative importance "
        "(they need NOT sum to any fixed value; the benchmark normalizes within each group). More "
        "important or more effort-intensive siblings get larger integers.\n\n"
        f"FULL RUBRIC (context only):\n{json.dumps(rubric, indent=2, ensure_ascii=False)}\n\n"
        + response_shape
    )
    parsed = parse_json_response(
        invoke_llm(client, system_blocks, [_text_message(instruction)], model, tracker=tracker)
    )
    return parsed.get("weights", parsed) if isinstance(parsed, dict) else {}


_SPLIT_CHECK_FINEGRAINED_CATEGORIES = ("Evaluation, Metrics & Benchmarking",)
_SPLIT_CHECK_TASK_CATEGORY = "Result Analysis"


def _split_check_candidates(branch_node: dict, include_all: bool = False) -> list:
    """Return leaves in branch_node's subtree in the split-check scope.

    When include_all is True (a branch that triggered the MAX_BRANCH_NODES cap — the branch
    most likely to contain runaway duplication), every leaf is a candidate regardless of
    category. Otherwise only leaves in the usual Evaluation/Result-Analysis category scope.
    """
    if include_all:
        return [n for n in iter_nodes(branch_node) if not n.get("sub_tasks")]
    return [
        n for n in iter_nodes(branch_node)
        if not n.get("sub_tasks")
        and (n.get("finegrained_task_category") in _SPLIT_CHECK_FINEGRAINED_CATEGORIES
             or n.get("task_category") == _SPLIT_CHECK_TASK_CATEGORY)
    ]


def run_split_check_llm(client, system_blocks, section_text, rubric: dict, branch_node: dict, model,
                        tracker=None, include_all_leaves: bool = False) -> dict:
    """Check leaves in a branch for semantic bundling (splits) and restated duplicate claims.

    Skips the LLM call entirely (returns empty splits/duplicates) when the branch has no leaves
    in scope, since the enumeration regex is the cheap first-tier filter and this second-tier
    semantic check is only worth the tokens where over-bundling actually concentrates.
    """
    leaves = _split_check_candidates(branch_node, include_all=include_all_leaves)
    if not leaves:
        return {"splits": {}, "duplicates": {}}
    max_tokens = min(8000 + 400 * len(leaves), 32000)
    leaf_lines = "\n".join(f'- "{n["id"]}": {n["requirements"]}' for n in leaves)
    instruction = (
        f"TASK: Review these candidate leaves in the subtree rooted at \"{branch_node['id']}\" for "
        "two kinds of problems:\n\n"
        f"{leaf_lines}\n\n"
        f"RELEVANT PAPER SECTION:\n{section_text}\n\n"
        "1. SPLITS — a leaf bundles 2+ independently verifiable claims (named entities, metrics, "
        "or table rows stated in the requirements text) into a single node. Return replacement "
        "child nodes to split it into (one per claim).\n"
        "2. DUPLICATES — a leaf restates a claim already covered by another leaf in this list — "
        "same underlying fact, different wording. Return the id of the leaf it duplicates; it "
        "should be removed rather than split.\n\n"
        "A leaf must not appear in both \"splits\" and \"duplicates\". Leaves that are already "
        "atomic and non-duplicate must NOT appear in the response at all.\n\n"
        "Do NOT include any analysis, cross-referencing notes, or reasoning text in your response "
        "— the system preamble's figure/self-check reasoning instructions do not apply to this "
        "task. Respond with the JSON object ONLY, no text before or after it: "
        '{"splits": {"<leaf_id>": [ <child objects> ], ...}, '
        '"duplicates": {"<leaf_id>": "<id_of_the_leaf_it_duplicates>", ...}}\n\n' + _CHILD_SHAPE
    )
    parsed = parse_json_response(
        invoke_llm(client, system_blocks, [_text_message(instruction)], model, max_tokens=max_tokens, tracker=tracker)
    )
    if not isinstance(parsed, dict):
        return {"splits": {}, "duplicates": {}}
    return {"splits": parsed.get("splits") or {}, "duplicates": parsed.get("duplicates") or {}}


def apply_split(rubric: dict, leaf_id: str, children: list, errors: list = None) -> list:
    """Replace a leaf with the given split children in place; return the new children.

    Reuses normalize_child for each replacement child, so a child that still enumerates
    >= MIN_ENUMERATED_ITEMS_TO_SPLIT items (or that the model marked expandable) is a sign
    the model under-split — it's forced into a valid leaf via the depth-guardrail fallback
    category and logged to errors as a warning rather than silently accepted, since this
    pass runs after expansion is complete and there's no BFS queue left to push it onto.
    """
    leaf = find_node(rubric, leaf_id)
    if leaf is None:
        raise ValueError(f"Node '{leaf_id}' not found in rubric.")
    used_ids = set(all_ids(rubric)) - {leaf_id}
    new_children = []
    for raw in children:
        node, hint = normalize_child(raw, used_ids)
        if hint is not None:
            node["task_category"] = _DEPTH_FALLBACK_CATEGORY
            if errors is not None:
                errors.append(
                    f"{node['id']}: split-check child of '{leaf_id}' still enumerates "
                    f">= {MIN_ENUMERATED_ITEMS_TO_SPLIT} items after splitting (model under-split)."
                )
        new_children.append(node)
    leaf["task_category"] = None
    leaf["finegrained_task_category"] = None
    leaf["sub_tasks"] = new_children
    return new_children


def apply_dedup(rubric: dict, leaf_id: str, duplicate_of_id: str, errors: list = None) -> None:
    """Remove leaf_id from its parent's sub_tasks because it duplicates duplicate_of_id.

    Raises ValueError if leaf_id has no parent (missing id, or the root). If removal leaves the
    parent with no children, an internal node with no children and no task_category is invalid,
    so the parent is force-flattened into a leaf via the same fallback category the depth and
    branch-cap guardrails use, with its own logged errors line.
    """
    parent = find_parent(rubric, leaf_id)
    if parent is None:
        raise ValueError(f"Node '{leaf_id}' not found in rubric (or is the root).")
    parent["sub_tasks"] = [n for n in parent["sub_tasks"] if n.get("id") != leaf_id]
    if errors is not None:
        errors.append(f"{leaf_id}: split-check removed as duplicate of '{duplicate_of_id}'.")
    if not parent["sub_tasks"]:
        parent["task_category"] = _DEPTH_FALLBACK_CATEGORY
        if errors is not None:
            errors.append(
                f"{parent['id']}: forced to leaf after its last child '{leaf_id}' was removed as a duplicate."
            )


