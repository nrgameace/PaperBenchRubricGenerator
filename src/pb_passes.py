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

from pb_schema import FINEGRAINED_CATEGORIES, LEAF_CATEGORIES, all_ids, find_node, iter_nodes

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
    """Construct the Anthropic client with the extended-cache-ttl beta enabled."""
    return anthropic.Anthropic(
        default_headers={
            "anthropic-beta": "prompt-caching-2024-07-31,extended-cache-ttl-2025-02-19"
        }
    )


def build_system_blocks(few_shot_json: str) -> list:
    """Build the cached system block: grounding rules plus the few-shot format exemplar."""
    text = f"{SYSTEM_PREAMBLE}\n\n{_EXAMPLE_HEADER}\n{few_shot_json}\n{_EXAMPLE_FOOTER}"
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def pdf_to_block(pdf_path) -> dict:
    """Read a PDF and return a cached base64 document content block."""
    data = Path(pdf_path).read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": encoded},
        "cache_control": {"type": "ephemeral"},
    }


def _human_message(pdf_block: dict, instruction: str) -> dict:
    """Return a user-role message containing the PDF block and a text instruction."""
    return {"role": "user", "content": [pdf_block, {"type": "text", "text": instruction}]}


def _text_message(instruction: str) -> dict:
    """Return a user-role message with a text instruction only (no PDF)."""
    return {"role": "user", "content": [{"type": "text", "text": instruction}]}


def invoke_llm(client, system_blocks, messages, model, max_tokens=8000, tracker=None) -> str:
    """Invoke the model and return its text. Raises RuntimeError on API failure."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=messages,
        )
    except Exception as exc:
        raise RuntimeError(f"Anthropic API call failed: {exc}") from exc
    if tracker is not None:
        tracker.record(model, response.usage)
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
    """Return the substring spanning the first JSON object/array in the text."""
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("No JSON object or array found in model response.")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        raise ValueError("Unbalanced JSON in model response.")
    return text[start : end + 1]


def parse_json_response(text: str):
    """Parse a model response into JSON, tolerating code fences and surrounding prose."""
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return json.loads(_extract_json_span(cleaned))


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
    """Convert a generated child object into a (node_dict, hint_or_None) pair, honoring its id."""
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
    node["task_category"] = raw.get("task_category")
    node["finegrained_task_category"] = raw.get("finegrained_task_category")
    return node, None


def apply_base(parsed: dict) -> tuple:
    """Build the initial rubric, expansion queue, and hints from a base-pass response."""
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
    return rubric, queue, hints


def _children_from_parsed(parsed):
    """Extract the list of child objects from an expansion response."""
    if isinstance(parsed, list):
        return parsed
    return parsed.get("children", [])


def apply_expansion(rubric: dict, node_id: str, parsed, hints: dict) -> list:
    """Attach generated children to ``node_id`` in place; return the new pending ids."""
    target = find_node(rubric, node_id)
    if target is None:
        raise ValueError(f"Node '{node_id}' not found in rubric.")
    target["sub_tasks"] = []
    new_pending = []
    used_ids = set(all_ids(rubric))
    for raw in _children_from_parsed(parsed):
        node, hint = normalize_child(raw, used_ids)
        target["sub_tasks"].append(node)
        if hint is not None:
            new_pending.append(node["id"])
            hints[node["id"]] = hint
    if target["sub_tasks"]:
        target["task_category"] = None
    hints.pop(node_id, None)
    return new_pending


def apply_weights(rubric: dict, weights: dict) -> None:
    """Set an integer weight on every node from an id->weight mapping (default 1 when missing)."""
    weights = weights or {}
    for node in iter_nodes(rubric):
        value = weights.get(node["id"])
        node["weight"] = int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else 1


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
        "Respond with JSON ONLY in exactly this shape:\n"
        '{"root": {"requirements": "<one sentence describing full reproduction of THIS paper>"}, '
        '"children": [ <child objects> ]}\n\n' + _CHILD_SHAPE
    )
    return parse_json_response(
        invoke_llm(client, system_blocks, [_human_message(pdf_block, instruction)], model, tracker=tracker)
    )


def run_expansion_llm(client, system_blocks, section_text, rubric: dict, node_id: str, hint: str, model, feedback: str = "", tracker=None) -> dict:
    """Run an expansion pass for one node; sends only the relevant section text (no PDF)."""
    target = find_node(rubric, node_id)
    instruction = (
        "TASK: Expand ONE node of the in-progress rubric into its DIRECT children, grounded in the "
        "paper. Do not repeat or modify any other node.\n\n"
        f"RELEVANT PAPER SECTION:\n{section_text}\n\n"
        f"NODE TO EXPAND:\n- requirements: {target['requirements']}\n- expansion hint: {hint}\n\n"
        f"FULL RUBRIC SO FAR (context only):\n{json.dumps(rubric, indent=2, ensure_ascii=False)}\n\n"
        'Respond with JSON ONLY: {"children": [ <child objects> ]}\n\n' + _CHILD_SHAPE
    )
    if feedback:
        instruction += f"\n\nUSER FEEDBACK (incorporate this when generating children):\n{feedback}"
    return parse_json_response(
        invoke_llm(client, system_blocks, [_text_message(instruction)], model, tracker=tracker)
    )


def run_weight_llm(client, system_blocks, content_list_text, rubric: dict, model, tracker=None) -> dict:
    """Run the weight pass; sends the full MinerU text only (no PDF)."""
    instruction = (
        "TASK: Assign integer WEIGHTS to every node in the completed rubric, reflecting each item's "
        "importance to reproducing the attached paper.\n\n"
        f"PAPER CONTENT (MinerU structured parse):\n{content_list_text}\n\n"
        "Within each group of sibling nodes, assign non-negative integers for relative importance "
        "(they need NOT sum to any fixed value; the benchmark normalizes within each group). More "
        "important or more effort-intensive siblings get larger integers. Use 1 for the root.\n\n"
        f"FULL RUBRIC (ids included):\n{json.dumps(rubric, indent=2, ensure_ascii=False)}\n\n"
        'Respond with JSON ONLY mapping EVERY node id to an integer: {"weights": {"<id>": <int>, ...}}'
    )
    parsed = parse_json_response(
        invoke_llm(client, system_blocks, [_text_message(instruction)], model, tracker=tracker)
    )
    return parsed.get("weights", parsed) if isinstance(parsed, dict) else {}
