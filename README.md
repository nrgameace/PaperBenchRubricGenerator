# PaperBenchRubricGenerator

Generate a [PaperBench](https://github.com/openai/preparedness)-style grading **rubric**
from a research paper PDF, so you can measure how well a codebase reproduces that paper.

The tool reads a paper and its [MinerU](https://github.com/opendatalab/MinerU) structured
parse, then drives Claude through a sequence of human-reviewed passes to build a tree of
`TaskNode`s — a hierarchy of precise, verifiable requirements that a faithful reproduction
of the paper must satisfy. You review and edit the model's output at every step.

> **Cost:** A full rubric for one paper costs roughly **~$4 USD** in Anthropic API usage.
> Costs are kept low via extended prompt caching (1-hour TTL on the PDF and system preamble)
> and model tiering (Opus only for the base pass; Sonnet for all expansion and weight passes).

The pipeline supports two modes: **agentic** (default — no prompts, runs end-to-end unattended) and
**human-in-the-loop** (`--review` flag — pauses after each pass for editing and feedback).

---

## How it works

The rubric is built as a tree and checkpointed after every approved pass, so a run is fully
resumable. There are three phases:

1. **Base pass** — `claude-opus-4-8` receives the full MinerU text plus the raw PDF (for
   figure analysis). Produces the root node and the top-level areas of work. Each top-level
   child is tagged as expandable (with an expansion hint naming the relevant paper section)
   or a leaf.

2. **Expansion passes** — `claude-sonnet-4-6`, breadth-first. For each top-level node, the
   tool fully expands its entire subtree (all depths) before pausing for review — **one review
   per top-level section**, not one per node. The MinerU content is sliced to the matching
   section using `difflib` fuzzy heading match; falls back to the full content list when no
   heading matches (similarity < 0.3).

   **Max depth guardrail.** No node may sit deeper than `MAX_EXPANSION_DEPTH` (7; root = depth
   0, top-level nodes = depth 1) — this bounds worst-case expansion cost. If the model marks a
   node at depth 8+ as expandable, the guardrail silently forces it into a valid leaf instead
   (fallback `task_category` of `Code Development`) and records the violation. See
   [Output files](#output-files) for `errors.txt`.

3. **Weight pass** — `claude-sonnet-4-6`, two sub-phases:
   - **Local passes** — one LLM call per top-level branch. Each call is focused on a single
     subtree so the model can reason about relative importance within that branch without
     distraction from unrelated sections. The full rubric is still sent for context.
   - **Global calibration** — one final LLM call that receives all locally-assigned weights
     and adjusts them for consistent relative importance across top-level siblings.
   - The root node is always set to weight 1 by the orchestrator (no LLM call).
   - Invalid weights (negative, non-numeric, boolean, or missing) are corrected after both
     sub-phases: in `--review` mode you are prompted per-node to enter a weight manually or
     queue it for an LLM retry; in agentic mode all invalid nodes are automatically re-queued.
   - **Max retry guardrail.** Weight correction is capped at `MAX_WEIGHT_RESOLUTION_RETRIES`
     (5) retry cycles, in both `--review` and agentic mode. If nodes are still invalid after 5
     retries, the run raises `MaxRetriesExceeded`, which `main()` turns into a clean
     `SystemExit` (no stack trace) telling you to re-run with `--resume` — it never spins
     forever burning tokens on a model that keeps returning bad weights.
   - Human review happens **once**, after both sub-phases and all invalid-weight correction are
     complete.

### Prompt caching

Two blocks are cached with a 1-hour TTL via the Anthropic extended-cache-ttl beta
(`prompt-caching-2024-07-31,extended-cache-ttl-2025-02-19`):

- **System preamble** — grounding rules and the few-shot format exemplar.
- **PDF document block** — base64-encoded PDF, sent only in the base pass.

The cache survives the human-review window between phases, so you don't pay to re-send the
PDF or system prompt on expansion and weight calls.

### Human-in-the-loop review (`--review` only)

After each pass the candidate rubric is written to `rubric_draft.json`. The tool pauses and
prompts you to edit that file as needed:

- **Press Enter** — your edits are validated against the PaperBench `TaskNode` schema. If
  valid, the draft is committed and the next pass begins.
- **Type any text + Enter** — your text is forwarded to the LLM as feedback and that pass
  reruns with the extra context. State is NOT checkpointed until you press Enter with no text.

State is saved atomically to `rubric_state.json` after every approved pass.

### Agentic mode (default)

Without `--review` the pipeline runs end-to-end without any prompts. Each pass accepts the
first LLM output directly. Invalid weights are silently re-queued for an LLM retry until all
are valid. State is still checkpointed after each phase so a run is resumable with `--resume`.

---

## Architecture

### Directory layout

```
data/
  input/<paper>/
    paper.pdf
    mineru_out/           ← any folder name; must contain content_list.json at its root
      content_list.json
  output/<paper>/
    rubric_state.json     ← resumable checkpoint (rubric + expansion queue + hints + errors)
    rubric_draft.json     ← editable draft for the current pass
    rubric_final.json     ← final validated rubric (written when all phases complete)
    errors.txt             ← guardrail violations, if any occurred (see Output files)
examples/
  example_rubric.json     ← few-shot format/depth exemplar (different paper)
tests/
```

### Module responsibilities

| File | Responsibility |
|---|---|
| `rubric_gen.py` | Entry point; CLI parsing (`--review`, `--resume`); orchestrates all 3 phases; `human_review` flag threaded through phase functions; `_resolve_invalid_weights` correction loop, capped at `MAX_WEIGHT_RESOLUTION_RETRIES` (5) and raising `MaxRetriesExceeded` past that; `write_error_log` writes `errors.txt`; prints cost report |
| `pb_passes.py` | Anthropic SDK calls; prompt construction; JSON parsing; node normalization; tree mutation (`apply_base`, `apply_expansion`, `apply_weights`); `MAX_EXPANSION_DEPTH` (7) guardrail enforced in `apply_expansion`; weight validation (`find_invalid_weights`); `run_weight_llm_branch` for per-branch local passes; `run_weight_llm_global` for cross-branch calibration; `run_weight_llm` for targeted invalid-weight retries |
| `pb_cost.py` | `CostTracker` — accumulates token usage (input, output, cache write, cache read) per model; computes and prints a formatted cost report |
| `pb_input.py` | Discovers PDF and MinerU folder from input dir; loads `content_list.json` |
| `pb_mineru.py` | Converts MinerU blocks to LLM-readable text (`blocks_to_text`); slices content to a section by heading fuzzy-match (`slice_section`) |
| `pb_schema.py` | Rubric dict traversal and validation (`validate_partial`, `validate_final`, `find_node`, `all_ids`, `node_depth`) |
| `pb_state.py` | State persistence; phase constants (`PHASE_BASE → EXPANSION → WEIGHT → DONE`); atomic write via temp-file rename; state includes an `errors` list of guardrail violations |
| `pb_review.py` | Blocks on `input()` for human review; raises `RerunPass(feedback)` when user types non-empty text; `collect_weight_corrections` handles interactive per-node weight correction |
| `task_node.py` | Frozen `TaskNode` dataclass (adapted from OpenAI's frontier-evals); leaf/internal validation in `__post_init__` |

### TaskNode rules

- **Leaf nodes**: `task_category` must be one of `Code Development`, `Code Execution`,
  `Result Analysis`; `sub_tasks` must be empty.
- **Internal nodes**: `task_category` must be `None`; `sub_tasks` must be non-empty.
- All node IDs must be unique within the tree and in kebab-case.
- `weight` is 0 during base/expansion phases; set to a non-negative integer by the weight pass.

### MinerU block format

`content_list.json` is a flat list of blocks. Relevant fields:

- `type`: `"text"`, `"image"`, `"table"`, `"equation"`
- `text_level` (int, text blocks only): present on headings; 1 = top-level section
- `img_caption`, `table_body`, `table_caption`

Section slicing: `difflib.SequenceMatcher` fuzzy-matches the expansion hint against heading
text. Falls back to the full list when the best heading score is below 0.3.

---

## Changelog

### v8 — Max-retry cap on weight resolution

**`MAX_WEIGHT_RESOLUTION_RETRIES` (5).** `_resolve_invalid_weights` previously looped
`while True:` with no bound until every node had a valid weight — if the model kept
returning invalid weights for the same node(s), agentic mode (and a human who always
chose "retry via LLM") could spin forever. The loop now counts retry cycles and raises
`MaxRetriesExceeded` after 5, in both agentic and `--review` modes.

**Clean failure, not a crash.** `main()` catches `MaxRetriesExceeded` around the weight
phase and re-raises it as `SystemExit` with a message naming the still-invalid node ids
and telling you to `--resume` — no raw stack trace. Since `commit()` only runs at the end
of `run_weight_phase` (after approval), no state is lost by this failure; `--resume`
simply restarts weight assignment from the last good checkpoint (the fully-expanded
rubric).

### v7 — Max expansion depth guardrail and errors.txt

**Depth cap.** `apply_expansion` now enforces `MAX_EXPANSION_DEPTH` (7, root = depth 0,
top-level = depth 1) via the new `pb_schema.node_depth` helper. If the model marks a node
past that depth as expandable, the node is forced into a valid leaf (`task_category` falls
back to `Code Development`) instead of being queued for further expansion, so the run
never becomes unbounded in cost.

**`errors.txt`.** Each guardrail violation is recorded as `"<node-id>: Model attempted to
expand past the maximum depth of 7 nodes."`. Violations are threaded through
`_expand_subtree`/`run_expansion_phase` the same way expansion hints are, persisted in
`state["errors"]` across `--resume` runs, and written to `errors.txt` next to
`rubric_final.json` at the end of the run — only if at least one violation occurred.

### v6 — Two-phase weight generation (local + global)

**Per-branch local weight passes.** The single global weight LLM call is replaced by one focused
call per top-level branch (`run_weight_llm_branch`). Each call targets only the nodes in that
subtree, giving the model a tighter context so it can reason about relative importance within a
branch without interference from unrelated sections. The full rubric is still included for reference.

**Global calibration pass.** After all local passes complete, a single calibration call
(`run_weight_llm_global`) reviews the complete weighted tree and adjusts weights across branches
for consistent relative importance at the top level.

**Root always 1.** The root node weight is set to 1 directly by the orchestrator and is never
passed to the LLM. If the global pass returns a different value it is overridden.

**Single review point.** Human review (in `--review` mode) still occurs exactly once — after
both the local and global sub-phases and all invalid-weight correction are complete. Feedback
typed at that review is forwarded to the global calibration call on rerun.

### v5 — Agentic mode and `--review` flag

**`--review` flag.** Human-in-the-loop review is now opt-in. Pass `--review` to get the
existing interactive behavior (edit `rubric_draft.json`, provide feedback, manual weight
correction). Omitting `--review` runs the pipeline end-to-end without any prompts.

**Fully agentic default.** Without `--review`, all three phases accept the first LLM output
directly and proceed immediately. Invalid weights are silently re-queued for an LLM retry
until all nodes have valid weights. State is still checkpointed after each phase.

**`human_review` parameter.** The boolean is threaded through `run_base_phase`,
`run_expansion_phase`, `run_weight_phase`, and `_resolve_invalid_weights` via a single
`human_review` keyword argument — no new modules, just `if human_review:` guards around
the existing review and correction calls.

### v4 — Interactive weight correction loop

**Invalid weight detection.** After the weight LLM responds, `find_invalid_weights` scans
every node for invalid values (negative, non-numeric, boolean, or missing from the response).
Previously these silently fell back to 1 with no indication to the user.

**Per-node interactive correction.** For each invalid weight you are prompted at the terminal:
press **Enter** to queue the node for an LLM retry, or type an integer to set the weight
directly. The correction loop collects all your choices before making any LLM call.

**Targeted LLM retry.** Nodes queued for regeneration are sent back to the model in a single
targeted call that names only those node IDs (the full rubric is still sent for context).
The retry cycle repeats until all weights are valid, then the normal draft-review prompt appears.

**Feedback forwarded on weight rerun.** Typing feedback at the weight review prompt now
correctly forwards that text to the LLM on the next attempt (previously the feedback was
captured but silently dropped).

### v3 — Per-subtree review, typed feedback, and cost reporting

**Per-subtree expansion review.** Phase 2 now fully expands an entire top-level node's
subtree (all depths, BFS) before pausing for review. Previously the tool reviewed after each
individual node expansion — for a paper with 8 top-level sections each going 3 levels deep,
this means 8 review prompts instead of ~24.

**Typed feedback on rerun.** Instead of a silent rerun, any text you type at a review prompt
is forwarded to the LLM as extra context when the pass reruns. This replaces the old `rerun`
keyword.

**Cost reporting.** A new `pb_cost.py` module adds `CostTracker`, which is threaded through
every LLM call in the pipeline. At the end of each run, a formatted report is printed showing
input, output, cache write, and cache read token counts per model along with the estimated
total cost. Pricing constants are embedded in `pb_cost.py` and reflect the 1-hour extended
cache TTL rates.

### v2 — Prompt caching, model tiering, and MinerU integration

The pipeline was refactored from a LangChain-backed single-model flow into a cost-optimized
multi-model pipeline built on the raw Anthropic SDK:

**LangChain removed.** `ChatAnthropic` replaced with `anthropic.Anthropic` directly.
`build_client()` attaches the extended-cache-ttl beta header at construction time.

**Extended prompt caching.** System preamble and PDF are each marked
`cache_control: {type: ephemeral}`. With a 1-hour TTL, the cache outlasts human-review
pauses between phases, eliminating repeated full-context charges.

**Model tiering.** `claude-opus-4-8` is used only for the base pass where deep paper
comprehension is needed. All expansion calls and the weight pass use `claude-sonnet-4-6`
(~5× cheaper per token), which is adequate for mechanical child generation and weight
assignment.

**MinerU integration.** Two new modules handle structured paper content:
- `pb_input.py` — discovers the PDF and MinerU output folder from the input directory and
  loads `content_list.json`.
- `pb_mineru.py` — converts MinerU blocks to LLM-readable text and slices the content list
  to the relevant section for each expansion call.

**CLI updated.** The positional `pdf_path` argument is replaced by named `--input` and
`--output` flags. Output files are written to the `--output` directory rather than alongside
the input PDF.

These changes together reduce the per-paper cost from ~$9 to ~$4.

---

## Setup

### 1. Python environment

Requires **Python 3.11**. Install via pyenv if needed:

```bash
pyenv install 3.11.12
```

Create and activate a virtual environment inside the repo:

```bash
python -m venv .venv --python ~/.pyenv/versions/3.11.12/bin/python3.11
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. API key

Create a `.env` file in the repo root:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

`rubric_gen.py` loads `.env` automatically via `python-dotenv`. Alternatively:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. MinerU

Run [MinerU](https://github.com/opendatalab/MinerU) on your PDF to produce a
`content_list.json`, then place it under `data/input/<paper>/`:

```
data/input/my-paper/
  paper.pdf
  mineru_out/
    content_list.json
```

---

## Usage

Generate a rubric (agentic, no prompts):

```bash
python rubric_gen.py --input data/input/my-paper --output data/output/my-paper
```

Generate a rubric with human review after each pass:

```bash
python rubric_gen.py --input data/input/my-paper --output data/output/my-paper --review
```

In `--review` mode the tool pauses after each pass for you to edit `rubric_draft.json`.
Press **Enter** to approve (with or without edits), or type feedback and press **Enter** to
discard the output and re-run that pass with your feedback forwarded to the model.

At the end of the run, a cost report is printed showing token counts and estimated USD cost
broken down by model.

### Resuming an interrupted run

Every approved pass is checkpointed atomically to `rubric_state.json`. To pick up where
you left off:

```bash
python rubric_gen.py --input data/input/my-paper --output data/output/my-paper --resume
```

Without `--resume`, an existing `rubric_state.json` is overwritten and the run starts fresh.

### Output files

| File | Description |
|---|---|
| `rubric_draft.json` | Editable draft for the current pass. |
| `rubric_state.json` | Resumable checkpoint (rubric + expansion queue + hints + guardrail errors). |
| `rubric_final.json` | Final validated rubric (written when all phases complete). |
| `errors.txt` | Guardrail violations, one per line. Only written if at least one occurred; today the only violation is the model trying to expand a node past `MAX_EXPANSION_DEPTH` (7), logged as `"<node-id>: Model attempted to expand past the maximum depth of 7 nodes."`. |

If `rubric_final.json` already exists and `--resume` is passed, the tool reports it and
exits — delete it or omit `--resume` to start over.

---

## Configuration

| Setting | How to set | Default |
|---|---|---|
| Few-shot example rubric | `FEW_SHOT_RUBRIC_PATH` env var | `examples/example_rubric.json` |

---

## Running the tests

The test suite runs entirely offline (Anthropic calls are faked):

```bash
.venv/bin/pytest tests/
```

All tests pass without an API key or network access.

---

## Attribution

The `TaskNode` schema in `task_node.py` is adapted from OpenAI's
[frontier-evals / PaperBench](https://github.com/openai/preparedness) project
(`project/paperbench/paperbench/rubric/tasks.py`), reproduced here so this repository stays
self-contained.
