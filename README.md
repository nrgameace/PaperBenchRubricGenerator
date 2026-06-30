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

3. **Weight pass** — `claude-sonnet-4-6` receives the full MinerU text and the completed
   tree and assigns an integer weight to every node reflecting its importance to the paper's
   core contributions. After the LLM responds, any invalid weights (negative, non-numeric,
   boolean, or missing) trigger an interactive correction loop — you are prompted per-node
   to either enter a weight manually or queue the node for an LLM retry. The retry cycle
   repeats until all weights are valid, then the normal review prompt appears.

### Prompt caching

Two blocks are cached with a 1-hour TTL via the Anthropic extended-cache-ttl beta
(`prompt-caching-2024-07-31,extended-cache-ttl-2025-02-19`):

- **System preamble** — grounding rules and the few-shot format exemplar.
- **PDF document block** — base64-encoded PDF, sent only in the base pass.

The cache survives the human-review window between phases, so you don't pay to re-send the
PDF or system prompt on expansion and weight calls.

### Human-in-the-loop review

After each pass the candidate rubric is written to `rubric_draft.json`. The tool pauses and
prompts you to edit that file as needed:

- **Press Enter** — your edits are validated against the PaperBench `TaskNode` schema. If
  valid, the draft is committed and the next pass begins.
- **Type any text + Enter** — your text is forwarded to the LLM as feedback and that pass
  reruns with the extra context. State is NOT checkpointed until you press Enter with no text.

State is saved atomically to `rubric_state.json` after every approved pass.

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
    rubric_state.json     ← resumable checkpoint (rubric + expansion queue + hints)
    rubric_draft.json     ← editable draft for the current pass
    rubric_final.json     ← final validated rubric (written when all phases complete)
examples/
  example_rubric.json     ← few-shot format/depth exemplar (different paper)
tests/
```

### Module responsibilities

| File | Responsibility |
|---|---|
| `rubric_gen.py` | Entry point; CLI parsing; orchestrates all 3 phases; human review loop; prints cost report |
| `pb_passes.py` | Anthropic SDK calls; prompt construction; JSON parsing; node normalization; tree mutation (`apply_base`, `apply_expansion`, `apply_weights`); weight validation (`find_invalid_weights`); optional `feedback`, `node_ids`, and `tracker` params on weight LLM calls |
| `pb_cost.py` | `CostTracker` — accumulates token usage (input, output, cache write, cache read) per model; computes and prints a formatted cost report |
| `pb_input.py` | Discovers PDF and MinerU folder from input dir; loads `content_list.json` |
| `pb_mineru.py` | Converts MinerU blocks to LLM-readable text (`blocks_to_text`); slices content to a section by heading fuzzy-match (`slice_section`) |
| `pb_schema.py` | Rubric dict traversal and validation (`validate_partial`, `validate_final`, `find_node`, `all_ids`) |
| `pb_state.py` | State persistence; phase constants (`PHASE_BASE → EXPANSION → WEIGHT → DONE`); atomic write via temp-file rename |
| `pb_review.py` | Blocks on `input()` for human review; raises `RerunPass(feedback)` when user types non-empty text; `collect_weight_corrections` handles interactive per-node weight correction |
| `task_node.py` | Frozen `TaskNode` dataclass (adapted from OpenAI's frontier-evals); leaf/internal validation in `__post_init__` |

### TaskNode rules

- **Leaf nodes**: `task_category` must be one of `Code Development`, `Code Execution`,
  `Result Analysis`, `Paper Analysis`; `sub_tasks` must be empty.
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

Generate a rubric:

```bash
python rubric_gen.py --input data/input/my-paper --output data/output/my-paper
```

The tool will pause after each pass for you to review and edit `rubric_draft.json` in the
output directory. Save your edits, return to the terminal, and press **Enter** to continue.
Type any feedback text and press **Enter** to discard the output and re-run that pass with
your feedback forwarded to the model.

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
| `rubric_state.json` | Resumable checkpoint (rubric + expansion queue + hints). |
| `rubric_final.json` | Final validated rubric (written when all phases complete). |

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
