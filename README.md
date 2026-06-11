# PaperBenchRubricGenerator

Generate a [PaperBench](https://github.com/openai/preparedness)-style grading **rubric**
from a research paper PDF, so you can measure how well a codebase reproduces that paper.

The tool reads a paper, then drives Claude through a sequence of human-reviewed passes to
build a tree of `TaskNode`s — a hierarchy of precise, verifiable requirements that a faithful
reproduction of the paper must satisfy. You review and edit the model's output at every step.

> **Cost:** A full rubric for one paper costs roughly **$9 USD** in Anthropic API usage
> (Claude Opus, large PDF context re-sent on every pass). Budget accordingly before running.

---

## How it works

The rubric is built as a tree and checkpointed after every approved pass, so a run is fully
resumable. There are three phases:

1. **Base node pass** — generates the root node and the top-level areas of work.
2. **Expansion passes** — breadth-first, expands every non-leaf node into its sub-tasks
   (one reviewed pass per node).
3. **Weight pass** — assigns an integer weight to every node, reflecting its importance to
   the paper's core contributions.

Each leaf node is tagged with a category (`Code Development`, `Code Execution`, or
`Result Analysis`) describing what a grader must check.

### Human-in-the-loop review

After each pass the candidate rubric is written to `rubric_draft.json`. The tool pauses and
prompts you to edit that file as needed. When you press **Enter**, your edits are validated
against the PaperBench `TaskNode` schema:

- If the draft is valid, it is committed and the next pass begins.
- If it fails validation, you can fix it again (`e`) or re-run the LLM pass from scratch (`r`).

State is saved to `rubric_state.json` after every approved pass.

---

## Project layout

| File | Purpose |
| --- | --- |
| `rubric_gen.py` | Entry point and pipeline orchestration. |
| `pb_passes.py` | Prompt construction, Claude invocation, JSON parsing, node merging. |
| `pb_review.py` | Human review loop (edit draft → validate → retry). |
| `pb_schema.py` | Rubric traversal and validation helpers. |
| `task_node.py` | Self-contained PaperBench `TaskNode` tree (adapted from OpenAI's frontier-evals). |
| `pb_state.py` | State persistence and phase derivation for resumable runs. |
| `examples/example_rubric.json` | Few-shot example rubric (format/depth reference only). |
| `tests/` | Unit and integration tests (no network required). |

This repository is **self-contained** — it does not depend on an external frontier-evals
checkout. The `TaskNode` class is reproduced locally in `task_node.py` with attribution.

---

## Setup

### 1. Python environment

Requires **Python 3.14**. This project uses a virtual environment named `env` located in the
**parent folder** of this repository.

Activate it (from inside the repo):

```bash
source ../env/bin/activate
```

If you need to create it from scratch instead:

```bash
python3.14 -m venv ../env
source ../env/bin/activate
pip install -r requirements.txt
```

To install/refresh dependencies into the existing environment:

```bash
pip install -r requirements.txt
```

### 2. API key

Copy the example env file and add your Anthropic API key:

```bash
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

`rubric_gen.py` loads `.env` automatically (via `python-dotenv`). Alternatively, export the
key directly:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

Generate a rubric from a paper PDF:

```bash
python rubric_gen.py path/to/paper.pdf
```

The tool will pause after each pass for you to review and edit `rubric_draft.json`. Save your
edits, return to the terminal, and press **Enter** to continue.

### Resuming an interrupted run

Every approved pass is checkpointed to `rubric_state.json`. To pick up where you left off:

```bash
python rubric_gen.py path/to/paper.pdf --resume
```

Without `--resume`, an existing `rubric_state.json` is overwritten and the run starts fresh.

### Output

When all phases complete, the validated rubric is written to **`rubric_final.json`**.

| File | Description |
| --- | --- |
| `rubric_draft.json` | The editable draft for the current pass. |
| `rubric_state.json` | Resumable checkpoint (rubric + expansion queue + hints). |
| `rubric_final.json` | The final, validated rubric. |

If `rubric_final.json` already exists, the tool reports it and exits — delete it (or run
without `--resume`) to start over.

---

## Configuration

- **Model / token budget:** set `MODEL` and `MAX_TOKENS` in `rubric_gen.py`
  (defaults: `claude-opus-4-8`, `8000`).
- **Few-shot example:** the example rubric in `examples/example_rubric.json` is used purely as
  a format and depth reference. Override it by setting `FEW_SHOT_RUBRIC_PATH` to another rubric
  JSON file (see `.env.example`).

---

## Running the tests

The test suite runs entirely offline (Claude calls are faked):

```bash
source ../env/bin/activate
pytest
```

All tests should pass without an API key or network access.

---

## Attribution

The `TaskNode` schema in `task_node.py` is adapted from OpenAI's
[frontier-evals / PaperBench](https://github.com/openai/preparedness) project
(`project/paperbench/paperbench/rubric/tasks.py`), reproduced here so this repository stays
self-contained.
