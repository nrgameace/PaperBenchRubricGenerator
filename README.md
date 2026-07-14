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
   or a leaf. It also emits a `section_map` — approximate page span, table count, and figure
   count per top-level child — consumed later by the weight pass's embedding-based rescale;
   this rides the same cached base-pass call, no extra PDF read.

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

   **Max branch node cap.** Each top-level branch is capped at `MAX_BRANCH_NODES` (40) total
   nodes, independent of the depth guardrail — a model stuck re-expanding many near-identical
   sub-items could otherwise stay within the depth limit while still generating an unbounded
   number of nodes. The cap is checked before every expansion call; once a branch reaches 40
   nodes, every node still queued for expansion in that branch is force-flattened into a leaf
   (fallback `task_category` of `Code Development`) without spending an LLM call on it, and the
   branch id is recorded so the split-check pass (below) prioritizes it for duplicate review.

   **Enumeration-triggered recursion.** Before a candidate leaf is finalized, its requirements
   text is scanned for enumeration signals — comma-separated lists after "including"/"such
   as"/a colon, `et al.` citations, or `Table`/`Figure` references. If it names three or more
   distinct sub-items (e.g. "All 10 environments are configured and runnable: env_a, env_b,
   ..."), the node is forced back into an expandable/pending state instead of being accepted
   as one dense leaf, and the next expansion call is explicitly told to produce at least that
   many children — one per named item. This is a cheap regex pass (`pb_enumeration.py`), no
   extra LLM call, and applies to both the base pass and every expansion pass since both share
   `normalize_child`.

3. **Split-check pass** — `claude-sonnet-4-6`, one call per top-level branch, runs after
   expansion is fully complete and before weighting. `pb_enumeration.py`'s regex heuristic
   only catches *syntactic* bundling (comma lists, `et al.`, Table/Figure refs). This pass
   catches *semantic* bundling with no consistent trigger phrase — e.g. "small, medium, and
   large model variants" — but only in the two categories where over-bundling concentrates:
   leaves with `finegrained_task_category == "Evaluation, Metrics & Benchmarking"` or
   `task_category == "Result Analysis"` — except for a branch that triggered the max branch
   node cap above, where every leaf in the branch is a candidate regardless of category, since
   that's the branch most likely to contain runaway duplication. For each top-level branch, the
   candidate leaves in scope (if any — the LLM call is skipped entirely for branches with none)
   are sent along with the branch's paper section text; the model returns two judgments:
   replacement children for any leaf that bundles 2+ independently verifiable claims
   (`"splits"`), and the id of an existing leaf that a candidate restates in different words
   (`"duplicates"`) — the duplicate is removed rather than split. If removing a leaf empties
   its parent's children, the parent is force-flattened into a leaf too, since an internal node
   with no children is invalid. Every applied split or duplicate removal is logged to
   `errors.txt`, and if a replacement child still trips the enumeration threshold that's logged
   too, as a model-under-split warning. Disable with `--no-split-check`.

4. **Weight pass** — one LLM sub-phase (`claude-sonnet-4-6`) followed by a deterministic,
   embedding-based rescale (no LLM call):
   - **Local passes** — one LLM call per top-level branch. Each call is focused on a single
     subtree so the model can reason about relative importance within that branch without
     distraction from unrelated sections. The full rubric is still sent for context.
   - **Deterministic embedding-based rescale** (`pb_embeddings.rescale_global_weights`) —
     replaces what used to be a second global LLM calibration call. That call tended to weight
     branches by leaf count rather than substance: on one production run, a branch about a minor
     scaling experiment (many leaves) ended up with nearly double the total leaf weight of the
     branch covering the paper's headline result (few leaves). The rescale instead: (1) embeds
     every leaf's requirements text in one batched OpenAI `text-embedding-3-small` call; (2)
     clusters each branch's leaves by cosine similarity (threshold 0.87) to count *distinct
     claims* rather than raw leaf count — this "branch mass" is what stops an over-decomposed
     branch from inflating its own importance; (3) derives each branch's target weight share
     from the base pass's `section_map` (page/table/figure span), falling back to a uniform
     share for any branch with a missing or malformed entry; (4) rescales every leaf's weight
     by `(target * total_mass) / branch_mass`, rounded and floored at 1. It also clusters all
     leaves again ignoring branch boundaries (stricter threshold 0.92) to flag likely
     cross-branch duplicates in `flagged_duplicates.json` for manual review — never
     auto-deleted. Requires an `OPENAI_API_KEY` (separate from `ANTHROPIC_API_KEY`).
   - The root node is always set to weight 1 by the orchestrator (no LLM call).
   - Invalid weights (negative, non-numeric, boolean, or missing) are corrected after both the
     local pass and the rescale: in `--review` mode you are prompted per-node to enter a weight
     manually or queue it for an LLM retry; in agentic mode all invalid nodes are automatically
     re-queued.
   - **Max retry guardrail.** Weight correction is capped at `MAX_WEIGHT_RESOLUTION_RETRIES`
     (5) retry cycles, in both `--review` and agentic mode. If nodes are still invalid after 5
     retries, the run raises `MaxRetriesExceeded`, which `main()` turns into a clean
     `SystemExit` (no stack trace) telling you to re-run with `--resume` — it never spins
     forever burning tokens on a model that keeps returning bad weights.
   - Human review happens **once**, after the local pass, the rescale, and all invalid-weight
     correction are complete. Since the rescale is deterministic, typed review feedback no
     longer steers it (there's no LLM call left to steer) — a rerun re-runs the local branch
     passes and the rescale, and direct hand-edits to `rubric_draft.json` remain the way to
     override individual weights.

### Prompt caching

Two blocks are cached with a 1-hour TTL (`cache_control: {type: ephemeral, ttl: "1h"}`,
no beta header required):

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
    rubric_state.json     ← resumable checkpoint (rubric + expansion queue + hints + errors + capped_branches + section_map)
    rubric_draft.json     ← editable draft for the current pass
    rubric_final.json     ← final validated rubric (written when all phases complete)
    errors.txt              ← guardrail violations, if any occurred (see Output files)
    flagged_duplicates.json ← cross-branch duplicate leaves flagged for manual review, if any (see Output files)
examples/
  example_rubric.json     ← few-shot format/depth exemplar (different paper)
tests/
```

### Module responsibilities

| File | Responsibility |
|---|---|
| `rubric_gen.py` | Entry point; CLI parsing (`--review`, `--resume`, `--no-split-check`); orchestrates all 4 phases; `human_review` flag threaded through phase functions; `_expand_subtree` enforces the `MAX_BRANCH_NODES` cap before every expansion call and threads a `capped_branches` set the same way it threads `errors`; `run_split_check_phase` runs the split-check pass per top-level branch before weighting, processing `"duplicates"` (via `apply_dedup`) before `"splits"` so a leaf flagged as both is removed rather than split, and passing `include_all_leaves=True` for branches in `capped_branches`; `run_weight_phase` runs the local branch LLM passes then `pb_embeddings.rescale_global_weights` for the deterministic cross-branch rescale; `_resolve_invalid_weights` correction loop, capped at `MAX_WEIGHT_RESOLUTION_RETRIES` (5) and raising `MaxRetriesExceeded` past that; `write_error_log` writes `errors.txt`; `write_flagged_duplicates` writes `flagged_duplicates.json`; prints cost report |
| `pb_passes.py` | Anthropic SDK calls; prompt construction; JSON parsing; node normalization; tree mutation (`apply_base`, `apply_expansion`, `apply_split`, `apply_dedup`, `apply_weights`); `apply_base` also extracts the base pass's `section_map` as its 4th return value (defaults to `{}` if absent); `MAX_EXPANSION_DEPTH` (7) guardrail enforced in `apply_expansion`; `MAX_BRANCH_NODES` (40) guardrail via `branch_size`/`force_branch_cap_leaves`, checked by `rubric_gen._expand_subtree` before every expansion call; enumeration-triggered recursion override in `normalize_child` and prompt injection in `run_expansion_llm` (via `pb_enumeration`); `run_split_check_llm` — the semantic bundling/duplicate check scoped to `Evaluation, Metrics & Benchmarking` / `Result Analysis` leaves (or every leaf in the branch via `include_all_leaves=True`), one call per branch returning both `"splits"` and `"duplicates"`, skipped entirely when a branch has no candidate leaves; `apply_dedup` removes a duplicate leaf (via `pb_schema.find_parent`) and force-flattens its parent if that empties the parent's children; weight validation (`find_invalid_weights`); `run_weight_llm_branch` for per-branch local passes; `run_weight_llm` for targeted invalid-weight retries; `invoke_llm` raises `RuntimeError` if the model response is truncated (`stop_reason == "max_tokens"`) instead of failing downstream as an opaque JSON parse error, and calls the Anthropic SDK's streaming API (`client.messages.stream(...)` / `get_final_message()`) rather than the blocking `create(...)`, since large `max_tokens` values can otherwise be rejected outright by the SDK; `run_split_check_llm` scales its `max_tokens` (8000 up to a 32000 cap) with candidate leaf count since `include_all_leaves` branches can need much longer responses; `parse_json_response` raises `ValueError` with a preview of the raw model text (up to 2000 chars) when a response isn't valid JSON, instead of a bare `JSONDecodeError` with no diagnostic context; `_extract_json_span` scans every `{`/`[` and keeps the longest balanced JSON decode rather than naively spanning first-open-to-last-close, so stray bracket characters in a model's reasoning prose (math ranges, citations) ahead of its real JSON answer can't be mistaken for the payload |
| `pb_embeddings.py` | Deterministic embedding-based replacement for the old global LLM weight-calibration pass. `build_embedding_client`/`embed_texts` wrap the OpenAI embeddings API (`text-embedding-3-small`, one batched call); `extract_leaves` tags every leaf with its top-level branch id; `cosine_similarity`/`cluster_by_threshold` are a hand-rolled greedy single-link clusterer (no numpy/scipy/sklearn); `branch_mass`/`compute_all_branch_masses` count distinct-claim clusters per branch, not raw leaf count; `derive_target_proportions` converts `section_map` into per-branch target weight shares, falling back to a uniform share per branch on a missing/malformed entry; `rescale_branch_weights` applies the per-branch factor, floored at weight 1; `cluster_cross_branch_duplicates`/`build_duplicate_report`/`write_flagged_duplicates` flag (never auto-delete) likely duplicate leaves across branches; `rescale_global_weights` is the top-level orchestrator |
| `pb_cost.py` | `CostTracker` — accumulates token usage (input, output, cache write, cache read) per model; computes and prints a formatted cost report |
| `pb_input.py` | Discovers PDF and MinerU folder from input dir; loads `content_list.json` |
| `pb_enumeration.py` | `count_enumerated_items` and `build_enumeration_hint` — pure regex heuristics (no LLM/network dependency) detecting enumerated sub-items in requirements text; `MIN_ENUMERATED_ITEMS_TO_SPLIT` (3) constant |
| `pb_mineru.py` | Converts MinerU blocks to LLM-readable text (`blocks_to_text`); slices content to a section by heading fuzzy-match (`slice_section`) |
| `pb_schema.py` | Rubric dict traversal and validation (`validate_partial`, `validate_final`, `find_node`, `find_parent`, `all_ids`, `node_depth`, `iter_nodes`) |
| `pb_state.py` | State persistence; phase constants (`PHASE_BASE → EXPANSION → WEIGHT → DONE`); atomic write via temp-file rename; state includes an `errors` list of guardrail violations, a `capped_branches` list of branch ids that hit `MAX_BRANCH_NODES`, and a `section_map` dict from the base pass |
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

### v13 — Deterministic embedding-based weight rescale (replaces global LLM calibration)

**`pb_embeddings.py` (new module).** The weight pass's global LLM calibration call
(`run_weight_llm_global`) is removed and replaced with a deterministic, embedding-based rescale
(`rescale_global_weights`). The global LLM pass tended to weight branches by leaf count rather
than substance — on a production run, a branch about a minor scaling experiment (many leaves)
ended up with nearly double the total leaf weight of the branch covering the paper's headline
result (few leaves), because the LLM effectively counted leaves instead of judging how much of
the paper's actual content or how many distinct claims each branch covered.

The replacement: one batched OpenAI `text-embedding-3-small` call embeds every leaf's
requirements text (the only new API cost — no further LLM judgment calls). Each branch's
leaves are clustered by cosine similarity (hand-rolled greedy single-link, threshold 0.87, no
numpy/scipy/sklearn) to compute a "branch mass" — its distinct-claim cluster count, not its raw
leaf count — which is what stops an over-decomposed branch from inflating its own importance.
Each branch's target weight share comes from a new `section_map` the base pass now also emits
(page span, table count, figure count per top-level child, riding the same cached PDF read — no
extra PDF load), falling back to a uniform share for any branch with a missing or malformed
entry rather than failing the whole rescale. Every leaf's weight is then rescaled by
`(target * total_mass) / branch_mass`, rounded and floored at 1 — pure arithmetic, deterministic
and reproducible run to run. The same embeddings are reused to cluster all leaves again ignoring
branch boundaries (stricter threshold 0.92), flagging any cluster spanning more than one branch
in `flagged_duplicates.json` for manual review (never auto-deleted) — this catches the same
underlying claim independently restated in different branches with no shared trigger phrase, the
cross-branch equivalent of the split-check pass's within-branch duplicate detection.

Since the rescale is now deterministic, human review of the weight phase still gates the final
result (approve or `RerunPass`), but typed feedback text is no longer forwarded to it — there is
no LLM call left to steer. A `RerunPass` re-runs the local per-branch LLM passes and the
deterministic rescale; direct hand-edits to `rubric_draft.json` remain the way to override
individual weights. Requires a new `OPENAI_API_KEY` (separate from `ANTHROPIC_API_KEY`) and adds
`openai` to `requirements.txt`.

### v12 — Per-branch node cap and duplicate-leaf detection

**`MAX_BRANCH_NODES` (40) guardrail (`pb_passes.branch_size` / `force_branch_cap_leaves`,
wired into `rubric_gen._expand_subtree`).** The depth guardrail bounds how *deep* a branch can
get but not how *wide* — a model stuck re-expanding many near-identical sub-items can stay
within `MAX_EXPANSION_DEPTH` while still generating an unbounded number of nodes. Since
`_expand_subtree` only ever processes one top-level branch per call (its `node_id` argument
*is* the branch id — sub-node pending ids never touch `state["queue"]`), the cap is checked at
the top of the BFS loop before every expansion call: once `branch_size` reaches 40, every node
still queued for that branch is force-flattened into a leaf in one call
(`force_branch_cap_leaves`) rather than spending further LLM calls, and the branch id is
recorded into a new `capped_branches` set — threaded through `run_expansion_phase` and
persisted in `state`/`rubric_state.json` the same way `errors` already is (discarded on a
rejected `RerunPass` candidate, written back only on approval).

**Duplicate-leaf detection, folded into the split-check pass.** The split-check LLM call now
asks for a second judgment alongside `"splits"`: `"duplicates"`, mapping a leaf id to the id of
another leaf in the branch it restates in different words (same underlying claim, no shared
trigger phrase — the kind of duplication a regex heuristic can't catch). `run_split_check_phase`
processes `"duplicates"` before `"splits"` so a leaf flagged as both ends up removed rather than
split, with existence/self-reference guards so a malformed model response can't crash the run.
The new `apply_dedup` (using a new `pb_schema.find_parent` helper) removes the leaf from its
parent's `sub_tasks`, force-flattening the parent into a leaf too if that empties its children
(an internal node can't have zero children). Branches that triggered the node cap above have
every leaf included as a duplicate/split candidate regardless of category — they're the ones
most likely to contain runaway duplication — via a new `include_all_leaves` flag on
`_split_check_candidates`/`run_split_check_llm`.

### v11 — Targeted split-check pass for semantic bundling

**`run_split_check_llm` / `apply_split` (`pb_passes.py`).** `pb_enumeration.py`'s regex
heuristic only catches *syntactic* bundling (comma lists, `et al.`, Table/Figure refs). This
adds a second, semantic-tier check for bundling with no consistent trigger phrase — e.g.
"small, medium, and large model variants" — scoped to leaves where
`finegrained_task_category == "Evaluation, Metrics & Benchmarking"` or `task_category ==
"Result Analysis"`, since that's where over-bundling concentrates. One Sonnet call per
top-level branch collects the branch's matching leaves (skipping the call entirely if none
match) and asks the model which leaves bundle 2+ independently verifiable claims and what to
split them into. `apply_split` reuses `normalize_child` for each replacement child, so the
existing enumeration sanity check doubles as an under-split detector: a replacement child
that still trips `MIN_ENUMERATED_ITEMS_TO_SPLIT` is forced into a valid leaf (the same
depth-guardrail fallback category) and logged as a warning rather than silently accepted.

**`run_split_check_phase` (`rubric_gen.py`).** Runs after expansion is fully complete and
before the weight pass, iterating top-level branches and logging every applied split (leaf
id, branch, child count) to the same `state["errors"]` list `errors.txt` is written from —
this doubles as the audit trail for what the pass changed. Idempotent across `--resume`: a
split leaf's category fields are cleared, so it no longer matches the filter on a later run,
with no new `pb_state.py` phase constant needed.

**`--no-split-check` flag.** Opt-out flag (pass runs by default), mirroring the `--review`
opt-in pattern, for disabling the pass during quick iteration.

### v10 — Live per-call cost indicator

**`invoke_llm` prints running cost.** Immediately after each Anthropic API call is recorded
into `CostTracker`, `invoke_llm` (the single choke point every `run_*_llm` function funnels
through) prints `Current usage: $X.XXXX` — the cumulative cost for the session so far, via
the existing `CostTracker.total_cost()`. This covers every phase and both agentic and
`--review` modes with one change, since all model calls already route through `invoke_llm`.
No print happens when no tracker is passed. `--resume` already starts `CostTracker` fresh
(it isn't persisted to `rubric_state.json`), so the live counter and final report both
reflect only the current session, restarting from `$0.0000` on resume.

### v9 — Enumeration-triggered recursion guardrail

**`pb_enumeration.py`.** A new pure module detects when a node's requirements text names
multiple distinct sub-items — comma-separated lists after "including"/"such as"/a colon,
`et al.` citations, or `Table`/`Figure` references — via `count_enumerated_items`. A count
`>= MIN_ENUMERATED_ITEMS_TO_SPLIT` (3) signals the text should split into one child per item
rather than remain a single dense node.

**Leaf override in `normalize_child`.** A node the model marked non-expandable is now forced
back into a pending/expandable state when its own requirements enumerate enough items, with a
synthesized hint (`build_enumeration_hint`) telling the next expansion call how many children
to produce. Since `normalize_child` is the single chokepoint shared by both `apply_base`
(Phase 1) and `apply_expansion` (Phase 2), this applies uniformly across both without any
changes to either function's body — a node forced into pending state composes with the
existing `MAX_EXPANSION_DEPTH` guardrail exactly like any other expandable node.

**Prompt injection in `run_expansion_llm`.** When the node being expanded itself enumerates
multiple items, an "ENUMERATION GUARDRAIL" instruction is appended to the Sonnet prompt asking
for at least that many children, one per named item.

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

**Extended prompt caching.** System preamble and PDF are each marked
`cache_control: {type: ephemeral, ttl: "1h"}`. With a 1-hour TTL, the cache outlasts human-review
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

### 2. API keys

Create a `.env` file in the repo root. Two keys are required: `ANTHROPIC_API_KEY` for the
base/expansion/split-check/weight LLM passes, and `OPENAI_API_KEY` for the weight pass's
embedding-based rescale.

```bash
cat >> .env <<'EOF'
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
EOF
```

`rubric_gen.py` loads `.env` automatically via `python-dotenv`. Alternatively:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
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

Add `--no-split-check` to skip the split-check pass (runs by default) that checks
`Evaluation, Metrics & Benchmarking` / `Result Analysis` leaves for semantic bundling before
weighting — useful for faster iteration when you don't need that extra pass.

After every single LLM call (in both agentic and `--review` mode), a running total is
printed to the screen: `Current usage: $0.0421`. This reflects only the current session —
a `--resume` run starts the counter back at `$0.0000`, even though the underlying rubric
state is restored. At the end of the run, a full cost report is printed showing token
counts and estimated USD cost broken down by model.

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
| `rubric_state.json` | Resumable checkpoint (rubric + expansion queue + hints + guardrail errors + capped branch ids + base-pass `section_map`). |
| `rubric_final.json` | Final validated rubric (written when all phases complete). |
| `errors.txt` | Guardrail violations and split-check activity, one per line. Only written if at least one occurred. Includes: the model trying to expand a node past `MAX_EXPANSION_DEPTH` (7), logged as `"<node-id>: Model attempted to expand past the maximum depth of 7 nodes."`; a branch hitting `MAX_BRANCH_NODES` (40), logged per forced node as `"<node-id>: Branch '<branch-id>' hit the 40-node cap; forced to leaf."`; every split applied by the split-check pass, logged as `"<leaf-id>: split-check split into N children (branch '<branch-id>')."`; every duplicate leaf removed by the split-check pass, logged as `"<leaf-id>: split-check removed as duplicate of '<other-leaf-id>'."` (plus a second line if removing it force-flattened its now-childless parent); and any split-check child that still trips the enumeration threshold, logged as a model-under-split warning. |
| `flagged_duplicates.json` | Leaves from *different* top-level branches that the weight pass's embedding-based rescale clustered together as likely restating the same claim (cosine similarity ≥ 0.92). Only written if at least one cross-branch cluster was found; never auto-deleted — for manual review. |

If `rubric_final.json` already exists and `--resume` is passed, the tool reports it and
exits — delete it or omit `--resume` to start over.

---

## Configuration

| Setting | How to set | Default |
|---|---|---|
| Few-shot example rubric | `FEW_SHOT_RUBRIC_PATH` env var | `examples/example_rubric.json` |
| Embedding model | `pb_embeddings.EMBEDDING_MODEL` | `text-embedding-3-small` |
| Within-branch cluster threshold | `pb_embeddings.BRANCH_CLUSTER_THRESHOLD` | `0.87` |
| Cross-branch duplicate threshold | `pb_embeddings.DUPLICATE_CLUSTER_THRESHOLD` | `0.92` |

---

## Running the tests

The test suite runs entirely offline (Anthropic and OpenAI calls are faked):

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
