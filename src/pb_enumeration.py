"""Cheap regex heuristics for detecting enumeration-triggered recursion signals.

Detects requirements text that names multiple distinct sub-items (baselines, model
variants, table rows, ablation settings) so the expansion pipeline can force such
nodes to split into one child per item instead of collapsing them into a single
dense node.
"""

import re

MIN_ENUMERATED_ITEMS_TO_SPLIT = 3

_LIST_TRIGGER_RE = re.compile(r"(?:including|such as|:)\s*(?P<segment>[^.;]+)", re.IGNORECASE)
_ET_AL_RE = re.compile(r"et al\.", re.IGNORECASE)
_TABLE_FIGURE_RE = re.compile(r"\b(?:Table|Figure)\s+\d+[A-Za-z]?\b")


def _count_comma_list_items(text: str) -> int:
    """Return the length of the longest comma-separated list following a list trigger phrase."""
    best = 0
    for match in _LIST_TRIGGER_RE.finditer(text):
        items = [re.sub(r"^and\s+", "", item.strip(), flags=re.IGNORECASE) for item in match.group("segment").split(",")]
        best = max(best, len([item for item in items if item]))
    return best


def _count_et_al_items(text: str) -> int:
    """Return the number of 'et al.' citations, each indicating one named work."""
    return len(_ET_AL_RE.findall(text))


def _count_table_figure_refs(text: str) -> int:
    """Return the number of distinct Table/Figure references, deduped by matched string."""
    return len(set(_TABLE_FIGURE_RE.findall(text)))


def count_enumerated_items(text: str) -> int:
    """Return the strongest enumeration signal in text: comma list, et al. count, or table/figure refs.

    Takes the max (not sum) across the three signals since they can overlap in the same
    text. A result >= MIN_ENUMERATED_ITEMS_TO_SPLIT means the text names multiple
    distinct sub-items that should become separate children rather than one dense node.
    """
    if not text:
        return 0
    return max(_count_comma_list_items(text), _count_et_al_items(text), _count_table_figure_refs(text))


def build_enumeration_hint(existing_hint: str, enum_count: int) -> str:
    """Build an expansion hint instructing the model to split into >= enum_count children.

    Falls back to the standard generic expansion hint when existing_hint is empty, then
    appends an explicit per-item split instruction referencing the detected count.
    """
    base = (existing_hint or "").strip() or "Expand this node into its sub-tasks based on the paper."
    return f"{base} Split into at least {enum_count} children, one per named item enumerated in the requirements text."
