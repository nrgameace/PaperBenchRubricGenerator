"""Tests for the enumeration-triggered recursion heuristic (no network)."""

import pb_enumeration

_ENV_SETUP_2_TEXT = (
    "All 10 environments are configured and runnable: ant_big_maze, ant_hardest_maze, "
    "arm_binpick_hard, arm_push_easy, arm_push_hard, humanoid, humanoid_big_maze, "
    "humanoid_u_maze, ant_u4_maze, ant_u5_maze."
)


def test_count_enumerated_items_empty_text_returns_zero():
    assert pb_enumeration.count_enumerated_items("") == 0
    assert pb_enumeration.count_enumerated_items(None) == 0


def test_count_enumerated_items_single_item_below_threshold():
    assert pb_enumeration.count_enumerated_items("The model uses dropout of 0.1.") == 0


def test_count_enumerated_items_colon_triggered_comma_list_matches_confirmed_example():
    count = pb_enumeration.count_enumerated_items(_ENV_SETUP_2_TEXT)
    assert count == 10
    assert count >= pb_enumeration.MIN_ENUMERATED_ITEMS_TO_SPLIT


def test_count_enumerated_items_such_as_trigger():
    text = "Baselines include such as ResNet, VGG, and DenseNet."
    assert pb_enumeration.count_enumerated_items(text) == 3


def test_count_enumerated_items_including_trigger_with_trailing_and():
    text = "The ablation covers three settings, including A, B, and C."
    assert pb_enumeration.count_enumerated_items(text) == 3


def test_count_enumerated_items_et_al_citations():
    text = "We compare against Smith et al., Jones et al., and Lee et al."
    assert pb_enumeration.count_enumerated_items(text) == 3


def test_count_enumerated_items_table_figure_refs_deduped():
    text = "See Table 1 and Table 1 again, and Figure 2."
    assert pb_enumeration.count_enumerated_items(text) == 2


def test_count_enumerated_items_takes_max_not_sum():
    text = "Compare Smith et al. and Jones et al. against baselines: X, Y, Z, W."
    assert pb_enumeration.count_enumerated_items(text) == 4


def test_build_enumeration_hint_falls_back_when_existing_hint_empty():
    hint = pb_enumeration.build_enumeration_hint(None, 5)
    assert hint.startswith("Expand this node into its sub-tasks based on the paper.")
    assert "at least 5 children" in hint


def test_build_enumeration_hint_preserves_existing_hint():
    hint = pb_enumeration.build_enumeration_hint("covers ablations", 4)
    assert "covers ablations" in hint
    assert "at least 4 children" in hint
