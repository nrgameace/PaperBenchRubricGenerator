"""Tests for CostTracker token accumulation and cost calculation."""

from types import SimpleNamespace

import pytest

from pb_cost import PRICING, CostTracker
from pb_embeddings import EMBEDDING_MODEL
from rubric_gen import JUDGE_MODEL, OPUS, SONNET


def _usage(input=0, output=0, cache_write=0, cache_read=0):
    return SimpleNamespace(
        input_tokens=input,
        output_tokens=output,
        cache_creation_input_tokens=cache_write,
        cache_read_input_tokens=cache_read,
    )


def test_record_accumulates_across_calls():
    tracker = CostTracker()
    tracker.record("claude-opus-4-8", _usage(input=100, output=50))
    tracker.record("claude-opus-4-8", _usage(input=200, output=100))
    totals = tracker.totals_for("claude-opus-4-8")
    assert totals["input"] == 300
    assert totals["output"] == 150


def test_record_tracks_cache_tokens():
    tracker = CostTracker()
    tracker.record("claude-sonnet-5", _usage(cache_write=500, cache_read=1000))
    totals = tracker.totals_for("claude-sonnet-5")
    assert totals["cache_write"] == 500
    assert totals["cache_read"] == 1000


def test_record_multiple_models():
    tracker = CostTracker()
    tracker.record("claude-opus-4-8", _usage(input=100))
    tracker.record("claude-sonnet-5", _usage(input=200))
    assert tracker.totals_for("claude-opus-4-8")["input"] == 100
    assert tracker.totals_for("claude-sonnet-5")["input"] == 200


def test_total_cost_calculation():
    tracker = CostTracker()
    tracker.record("claude-opus-4-8", _usage(input=1_000_000))
    assert abs(tracker.total_cost() - 5.00) < 0.001


def test_total_cost_all_token_types():
    tracker = CostTracker()
    tracker.record("claude-opus-4-8", _usage(
        input=1_000_000, output=1_000_000,
        cache_write=1_000_000, cache_read=1_000_000,
    ))
    assert abs(tracker.total_cost() - 40.50) < 0.01


def test_total_cost_zero_when_empty():
    assert CostTracker().total_cost() == 0.0


def test_print_report_outputs_header_and_total(capsys):
    tracker = CostTracker()
    tracker.record("claude-opus-4-8", _usage(input=1000, output=500))
    tracker.print_report()
    out = capsys.readouterr().out
    assert "TOKEN USAGE" in out
    assert "$" in out


def test_print_report_unknown_model_does_not_crash():
    tracker = CostTracker()
    tracker.record("claude-unknown-model", _usage(input=100))
    tracker.print_report()


def test_pricing_covers_models_actually_used_by_the_pipeline():
    for model in (OPUS, SONNET):
        assert model in PRICING, f"{model} has no pricing entry; cost report will be silently wrong"
        rates = PRICING[model]
        assert all(rates[t] > 0 for t in ("input", "output", "cache_write", "cache_read"))


def test_pricing_covers_judge_model():
    assert JUDGE_MODEL in PRICING, f"{JUDGE_MODEL} has no pricing entry; cost report will be silently wrong"
    rates = PRICING[JUDGE_MODEL]
    assert rates["input"] > 0
    assert rates["output"] > 0


def test_total_cost_prices_judge_model():
    tracker = CostTracker()
    tracker.record(JUDGE_MODEL, _usage(input=1_000_000, output=1_000_000))
    expected = PRICING[JUDGE_MODEL]["input"] * 1_000_000 + PRICING[JUDGE_MODEL]["output"] * 1_000_000
    assert abs(tracker.total_cost() - expected) < 0.001


def test_pricing_covers_embedding_model():
    assert EMBEDDING_MODEL in PRICING, f"{EMBEDDING_MODEL} has no pricing entry; cost report will be silently wrong"
    rates = PRICING[EMBEDDING_MODEL]
    assert rates["input"] > 0
    assert rates["output"] == 0
    assert rates["cache_write"] == 0
    assert rates["cache_read"] == 0


def test_total_cost_prices_embedding_model():
    tracker = CostTracker()
    tracker.record(EMBEDDING_MODEL, _usage(input=1_000_000))
    expected = PRICING[EMBEDDING_MODEL]["input"] * 1_000_000
    assert abs(tracker.total_cost() - expected) < 0.001


def test_print_report_prints_provider_subtotals_for_anthropic_and_openai(capsys):
    tracker = CostTracker()
    tracker.record(SONNET, _usage(input=1000, output=500))
    tracker.record(JUDGE_MODEL, _usage(input=1000, output=500))
    tracker.print_report()
    out = capsys.readouterr().out
    assert "Anthropic subtotal" in out
    assert "OpenAI subtotal" in out
