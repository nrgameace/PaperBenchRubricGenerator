"""Tests for the coverage-judge checkpoint (pb_judge.py)."""

import json
from types import SimpleNamespace

import pytest

import pb_cost
import pb_judge


# ── fake OpenAI chat-completions client ────────────────────────────────────────

class _FakeChatMessage:
    def __init__(self, content):
        self.content = content


class _FakeChatChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = _FakeChatMessage(content)
        self.finish_reason = finish_reason


class _FakeChatUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=5, cached_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = SimpleNamespace(cached_tokens=cached_tokens)


class _FakeChatResponse:
    def __init__(self, content, finish_reason="stop", usage=None):
        self.choices = [_FakeChatChoice(content, finish_reason)]
        self.usage = usage or _FakeChatUsage()


class _FakeChatCompletions:
    def __init__(self, response):
        self.calls = []
        self._response = response

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeChatClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(response))


def _leaf(id_, requirements="did something", task_category="Code Development"):
    return {"id": id_, "requirements": requirements, "weight": 0, "sub_tasks": [],
            "task_category": task_category, "finegrained_task_category": None}


# ── run_coverage_judge ──────────────────────────────────────────────────────────

def test_run_coverage_judge_short_circuits_with_no_leaves():
    client = _FakeChatClient(_FakeChatResponse('{"verdict": "sufficient", "missing": []}'))
    result = pb_judge.run_coverage_judge(client, "section text", "", [])
    assert result == {"verdict": "sufficient", "missing": []}
    assert client.chat.completions.calls == []


def test_run_coverage_judge_returns_insufficient_with_missing_claims():
    response_json = json.dumps({
        "verdict": "insufficient",
        "missing": [{"claim": "uses a batch size of 32", "evidence": "we set batch size to 32"}],
    })
    client = _FakeChatClient(_FakeChatResponse(response_json))
    result = pb_judge.run_coverage_judge(client, "section text", "", [_leaf("leaf-a")])
    assert result["verdict"] == "insufficient"
    assert result["missing"] == [{"claim": "uses a batch size of 32", "evidence": "we set batch size to 32"}]


def test_run_coverage_judge_drops_missing_entries_with_empty_evidence():
    response_json = json.dumps({
        "verdict": "insufficient",
        "missing": [
            {"claim": "good claim", "evidence": "quoted text"},
            {"claim": "bad claim", "evidence": ""},
        ],
    })
    client = _FakeChatClient(_FakeChatResponse(response_json))
    result = pb_judge.run_coverage_judge(client, "section text", "", [_leaf("leaf-a")])
    assert result["missing"] == [{"claim": "good claim", "evidence": "quoted text"}]


def test_run_coverage_judge_downgrades_to_sufficient_when_all_missing_entries_dropped():
    response_json = json.dumps({"verdict": "insufficient", "missing": [{"claim": "x", "evidence": ""}]})
    client = _FakeChatClient(_FakeChatResponse(response_json))
    result = pb_judge.run_coverage_judge(client, "section text", "", [_leaf("leaf-a")])
    assert result == {"verdict": "sufficient", "missing": []}


def test_run_coverage_judge_treats_non_dict_response_as_sufficient():
    client = _FakeChatClient(_FakeChatResponse("[]"))
    result = pb_judge.run_coverage_judge(client, "section text", "", [_leaf("leaf-a")])
    assert result == {"verdict": "sufficient", "missing": []}


def test_run_coverage_judge_includes_branch_errors_context_in_prompt():
    client = _FakeChatClient(_FakeChatResponse('{"verdict": "sufficient", "missing": []}'))
    pb_judge.run_coverage_judge(client, "section text", "leaf-a: split-check split into 2 children.", [_leaf("leaf-a")])
    sent = client.chat.completions.calls[0]["messages"][0]["content"]
    assert "leaf-a: split-check split into 2 children." in sent


def test_run_coverage_judge_records_cost_via_tracker():
    client = _FakeChatClient(_FakeChatResponse('{"verdict": "sufficient", "missing": []}',
                                               usage=_FakeChatUsage(prompt_tokens=100, completion_tokens=20, cached_tokens=10)))
    tracker = pb_cost.CostTracker()
    pb_judge.run_coverage_judge(client, "section text", "", [_leaf("leaf-a")], model="o3-mini", tracker=tracker)
    totals = tracker.totals_for("o3-mini")
    assert totals["input"] == 100
    assert totals["output"] == 20
    assert totals["cache_read"] == 10


# ── _invoke_judge_llm ────────────────────────────────────────────────────────────

def test_invoke_judge_llm_raises_runtime_error_on_truncation():
    client = _FakeChatClient(_FakeChatResponse("{incomplete", finish_reason="length"))
    with pytest.raises(RuntimeError, match="max_completion_tokens"):
        pb_judge._invoke_judge_llm(client, "instruction", "o3-mini")


def test_invoke_judge_llm_raises_runtime_error_on_empty_content():
    client = _FakeChatClient(_FakeChatResponse(None))
    with pytest.raises(RuntimeError, match="no message content"):
        pb_judge._invoke_judge_llm(client, "instruction", "o3-mini")


def test_invoke_judge_llm_raises_runtime_error_on_api_failure():
    class _RaisingCompletions:
        def create(self, **kwargs):
            raise ValueError("boom")

    client = SimpleNamespace(chat=SimpleNamespace(completions=_RaisingCompletions()))
    with pytest.raises(RuntimeError, match="OpenAI API call failed"):
        pb_judge._invoke_judge_llm(client, "instruction", "o3-mini")


# ── format_missing_as_feedback ────────────────────────────────────────────────

def test_format_missing_as_feedback_formats_claim_and_evidence():
    missing = [{"claim": "missing metric", "evidence": "table 3 reports F1"}]
    feedback = pb_judge.format_missing_as_feedback(missing)
    assert "missing metric" in feedback
    assert "table 3 reports F1" in feedback


def test_format_missing_as_feedback_empty_list_returns_empty_string():
    assert pb_judge.format_missing_as_feedback([]) == ""
