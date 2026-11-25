"""A clipped API response is a failed evaluation, not a subset of valid verdicts."""
from types import SimpleNamespace

import pytest

from citation_verifier.backends.relevance_judge import (
    LLMRelevanceJudge,
    _max_output_tokens,
    _parse_group_verdicts,
)
from citation_verifier.errors import ModelUnavailable

PREFIX = ('[{"cite_key":"first","claims":[{"claim_id":"p1",'
          '"supports_claim":"supports","priority":"helpful",'
          '"confidence":0.8,"justification":"The source supports this claim."}]},'
          '{"cite_key":"second","claims":[')


def fake_judge(text, stop_reason):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason,
            usage=SimpleNamespace(input_tokens=100, output_tokens=512),
        )

    judge = LLMRelevanceJudge(mode="messages")
    judge._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return judge, calls


def test_truncated_prefix_is_not_accepted_even_when_parser_can_salvage_it():
    assert "first" in _parse_group_verdicts(PREFIX)
    judge, calls = fake_judge(PREFIX, "max_tokens")
    with pytest.raises(ModelUnavailable, match="output token limit"):
        judge._run_messages("system", "user", 512)
    assert judge.calls == 1
    assert judge.usage.output_tokens == 512
    assert "incomplete Messages API judgment" in judge.unavailable
    with pytest.raises(ModelUnavailable):
        judge._run_messages("system", "retry", 512)
    assert len(calls) == 1


def test_even_well_formed_json_is_rejected_when_api_reports_truncation():
    judge, _ = fake_judge("[]", "max_tokens")
    with pytest.raises(ModelUnavailable):
        judge._run_messages("system", "user", 512)


def test_complete_response_keeps_existing_transport_behavior_and_usage():
    judge, calls = fake_judge("[]", "end_turn")
    assert judge._run_messages("system", "user", 4096) == "[]"
    assert not judge.unavailable
    assert judge.usage.input_tokens == 100
    assert calls[0]["max_tokens"] == 4096


def test_grouped_output_allowance_has_headroom_and_remains_bounded():
    assert _max_output_tokens(1) >= 4096
    assert _max_output_tokens(9) >= 256 * 9
    assert _max_output_tokens(40) >= 384 * 40
    assert _max_output_tokens(1000) == 32768
