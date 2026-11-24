"""A model that cannot be reached must not pass for a model that could not decide.

Measured on a real run: the Claude Code login had expired, every judge call failed with
"401 OAuth access token has expired", and the pipeline exited 0 with an empty stderr and a
report in which all 59 pairs read "inconclusive — batch judge returned no verdict". The
Discord bot would have announced that paper as clean.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from citation_verifier.backends.relevance_judge import LLMRelevanceJudge
from citation_verifier.errors import (
    MODEL_UNAVAILABLE,
    ModelUnavailable,
    model_unavailable_errors,
    unavailable_message,
)
from citation_verifier.interfaces import RunUsage, VerificationResult
from citation_verifier.render import render_report
from citation_verifier.schema import SupportsClaim
from citation_verifier.tables import tables_from_latex, verify_table
from citation_verifier.tables.model import CellVerdict, GlossSource

_EXPIRED = "Failed to authenticate. API Error: 401 OAuth access token has expired. Re-authenticate to continue."


def _fake_sdk(*, error: str, text: str, raises: bool = True):
    """An Agent SDK whose ``query`` answers the way the CLI does when it has no access:
    a synthetic assistant message carrying the error text, an error result, and — on
    current SDKs — an exception once the stream ends."""
    calls = SimpleNamespace(n=0)

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class AssistantMessage:
        def __init__(self):
            self.content = [TextBlock(text)]
            self.error = error

    class ResultMessage:
        is_error = True
        result = text
        usage = {"input_tokens": 0, "output_tokens": 0}
        total_cost_usd = 0.0
        num_turns = 1

    async def query(prompt, options):
        calls.n += 1
        yield AssistantMessage()
        yield ResultMessage()
        if raises:
            raise Exception("Claude Code returned an error result: success")

    return SimpleNamespace(
        TextBlock=TextBlock, AssistantMessage=AssistantMessage, ResultMessage=ResultMessage,
        ClaudeAgentOptions=lambda **kw: SimpleNamespace(**kw), query=query, calls=calls,
    )


@pytest.mark.parametrize("raises", [True, False])
def test_an_expired_login_is_named_and_not_asked_twice(raises):
    judge = LLMRelevanceJudge(settings=None)
    sdk = _fake_sdk(error="authentication_failed", text=_EXPIRED, raises=raises)
    with pytest.raises(ModelUnavailable) as exc:
        asyncio.run(judge._run_query(sdk, "system", "user"))
    assert "OAuth access token has expired" in str(exc.value)
    assert "OAuth access token has expired" in judge.unavailable
    # The second call does not go out at all.
    with pytest.raises(ModelUnavailable):
        asyncio.run(judge._run_query(sdk, "system", "user"))
    assert sdk.calls.n == 1


def test_the_error_text_alone_is_enough_on_an_sdk_that_does_not_label_it():
    judge = LLMRelevanceJudge(settings=None)
    with pytest.raises(ModelUnavailable):
        asyncio.run(judge._run_query(_fake_sdk(error="", text=_EXPIRED, raises=False), "s", "u"))


def test_a_transient_failure_is_an_error_but_not_unavailability():
    judge = LLMRelevanceJudge(settings=None)
    sdk = _fake_sdk(error="server_error", text="API Error: 529 Overloaded")
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(judge._run_query(sdk, "system", "user"))
    assert not isinstance(exc.value, ModelUnavailable)
    assert judge.unavailable == ""          # worth asking again
    assert "529" in str(exc.value)          # and the CLI's own words survive


def test_unjudged_pairs_say_why_and_the_run_says_it_once(monkeypatch):
    judge = LLMRelevanceJudge(settings=None)
    judge.escalate_full_text = False
    monkeypatch.setattr(judge, "_prefetch_intros", lambda items: None)
    monkeypatch.setattr(judge, "_assemble_evidence", lambda abstract, resolved: (abstract, "L0"))

    def chunk(_chunk):
        raise judge._mark_unavailable(_EXPIRED)

    monkeypatch.setattr(judge, "_judge_citation_chunk", chunk)
    out = judge.judge_batch([
        {"cite_key": "a", "claim_id": "1", "claim": "X improves Y.", "abstract": "We improve Y."},
        {"cite_key": "b", "claim_id": "2", "claim": "Z does W.", "abstract": "Z is studied."},
    ])
    assert [v.supports_claim for v in out] == [SupportsClaim.INCONCLUSIVE] * 2
    for v in out:
        assert "not judged — the model could not be reached" in v.justification
        assert "OAuth access token has expired" in v.justification

    # The backend lifts it to one run-level error.
    from citation_verifier.backends.agentic import AgenticBackend

    backend = AgenticBackend.__new__(AgenticBackend)
    backend.judge = SimpleNamespace(judge_batch=lambda items: [], unavailable=judge.unavailable)
    result = VerificationResult(paper_id="p", backend="agentic")
    monkeypatch.setattr("citation_verifier.stages.relevance.fill_relevance_batch", lambda *a, **k: None)
    rec = SimpleNamespace(supports_claim=None, model_tier=None)
    backend._relevance_pass([rec, rec], None, None, RunUsage(), result)
    down = model_unavailable_errors(result.errors)
    assert len(down) == 1 and "relevance was NOT judged for 2 pair(s)" in down[0]
    assert "OAuth access token has expired" in down[0] and "/login" in down[0]


def test_the_report_and_the_exit_code_carry_it(monkeypatch, capsys):
    msg = unavailable_message(_EXPIRED, "relevance was NOT judged for 59 pair(s)")
    assert msg.startswith(MODEL_UNAVAILABLE)
    result = VerificationResult(paper_id="p", backend="agentic", errors=["some note", msg])
    report = render_report(result)
    head = report.split("\n\n")[1]
    assert head.startswith("> **⚠ Incomplete report") and "means **not judged**" in head
    assert "Incomplete report" not in render_report(VerificationResult(paper_id="p", backend="agentic"))

    from citation_verifier import cli

    monkeypatch.setattr("citation_verifier.orchestrator.run_verification", lambda *a, **k: result)
    assert cli.main(["1706.03762"]) == 3
    err = capsys.readouterr().err
    assert f"error: {MODEL_UNAVAILABLE}" in err and "note: some note" in err
    # …and an ordinary run still exits 0
    ok = VerificationResult(paper_id="p", backend="agentic", errors=["some note"])
    monkeypatch.setattr("citation_verifier.orchestrator.run_verification", lambda *a, **k: ok)
    assert cli.main(["1706.03762"]) == 0


# ── comparison tables ────────────────────────────────────────────────
_TABLE = r"""
\begin{table}
\caption{Comparison of methods}\label{tab:cmp}
\begin{tabular}{lcc}
\toprule
Method & Fast & Exact \\
\midrule
Alpha~\cite{alpha2020} & \cmark & \xmark \\
Beta~\cite{beta2021} & \xmark & \cmark \\
Gamma~\cite{gamma2022} & \cmark & \cmark \\
\bottomrule
\end{tabular}
\end{table}
"""


def test_a_table_stops_asking_and_stops_fetching_once_the_model_is_down():
    table = tables_from_latex(_TABLE, paper_id="p")[0]
    for d in table.dimensions:
        d.gloss, d.gloss_source = f"the method is {d.header.lower()}", GlossSource.BODY.value
    fetched, asked = [], []

    def evidence_for(key, label):
        fetched.append(key)
        return ("Some text about the method.", "http://x", {})

    def judge(payload):
        asked.append(payload)
        raise ModelUnavailable(_EXPIRED)

    report = verify_table(table, evidence_for=evidence_for, judge=judge)
    assert len(asked) == 1 and len(fetched) == 1      # one doomed call, then nothing
    cells = [f for f in report.findings if f.verdict != CellVerdict.SKIPPED.value]
    assert len(cells) == 6
    for f in cells:
        assert f.verdict == CellVerdict.UNVERIFIABLE.value
        assert "the model could not be reached" in f.justification, f.justification
        assert "never judged" in f.justification
        assert "no evidence could be retrieved" not in f.justification
    assert any("the model could not be reached" in n for n in report.notes)


def test_the_table_transport_does_not_retry_an_expired_login(monkeypatch):
    from citation_verifier.tables import llm

    monkeypatch.setattr("time.sleep", lambda s: pytest.fail("slept before a retry that cannot help"))
    calls = []

    def call():
        calls.append(1)
        raise ModelUnavailable(_EXPIRED)

    with pytest.raises(ModelUnavailable):
        llm._with_retry(call)
    assert len(calls) == 1


# ── an answer that cannot be read is not an answer of "unclear" ──────────────────────
_PROSE_THEN_JSON = (
    "Based on my reading of the complete cited paper, here is my assessment.\n\n"
    'The system links mentions to clusters with actions such as "(1 → [1)" (see [6, 7]).\n\n'
    '[{"col_index": 1, "answer": "has", "quote": "the action appends the mention \\"You\\" to '
    'cluster [1 and opens [2 for the next one", "justification": "it resolves pronouns", '
    '"confidence": 0.9}]\n\nLet me know if you need anything else [end].'
)


def test_the_answer_is_found_whatever_brackets_surround_and_fill_it():
    """One cited paper writes its actions as "[1" with no closing bracket. Counting
    brackets to find the array never got back to zero, and two cells of that row came back
    with no answer, no justification and no error."""
    from citation_verifier.tables.llm import _parse_json_array

    rows = _parse_json_array(_PROSE_THEN_JSON)
    assert len(rows) == 1 and rows[0]["answer"] == "has" and "[1 and opens [2" in rows[0]["quote"]
    assert _parse_json_array('```json\n[{"col_index": 2, "answer": "lacks"}]\n```')[0]["answer"] == "lacks"
    assert _parse_json_array("I could not decide [sorry].") == []
    assert _parse_json_array("") == []


def test_an_unreadable_answer_fails_the_cell_instead_of_blanking_it():
    from citation_verifier.tables.llm import build_cell_judge

    table = tables_from_latex(_TABLE, paper_id="p")[0]
    for d in table.dimensions:
        d.gloss, d.gloss_source = f"the method is {d.header.lower()}", GlossSource.BODY.value

    def judge_saying(text):
        fake = SimpleNamespace(mode="messages", _run_messages=lambda system, user, max_tokens, prefix="": text)
        return build_cell_judge(fake)

    report = verify_table(table, evidence_for=lambda key, label: ("Text about the method.", "http://x", {}),
                          judge=judge_saying("I am not able to answer in the requested format."))
    cells = [f for f in report.findings if f.verdict != CellVerdict.SKIPPED.value]
    assert cells and all(f.verdict == CellVerdict.UNVERIFIABLE.value for f in cells)
    assert all("pipeline failure" in f.justification for f in cells), cells[0].justification
    assert any("not a JSON array" in n for n in report.notes)
    # …while a readable one, preamble and all, is judged.
    report = verify_table(table, evidence_for=lambda key, label: ("Text about the method.", "http://x", {}),
                          judge=judge_saying(_PROSE_THEN_JSON))
    assert any(f.justification == "it resolves pronouns" or "resolves pronouns" in f.justification
               for f in report.findings)


def test_the_prose_judges_answer_is_found_past_a_preamble_with_brackets():
    from citation_verifier.backends.relevance_judge import _parse_group_verdicts

    text = ('Verdicts for citations [6, 7] follow.\n'
            '[{"cite_key": "ref-6", "claims": [{"claim_id": "c1", "supports_claim": "supports"}]}]\n'
            'Note: [7] had no evidence.')
    got = _parse_group_verdicts(text)
    assert set(got) == {"ref-6"} and got["ref-6"]["c1"].supports_claim is SupportsClaim.SUPPORTS


# ── the judge is one answer from the prompt, not an agent in the repository ──────────
def test_a_judge_call_has_no_tools_no_project_and_no_settings():
    """The Agent SDK starts a Claude Code SESSION. Left with its tools and the caller's
    working directory, the model spent more than half of one run's judge calls reading the
    project's CLAUDE.md or the verification skill before answering — and once reached for
    Bash. ``allowed_tools=[]`` pre-approves nothing; it removes nothing."""
    import dataclasses
    import os

    from citation_verifier.backends.relevance_judge import _judgement_options

    @dataclasses.dataclass
    class Options:
        system_prompt: object = None
        tools: list | None = None
        allowed_tools: list = dataclasses.field(default_factory=list)
        setting_sources: list | None = None
        cwd: str | None = None
        model: str | None = None
        max_turns: int | None = None

    got = _judgement_options(SimpleNamespace(ClaudeAgentOptions=Options), "SYSTEM", "some-model")
    assert got.tools == [] and got.allowed_tools == []     # none exist, none are approved
    assert got.setting_sources == []                       # no CLAUDE.md, skills, hooks, plugins
    assert got.cwd and os.path.realpath(got.cwd) != os.path.realpath(os.getcwd())
    assert not os.listdir(got.cwd) or all(n.startswith("system-") for n in os.listdir(got.cwd))
    assert got.system_prompt == "SYSTEM" and got.model == "some-model"

    # A system prompt handed over as a file passes straight through.
    as_file = {"type": "file", "path": "/tmp/x.txt"}
    assert _judgement_options(SimpleNamespace(ClaudeAgentOptions=Options), as_file, "m").system_prompt == as_file

    # An older SDK that knows none of the new fields still gets a valid call.
    @dataclasses.dataclass
    class OldOptions:
        system_prompt: object = None
        allowed_tools: list = dataclasses.field(default_factory=list)
        model: str | None = None
        max_turns: int | None = None

    old = _judgement_options(SimpleNamespace(ClaudeAgentOptions=OldOptions), "SYSTEM", "m")
    assert old.system_prompt == "SYSTEM" and old.allowed_tools == []


def test_the_installed_sdk_accepts_the_isolated_options():
    sdk = pytest.importorskip("claude_agent_sdk")
    from citation_verifier.backends.relevance_judge import _judgement_options

    got = _judgement_options(sdk, "SYSTEM", "claude-opus-4-8")
    assert got.tools == [] and got.setting_sources == [] and got.cwd
