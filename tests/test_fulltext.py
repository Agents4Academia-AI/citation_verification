"""
Offline tests for the Stage-2 evidence core (`grounding/fulltext.py`): splitting a
fetched paper into sections and retrieving the few claim-relevant chunks the
relevance judge needs — no network, pure functions.
"""

from __future__ import annotations

from citation_verifier.grounding.fulltext import select_evidence_chunks, split_sections

_LATEX = r"""
\section{Introduction}
Conversational AI lets machines understand and respond to human language.
\section{Method}
We fine-tune a transformer on the WebText corpus with 1.5B parameters.
\section{Results}
On the VQA benchmark our model reaches 81.2 F1, outperforming the baseline.
\section{Conclusion}
The approach generalizes across dialogue tasks.
"""

_PLAIN = """Introduction

Conversational AI lets machines understand human language.

Experiments

We evaluate on the SQuAD dataset and report exact-match accuracy.
"""


def test_split_sections_latex_and_plain():
    secs = dict(split_sections(_LATEX))
    assert set(secs) == {"Introduction", "Method", "Results", "Conclusion"}
    assert "WebText" in secs["Method"]

    heads = [h for h, _ in split_sections(_PLAIN)]
    assert "Introduction" in heads and "Experiments" in heads


def test_split_sections_no_structure_is_single_block():
    assert split_sections("Just one blob of prose with no headings at all.") == [
        ("", "Just one blob of prose with no headings at all.")
    ]


def test_select_chunks_prefers_claim_overlap():
    sections = split_sections(_LATEX)
    # a generic (non-experimental) claim -> stays in default sections
    hits = select_evidence_chunks("Conversational AI understands human language", sections, k=2)
    assert hits and "understand" in hits[0][1].lower()


def test_select_chunks_pulls_experimental_section_only_when_claim_warrants_it():
    sections = split_sections(_LATEX)
    # claim about a metric/benchmark -> the Results section becomes in-scope and wins
    exp = select_evidence_chunks("the model achieves 81.2 F1 on the VQA benchmark", sections, k=1)
    assert exp and exp[0][0] == "Results"
    # a non-experimental claim must NOT surface the Results/Method chunks
    generic = select_evidence_chunks("conversational AI understands language", sections, k=4)
    assert all(h not in ("Results", "Method") for h, _ in generic)
