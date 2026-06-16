"""
comparison — STEP 3 (Objectiveness of Comparison).  [RESERVED SEAM]

For *obligatory* citations that are competing methods/baselines, this stage will
check whether the draft actually compares against them in its experiments (a
top-priority baseline cited but never compared is a red flag a reviewer will
catch). It fills exactly one field:

    record.compared_against : bool | None   (None = n/a / unchecked)

This is deliberately a no-op for the MVP per docs/decisions-phy.md: the schema
field exists now so comparison-objectiveness lands later without a breaking
change, but the implementation is reserved. :func:`fill_comparison` is a safe,
import-clean :class:`citation_verifier.interfaces.StageFn` that returns the record
unchanged.

When implemented (post-MVP), it will:
  - run only for ``priority == "obligatory"`` records;
  - read the paper's results/experiments section from ``paper_source``;
  - look for the cite_key / method name in comparison tables;
  - set ``compared_against`` True/False with quoted evidence.
"""

from __future__ import annotations

from typing import Any

from ..schema import CitationRecord


def fill_comparison(record: CitationRecord, *, paper_source: Any = None) -> CitationRecord:
    """Set ``compared_against`` for obligatory baselines. RESERVED — MVP no-op.

    Args:
        record: the record to (eventually) annotate.
        paper_source: optional :class:`citation_verifier.interfaces.PaperSource`
            giving access to the paper's experiments section. Unused in the MVP.

    Returns:
        The record unchanged (``compared_against`` stays ``None`` = n/a/unchecked).
        Never raises — satisfies the degrade-not-crash StageFn contract.
    """
    # Reserved seam: leave compared_against = None (n/a) until STEP 3 is built.
    return record
