"""
inject.py — synthetic hallucination injection (controlled positives).

Natural hallucinations (from the GPTZero lists) are scarce; to give the benchmark
balanced, controlled positive examples we synthesize two kinds of corruption from
clean gold records, each producing a NEW gold record with ``labels`` updated and
``is_hallucinated=True``:

  * :func:`inject_fabrication`   — turn a real citation into a fabricated one
    (``exists='no'``): scramble the resolved match away, mark the reference as a
    plausible-but-nonexistent paper.
  * :func:`perturb_metadata`     — keep the paper real but corrupt one claimed
    field (author / year / venue / title), producing a metadata error
    (``exists='yes'`` + ``metadata_issues``).

Every synthetic record stamps its origin in ``labels.provenance`` (``synthetic:*``)
so eval can split natural vs synthetic positives and so the corruption is fully
auditable. Records returned are schema-valid :class:`CitationRecord` objects.

Pure, deterministic, offline: no network, no LLM, no randomness by default (a
``seed`` arg is accepted for reproducible variety where a choice is needed).
"""

from __future__ import annotations

import copy
import random

from citation_verifier.schema import (
    CitationRecord,
    Exists,
    Labels,
    SupportsClaim,
    derive_severity,
)

# Fields perturb_metadata knows how to corrupt.
PERTURBABLE_FIELDS = ("author", "authors", "year", "venue", "title")


def _labels_or_new(record: CitationRecord) -> Labels:
    """Return a mutable copy of the record's labels (or a fresh ``Labels``)."""
    if record.labels is None:
        return Labels()
    return Labels(**record.labels.model_dump())


def inject_fabrication(record: CitationRecord) -> CitationRecord:
    """Return a copy of ``record`` turned into a fabricated citation.

    The cited reference is kept (it still "looks" plausible in the draft) but it
    is marked as not resolving to any real paper: ``exists='no'``,
    ``resolved=None``, and ``labels.is_hallucinated=True``. Severity is re-derived
    (fabrication => high). Provenance is stamped ``synthetic:fabrication``.

    Args:
        record: a clean gold record to corrupt.

    Returns:
        A new schema-valid :class:`CitationRecord` (the input is not mutated).
    """
    rec = record.model_copy(deep=True)
    rec.exists = Exists.NO
    rec.resolved = None
    rec.metadata_issues = []
    rec.notes = "synthetic fabrication (was a real citation)"

    labels = _labels_or_new(rec)
    labels.exists = Exists.NO
    labels.is_hallucinated = True
    labels.severity = derive_severity(
        Exists.NO,
        labels.supports_claim or SupportsClaim.INCONCLUSIVE,
        labels.priority or rec.priority,
    )
    labels.provenance = "synthetic:fabrication"
    rec.labels = labels
    rec.severity = labels.severity
    return rec


def perturb_metadata(record: CitationRecord, field: str, *, seed: int = 0) -> CitationRecord:
    """Return a copy of ``record`` with one ``cited_as`` ``field`` corrupted.

    The cited paper stays real (``exists='yes'``) but the claimed metadata no
    longer matches the canonical record, creating a metadata error:
    ``labels.is_hallucinated=True`` and a ``metadata_issues`` entry describing the
    corruption. Provenance is stamped ``synthetic:perturb:<field>``.

    Args:
        record: a clean gold record whose ``cited_as`` will be corrupted.
        field: one of :data:`PERTURBABLE_FIELDS` (``author``/``authors`` are
            aliases). Unknown fields raise ``ValueError``.
        seed: makes the (otherwise deterministic) corruption reproducibly varied.

    Returns:
        A new schema-valid :class:`CitationRecord` (the input is not mutated).

    Raises:
        ValueError: if ``field`` is not perturbable.
    """
    if field not in PERTURBABLE_FIELDS:
        raise ValueError(f"field must be one of {PERTURBABLE_FIELDS!r}, got {field!r}")

    rec = record.model_copy(deep=True)
    rng = random.Random(seed)
    issue: str

    if field in ("author", "authors"):
        original = list(rec.cited_as.authors) or ["A. Author"]
        rec.cited_as.authors = ["Q. Fabricated"] + original[1:]
        issue = "authors: first author replaced (synthetic perturbation)"
    elif field == "year":
        base = rec.cited_as.year or 2020
        rec.cited_as.year = base + rng.choice([-3, -2, 2, 3])
        issue = f"year: shifted to {rec.cited_as.year} (synthetic perturbation)"
    elif field == "venue":
        rec.cited_as.venue = "Journal of Imaginary Results"
        issue = "venue: replaced with a wrong venue (synthetic perturbation)"
    else:  # title
        rec.cited_as.title = (rec.cited_as.title or "Untitled") + " (revised edition)"
        issue = "title: altered from canonical (synthetic perturbation)"

    rec.exists = Exists.YES
    rec.metadata_issues = [*copy.copy(rec.metadata_issues), issue]
    rec.notes = f"synthetic metadata perturbation: {field}"

    labels = _labels_or_new(rec)
    labels.exists = Exists.YES
    labels.is_hallucinated = True
    labels.severity = derive_severity(
        Exists.YES,
        labels.supports_claim or SupportsClaim.INCONCLUSIVE,
        labels.priority or rec.priority,
    )
    labels.provenance = f"synthetic:perturb:{field}"
    rec.labels = labels
    rec.severity = labels.severity
    return rec
