"""On-disk cache for a column's model-written gloss.

The glosser is the one non-deterministic step upstream of everything else, and its output
becomes the retrieval query: chunk scoring is pure lexical overlap, so a gloss that says
"faithful to the sampled meta-training tasks" and one that says "faithful to the true task
distribution" — the same claim, different words — select different passages, hand the
judge different evidence, and reach different verdicts. Measured: one paper judged three
different ways across three runs of identical code, with 5 of 12 cells changing.

Caching the gloss makes the whole chain downstream of it deterministic. The key includes
the material the glosser was shown, so a column whose retrieved passages changed is
re-glossed rather than answered from a stale entry.
"""

from __future__ import annotations

import json
from typing import Any

from ..diskcache import clear as _clear
from ..diskcache import key_for, namespace_dir, read_json, write_json

__all__ = ["cache_dir", "clear", "read", "write"]

_VERSION = "v1"  # bump when the glosser prompt changes materially


def cache_dir():
    """Where glosses are stored, or ``None`` when caching is off."""
    return namespace_dir("gloss", _VERSION)


def _key(paper_id: str, table_id: str, payload: dict) -> str:
    """Identify one column by where it is AND by what the glosser was shown."""
    material = json.dumps(
        {
            "header": payload.get("header"),
            "caption": payload.get("caption"),
            "siblings": sorted(payload.get("siblings") or []),
            "legend": list(payload.get("legend") or []),
            "snippets": [s.get("quote") for s in payload.get("snippets") or []],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return key_for(paper_id or "", table_id or "", payload.get("header") or "", material)


def read(paper_id: str, table_id: str, payload: dict) -> dict | None:
    """The cached ``{gloss, test_question}`` for this column, or ``None``."""
    got = read_json(cache_dir(), _key(paper_id, table_id, payload))
    return got if isinstance(got, dict) else None


def write(paper_id: str, table_id: str, payload: dict, answer: Any) -> None:
    """Store one column's gloss. An empty gloss is a real answer and is cached too.

    Unlike a failed lookup elsewhere in the project, "the passages do not pin this term
    down" is a conclusion about the paper, not a transient failure, so it is worth keeping.
    """
    if not isinstance(answer, dict):
        return
    write_json(
        cache_dir(),
        _key(paper_id, table_id, payload),
        {"gloss": (answer.get("gloss") or "").strip(),
         "test_question": (answer.get("test_question") or "").strip()},
    )


def clear() -> int:
    """Delete every cached gloss; returns how many were removed."""
    return _clear(cache_dir())
