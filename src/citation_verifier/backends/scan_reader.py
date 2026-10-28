"""
backends/scan_reader.py — a model reads the text off a scanned PDF.

A PDF whose pages are images — an old technical report, a scanned book chapter — has no
text layer, so the cited work's own words never reach the relevance judge and the pair
ends "inconclusive" with the paper in hand. :class:`ScanReader` renders the pages and asks
a model to transcribe them, verbatim; :func:`citation_verifier.grounding.fulltext.reading_scans`
installs it for a run, and the full-text layer labels what it returns as a transcription
(``SCAN_LABEL``) and caches it on disk, so a scan is read once.

The model only transcribes: it is not told which work it is reading or what the citing
paper claims, so neither can steer the reading, and it is told to mark what it cannot
read rather than fill it in.

Two transports, as for the relevance judge: the Messages API when an
``ANTHROPIC_API_KEY`` is configured, else an Agent SDK session (the Claude Code
subscription) with no tools and the page images in the user message. Usage is accounted
in :attr:`ScanReader.usage`. An unreachable model raises
:class:`~citation_verifier.errors.ModelUnavailable` and later calls fail at once.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from typing import Any

from ..config import apply_auth
from ..errors import ModelUnavailable
from ..interfaces import RunUsage
from .usage import usage_from_message, usage_from_result_message

__all__ = ["ScanReader", "render_pages"]

_MAX_PAGES = 20  # pages read from one scan (the rest is noted as unread)
_PAGES_PER_CALL = 5  # page images in one request
_TOKENS_PER_PAGE = 1500  # transcription budget per page
_LONG_EDGE = 1568  # pixels: the size a model reads an image at without downscaling it

_SYSTEM = (
    "You transcribe scanned document pages. Write out the text printed on the pages, in "
    "reading order, starting each page with a line [page N]. Keep the wording, numbers and "
    "symbols exactly as printed; write mathematics in LaTeX. Do not summarise, correct, "
    "translate, complete or comment, and add nothing that is not on the page. Where "
    "something cannot be read, write [illegible]. Answer with the transcription only."
)


class ScanReader:
    """Callable ``(pdf_bytes, title="") -> text``: a model transcribes a scanned PDF.

    ``title`` is accepted for the full-text layer's interface and not shown to the model.
    Returns ``""`` when the pages cannot be rendered (PyMuPDF missing, a broken file).
    """

    def __init__(self, settings: Any = None, *, model: str | None = None, max_pages: int | None = None) -> None:
        self.settings = settings
        self.model = model or _setting(settings, "model_bulk", "claude-haiku-4-5-20251001")
        self.max_pages = int(max_pages or _setting(settings, "scan_max_pages", _MAX_PAGES) or _MAX_PAGES)
        self.mode = "messages" if _has_api_key(settings) else "sdk"
        self.usage = RunUsage(backend="agentic", model=self.model)
        self.calls = 0
        self.unavailable = ""
        self._lock = threading.Lock()
        self._client: Any = None

    def drain_usage(self) -> RunUsage:
        """The usage accumulated since the last drain (and reset it)."""
        with self._lock:
            usage, self.usage = self.usage, RunUsage(backend="agentic", model=self.model)
        return usage

    def __call__(self, data: bytes, *, title: str = "") -> str:
        if self.unavailable:
            raise ModelUnavailable(self.unavailable)
        pages, total = render_pages(data, self.max_pages)
        if not pages:
            return ""
        parts: list[str] = []
        for first in range(0, len(pages), _PAGES_PER_CALL):
            batch = pages[first : first + _PAGES_PER_CALL]
            parts.append(self._transcribe(batch, first + 1).strip())
        text = "\n\n".join(p for p in parts if p)
        if text and total > len(pages):
            text += f"\n\n[pages {len(pages) + 1}–{total} of the scan were not read]"
        return text

    # ── one request: a few page images in, their text out ─────────
    def _transcribe(self, images: list[bytes], first_page: int) -> str:
        last = first_page + len(images) - 1
        ask = (
            f"These are pages {first_page}–{last} of a scanned document. Transcribe them."
            if len(images) > 1 else f"This is page {first_page} of a scanned document. Transcribe it."
        )
        blocks: list[dict] = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(img).decode()},
            }
            for img in images
        ]
        blocks.append({"type": "text", "text": ask})
        max_tokens = _TOKENS_PER_PAGE * len(images) + 500
        if self.mode == "messages":
            return self._run_messages(blocks, max_tokens)
        import claude_agent_sdk as sdk

        apply_auth(self.settings)
        return asyncio.run(self._run_query(sdk, blocks))

    def _run_messages(self, blocks: list[dict], max_tokens: int) -> str:
        client = self._anthropic_client()
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=_SYSTEM,
                messages=[{"role": "user", "content": blocks}],
            )
        except Exception as exc:
            if type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError"):
                raise self._mark_unavailable(f"{type(exc).__name__}: {exc}") from exc
            raise
        with self._lock:
            self.usage.add(usage_from_message(resp, self.model, "agentic"))
            self.calls += 1
        return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")

    def _anthropic_client(self) -> Any:
        with self._lock:
            if self._client is None:
                import anthropic

                key = _setting(self.settings, "anthropic_api_key", None)
                self._client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
            return self._client

    async def _run_query(self, sdk: Any, blocks: list[dict]) -> str:
        options = _reading_options(sdk, self.model)

        async def prompt():
            # Streaming input: the only way to hand a session images without tools.
            yield {
                "type": "user",
                "message": {"role": "user", "content": blocks},
                "parent_tool_use_id": None,
                "session_id": "default",
            }

        out: list[str] = []
        usage: RunUsage | None = None
        failure = ""
        try:
            async for message in sdk.query(prompt=prompt(), options=options):
                if isinstance(message, sdk.AssistantMessage):
                    if getattr(message, "error", None):
                        failure = failure or str(message.error)
                    for block in message.content:
                        if isinstance(block, sdk.TextBlock):
                            out.append(block.text)
                elif isinstance(message, sdk.ResultMessage):
                    usage = usage_from_result_message(message, self.model, "agentic")
                    if getattr(message, "is_error", False):
                        failure = failure or str(getattr(message, "result", "") or "error result")
        except Exception as exc:  # noqa: BLE001 — reported below
            failure = failure or str(exc)
        with self._lock:
            if usage is not None:
                self.usage.add(usage)
            self.calls += 1
        if failure:
            said = " ".join(failure.split())[:300]
            if any(w in said.lower() for w in ("auth", "401", "403", "login", "credit", "billing")):
                raise self._mark_unavailable(said)
            raise RuntimeError(f"scan reading failed: {said}")
        return "".join(out)

    def _mark_unavailable(self, reason: str) -> ModelUnavailable:
        with self._lock:
            self.unavailable = self.unavailable or " ".join(str(reason).split())[:300]
        return ModelUnavailable(self.unavailable)


def render_pages(data: bytes, max_pages: int = _MAX_PAGES) -> tuple[list[bytes], int]:
    """``(png_per_page, page_count)`` for the first ``max_pages`` pages of a PDF.

    Greyscale, long edge :data:`_LONG_EDGE` pixels. ``([], 0)`` when PyMuPDF is not
    installed or the bytes do not open.
    """
    try:
        import fitz  # noqa: PLC0415 — PyMuPDF, optional
    except Exception:  # noqa: BLE001
        return [], 0
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            total = doc.page_count
            out: list[bytes] = []
            for page in list(doc)[: max(0, max_pages)]:
                rect = page.rect
                zoom = min(_LONG_EDGE / max(rect.width, rect.height, 1.0), 4.0)
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY)
                out.append(pix.tobytes("png"))
            return out, total
    except Exception:  # noqa: BLE001 — a broken file is not readable, not an error
        return [], 0


def _reading_options(sdk: Any, model: str) -> Any:
    """Agent-SDK options for one transcription: no tools, no project settings, an empty
    working directory (as for the relevance judge's sessions), and no thinking — copying
    text needs none, and a session's default thinking was three quarters of what the
    first transcription of a 4-page scan cost."""
    import dataclasses
    import tempfile
    from pathlib import Path

    cwd = Path(tempfile.gettempdir()) / "cverify-scan"
    try:
        cwd.mkdir(parents=True, exist_ok=True)
    except OSError:
        cwd = Path(tempfile.gettempdir())
    wanted: dict[str, Any] = {
        "system_prompt": _SYSTEM,
        "tools": [],
        "allowed_tools": [],
        "setting_sources": [],
        "cwd": str(cwd),
        "model": model,
        "max_turns": 2,
        "thinking": {"type": "disabled"},
    }
    try:
        known = {f.name for f in dataclasses.fields(sdk.ClaudeAgentOptions)}
    except TypeError:
        known = set(wanted)
    return sdk.ClaudeAgentOptions(**{k: v for k, v in wanted.items() if k in known})


def _has_api_key(settings: Any) -> bool:
    import os

    return bool(_setting(settings, "anthropic_api_key", None) or os.environ.get("ANTHROPIC_API_KEY"))


def _setting(settings: Any, attr: str, default: Any) -> Any:
    if settings is None:
        return default
    if isinstance(settings, dict):
        return settings.get(attr, default)
    return getattr(settings, attr, default)
