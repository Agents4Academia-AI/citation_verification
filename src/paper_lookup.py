"""
paper_lookup — the "grounding" tool for the citation verifier.

This is the one piece that stops the agent from *guessing*. An LLM asked
"is this citation real and correct?" will happily hallucinate an answer.
Instead we look the paper up in two authoritative metadata sources and hand the
agent the *canonical* record to compare against:

  - Crossref  (api.crossref.org)  -> journals, conferences, books, most DOIs
  - arXiv     (export.arxiv.org)  -> preprints

Design borrowed from PaperArena's tools/cross_ref_lookup.py
(https://github.com/ustc-ai4science/PaperArena): take a free-text reference,
hit a scholarly API, return structured metadata. We improve on it by querying
BOTH Crossref and arXiv (PaperArena only hit arXiv despite the file name), and
by keeping this layer "dumb": it just fetches candidates. The *matching* and
*comparison* are left to the agent (the LLM), exactly as PaperArena delegates
citation-key matching to an LLM.

Pure standard library (urllib) on purpose: no `requests`/`feedparser` to
install, and every function here is unit-testable WITHOUT the LLM. Try it:

    python paper_lookup.py "Attention is all you need"
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

# Crossref asks API users to identify themselves; doing so puts you in the
# faster "polite pool". Set CROSSREF_MAILTO to your email to opt in.
CONTACT_EMAIL = os.environ.get("CROSSREF_MAILTO", "")
USER_AGENT = (
    f"citation-verifier/0.1 (mailto:{CONTACT_EMAIL})"
    if CONTACT_EMAIL
    else "citation-verifier/0.1"
)
TIMEOUT = 20


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read()


def _clean(text: str | None, limit: int = 600) -> str:
    """Strip markup/whitespace from an abstract and truncate."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)          # JATS/HTML tags
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


# ── Crossref ─────────────────────────────────────────────────────────────────
def search_crossref(query: str, rows: int = 4) -> list[dict]:
    """Look a reference up in Crossref. Best for published venues + DOIs."""
    params = urllib.parse.urlencode({"query.bibliographic": query, "rows": rows})
    if CONTACT_EMAIL:
        params += "&mailto=" + urllib.parse.quote(CONTACT_EMAIL)
    try:
        data = json.loads(_get(f"https://api.crossref.org/works?{params}"))
    except Exception as e:  # network / rate-limit / parse — fail soft
        return [{"error": f"crossref lookup failed: {e}"}]

    out = []
    for it in data.get("message", {}).get("items", []):
        authors = [
            " ".join(p for p in (a.get("given"), a.get("family")) if p)
            for a in it.get("author", [])
        ]
        year = None
        parts = (it.get("issued") or {}).get("date-parts") or [[None]]
        if parts and parts[0]:
            year = parts[0][0]
        out.append({
            "source": "crossref",
            "title": (it.get("title") or [""])[0],
            "authors": authors,
            "year": year,
            "venue": (it.get("container-title") or [""])[0],
            "type": it.get("type", ""),
            "doi": it.get("DOI", ""),
            "url": it.get("URL", ""),
            "abstract": _clean(it.get("abstract")),
        })
    return out


# ── arXiv ────────────────────────────────────────────────────────────────────
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"


def search_arxiv(query: str, max_results: int = 4) -> list[dict]:
    """Look a reference up on arXiv. Best for preprints."""
    q = urllib.parse.quote(query)
    url = (
        f"http://export.arxiv.org/api/query?search_query=all:{q}"
        f"&start=0&max_results={max_results}"
    )
    try:
        root = ET.fromstring(_get(url))
    except Exception as e:
        return [{"error": f"arxiv lookup failed: {e}"}]

    out = []
    for e in root.findall(f"{_ATOM}entry"):
        def text(tag):
            el = e.find(tag)
            return el.text.strip() if el is not None and el.text else ""

        arxiv_url = text(f"{_ATOM}id")
        out.append({
            "source": "arxiv",
            "title": re.sub(r"\s+", " ", text(f"{_ATOM}title")),
            "authors": [
                a.findtext(f"{_ATOM}name", "").strip()
                for a in e.findall(f"{_ATOM}author")
            ],
            "year": (text(f"{_ATOM}published") or "")[:4] or None,
            "venue": text(f"{_ARXIV}journal_ref") or "arXiv (preprint)",
            "type": "preprint",
            "doi": text(f"{_ARXIV}doi"),
            "url": arxiv_url,
            "arxiv_id": arxiv_url.rsplit("/", 1)[-1],
            "abstract": _clean(text(f"{_ATOM}summary")),
        })
    return out


# ── Combined entry point (this is what the agent's tool calls) ────────────────
def lookup_paper(query: str, source: str = "auto", max_results: int = 4) -> dict:
    """
    Return canonical metadata candidates for a cited reference.

    query  : the reference string or title as it appears in the bibliography.
    source : "crossref", "arxiv", or "auto" (query both and merge).

    The caller (the agent) compares these candidates against what the draft
    *claims* and decides: does it exist? are authors/venue/year correct?
    """
    candidates: list[dict] = []
    if source in ("crossref", "auto"):
        candidates += [c for c in search_crossref(query, max_results) if "error" not in c]
    if source in ("arxiv", "auto"):
        candidates += [c for c in search_arxiv(query, max_results) if "error" not in c]

    return {
        "query": query,
        "found": len(candidates),
        "candidates": candidates,
        "note": (
            "No candidates found — the paper may not exist, may be mis-titled, "
            "or may live in a source not covered here (try a web search before "
            "concluding it is fabricated)."
            if not candidates else
            "Compare each candidate's title/authors/year/venue against what the "
            "draft claims. A strong title match with mismatched authors/year/venue "
            "is a metadata error; no strong title match anywhere is a red flag."
        ),
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Attention is all you need"
    print(f"Looking up: {q!r}\n")
    print(json.dumps(lookup_paper(q), indent=2, ensure_ascii=False))
