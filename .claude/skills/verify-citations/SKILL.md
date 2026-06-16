---
name: verify-citations
description: Use when asked to verify, check, or audit the citations/references in
  a paper draft or preprint — whether each cited work is real, whether its
  authors/venue/year are correct, and whether it actually supports the claim it
  is attached to. Produces a fixed citation-verification table.
---

# Verify the citations in a paper

Your job is to catch citation problems *before a reviewer does*: fabricated or
wrong references, incorrect metadata, and citations that don't actually support
the claim they're attached to.

## The one rule that matters most

**Never decide correctness from memory.** An LLM "remembering" that a paper
exists is exactly the failure mode you are here to catch. Every statement about
whether a paper is real, and about its authors / venue / year, MUST be grounded
in a `lookup_paper` result or a web result you actually retrieved. If you cannot
verify something, label it `unverified` — never guess.

## Workflow

1. **Read the draft.** Use `Read` on the PDF. Locate the reference list / bibliography
   AND the in-text citations.
2. **Build the citation list.** For each reference, record: (a) the full reference
   string as written, and (b) every place it is cited in the body, with the exact
   sentence/claim it supports. A reference cited in several places may be
   obligatory in one spot and background in another — note each use.
3. **Verify correctness** — call `lookup_paper` with the *fullest* reference string
   you have (authors + title + year + venue; a bare title is noisy). Compare the
   returned canonical record(s) against what the draft claims (see rubric below).
   If `lookup_paper` returns nothing, try `WebSearch` before concluding a paper is
   fabricated — it may be a book, thesis, workshop, or non-indexed venue.
4. **Verify relevance** — for each citation *use*, judge whether the cited paper
   actually supports that specific claim. Base this on evidence you retrieved: the
   abstract from `lookup_paper`, or `WebFetch` the paper's page for more. Quote the
   supporting (or contradicting) snippet. If you couldn't retrieve enough, mark
   `unverified`.
5. **Assign priority** — obligatory vs. helpful (rubric below).
6. **(Optional) Comparison objectiveness** — for *obligatory* citations that are
   competing methods/baselines, check whether the draft actually compares against
   them in its experiments. Flag top-priority baselines that are cited but never
   compared.
7. **Emit the table and summary** (format below).

## Correctness rubric

For each reference decide **Exists?** = `yes` / `no` / `unverified`, then list
**metadata issues** by comparing claimed vs. canonical:

- **Exists = no** — no strong title match in Crossref, arXiv, or on the web. This
  is the most serious finding (likely fabricated or hallucinated). State where you
  looked.
- **Author errors** — wrong first author, missing/extra authors, misspellings,
  wrong "et al." attribution.
- **Venue errors** — wrong conference/journal; or cited as a formal publication
  when it is only an arXiv preprint (or vice-versa); citing a workshop as the main
  conference.
- **Year errors** — any mismatch. Watch the common arXiv-vs-published year gap,
  and ignore mirror/repost DOIs that show a much later year than the original.
- **Wrong-paper** — the title/DOI cited resolves to a *different* paper than the
  one the claim clearly intends.
- Cross-check sources: if Crossref and arXiv disagree, prefer the original/canonical
  record and say which you trusted.

## Relevance rubric

**Supports claim?** = one of:

- `supports` — the cited paper clearly backs the specific claim (quote evidence).
- `partial` — related but weaker than the claim implies (e.g. claim says "proven",
  paper only shows it empirically on one dataset).
- `does not` — the paper does not support, or even contradicts, the claim. Serious.
- `unverified` — you could not retrieve enough of the cited paper to judge.

## Priority rubric

**Priority** = `obligatory` or `helpful`:

- **obligatory** — the claim depends on this specific source: a method being used
  or extended, a baseline/competing method, a dataset used, a specific quantitative
  result, a direct quote, or "X showed that …". If this citation is wrong or
  missing, the claim is unsupported.
- **helpful** — background / breadth: "see also", surveys, general context,
  motivation. Removing it would not break a specific claim.

A wrong **obligatory** citation is high severity; a wrong **helpful** one is low.

## Output format

First a one-line scope statement (how many references; if there are many, which you
deep-checked for relevance and which you only checked for existence — never silently
cap; say what you sampled).

Then the table — exactly these columns:

| # | Citation (authors, short title, year) | Cited where (the claim) | Exists? | Metadata issues | Supports claim? | Priority | Issue / Severity |
|---|---|---|---|---|---|---|---|

- Keep each cell short; put detail (quoted evidence, the canonical record) in
  footnotes under the table if needed.
- **Issue / Severity**: `high` (fabricated, wrong-paper, or wrong obligatory
  citation / `does not` support), `medium` (metadata error, `partial` support on
  an obligatory cite), `low` (minor metadata or helpful-cite issue), `ok` (clean).

Then a **Summary**: counts of references checked, not-found, metadata errors,
relevance problems; and a short **Fix before submission** list of the high-severity
items in priority order.

## Style

- Be specific and verifiable. Quote evidence; cite the DOI/arXiv id you matched.
- No hedging adjectives. If something is wrong, say what is wrong and how to fix it.
- Distinguish "I verified this is wrong" from "I could not verify" — they are
  different and the table must not blur them.
