"""
venues.py — are these two strings the same publication venue?

A venue is written a dozen ways. A bibliography says "NeurIPS", "NIPS", "Advances in
Neural Information Processing Systems 35" or "Adv. Neural Inf. Process. Syst."; the record
it is checked against says "Neural Information Processing Systems" (Semantic Scholar),
"NeurIPS" (DBLP) or "Proceedings of the 36th Conference on Neural Information Processing
Systems (NeurIPS 2022)" (Crossref). Compared as strings these share almost nothing, and
every correct citation came back as a "venue mismatch".

Three layers, most certain first:

1. **Strip the decoration.** Years, ordinals ("36th", "Thirty-sixth"), volume and page
   numbers carry no identity and are dropped before anything is compared.
2. **The alias table** (``data/venues.yaml``): every spelling of a known venue — acronym,
   former name, full name — maps to one key. A string is scanned for the longest aliases
   first, so "International Conference on Machine Learning and Applications" is ICMLA and
   never ICML, and a name is only recognised when it is not the inside of a longer one.
3. **Rules for everything the table does not list.** An ISO-4 style abbreviation is
   aligned word by word with the name it shortens ("Trans. Pattern Anal. Mach. Intell." /
   "Transactions on Pattern Analysis and Machine Intelligence"); this also recognises the
   abbreviated form of every name IN the table, which is why the table lists no
   abbreviations.

What the comparison is allowed to conclude follows from how much it knows
(:func:`compare_venues`): two different KNOWN venues are a mismatch; a proceedings series
("PMLR", "LNCS") or a preprint record on the canonical side can contradict nothing; a
conference and the journal that prints its proceedings (SIGGRAPH / ACM Transactions on
Graphics) are one venue; and when either string is unknown the answer is "unknown", left
to the caller's softer tests. A wrong recognition can therefore only cost a missed
mismatch, never invent one, unless BOTH strings are recognised as different known venues.

Inspect a string or a pair from the shell::

    python -m citation_verifier.venues "Adv. Neural Inf. Process. Syst." "NeurIPS"

Pure and offline. ``pyyaml`` is a core dependency; without the table the rules still work.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

__all__ = [
    "compare_venues", "venue_ids", "venue_kind", "is_workshop", "is_preprint",
    "is_abbreviation_of", "SAME", "DIFFERENT", "UNKNOWN", "UNCOMPARABLE",
]

SAME, DIFFERENT, UNKNOWN, UNCOMPARABLE = "same", "different", "unknown", "uncomparable"

_REAL = ("conference", "journal")

# Words with no identity of their own.
_STOP = frozenset("of on in the and for a an to at de la le du des et".split())
_ORDINAL_WORDS = frozenset(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth "
    "thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth "
    "thirtieth fortieth fiftieth sixtieth seventieth eightieth ninetieth twenty thirty forty "
    "fifty sixty seventy eighty ninety".split()
)
# Wrappers around a name that are not part of it (for abbreviation alignment).
_WRAPPER = frozenset("proceedings proc procs ieee acm cvf".split())
# What may sit directly before / after a recognised name without making it a longer,
# different name: "Proceedings of the …", "IEEE …", "… Workshops", "… Companion".
_LEFT_OK = frozenset(
    "proceedings proc procs of the in on at annual international intl joint acm ieee cvf "
    "usenix siam aaai sigkdd sigir sigmod sigchi sigplan sigsoft sigcomm".split()
)
_RIGHT_OK = frozenset(
    "workshop workshops companion proceedings findings extended abstracts track tracks "
    "datasets demos demo demonstrations tutorials volume vol part main supplement".split()
)
_WORKSHOP = frozenset({"workshop", "workshops"})
# "… and the 9th International Joint Conference on …": what may stand between the two
# names of a joint venue.
_JOINT = frozenset("and with the jointly in conjunction co located".split())
# A locator word is noise only when a number follows it ("volume 645", "pages 633-638").
_LOCATOR = frozenset("vol volume pages page pp no nr number issue".split())


@dataclass(frozen=True)
class _Tok:
    norm: str   # lowercase, diacritics folded
    raw: str    # as written (diacritics folded, case kept)
    start: int
    end: int


def _fold(text: str) -> str:
    """Diacritics folded, case kept; ``&`` spelled out so "S&P" and "S and P" agree, and a
    dotted initialism made one word ("U.S.A." -> "USA")."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    plain = re.sub(r"\b(?:[A-Za-z]\.){2,}", lambda m: m.group(0).replace(".", ""), plain)
    return plain.replace("&", " and ")


def _is_noise(raw: str) -> bool:
    """A token that carries no identity: a year, a volume, an ordinal."""
    low = raw.lower()
    return bool(
        low.isdigit() or re.fullmatch(r"\d+(?:st|nd|rd|th)", low) or low in _ORDINAL_WORDS
    )


def _tokens(text: str) -> tuple[str, list[_Tok]]:
    folded = _fold(text)
    found = list(re.finditer(r"[A-Za-z0-9]+", folded))
    toks: list[_Tok] = []
    for i, m in enumerate(found):
        raw = m.group(0)
        if _is_noise(raw):
            continue
        nxt = found[i + 1].group(0) if i + 1 < len(found) else ""
        if raw.lower() in _LOCATOR and nxt[:1].isdigit():
            continue  # "volume 645", "pp. 633-638": where in the venue, not which venue
        toks.append(_Tok(raw.lower(), raw, m.start(), m.end()))
    return folded, toks


def _words(text: str) -> tuple[str, ...]:
    return tuple(t.norm for t in _tokens(text)[1])


# ───────────────────────────────────────────────────────────────
# The table
# ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class _Table:
    kind: dict[str, str]                               # id -> kind
    acronyms: dict[str, tuple[str, frozenset[str]]]    # lowercase acronym -> (id, listed forms)
    phrases: tuple[tuple[tuple[str, ...], str], ...]   # distinctive names, longest first
    exact: dict[tuple[str, ...], str]                  # short names (whole string only) -> id
    names: tuple[tuple[tuple[str, ...], str], ...]     # every name's content words, id
    hosts: dict[str, frozenset[str]]                   # journal id -> conferences it prints


@lru_cache(maxsize=1)
def _table() -> _Table:
    """Load ``data/venues.yaml``. An unreadable table is an empty one: the rules still run."""
    try:
        from importlib import resources

        import yaml

        raw = resources.files("citation_verifier").joinpath("data/venues.yaml").read_text("utf-8")
        entries = (yaml.safe_load(raw) or {}).get("venues") or {}
    except Exception:  # noqa: BLE001 — no table is survivable
        entries = {}
    kind: dict[str, str] = {}
    forms: dict[str, tuple[str, set[str]]] = {}
    phrases: list[tuple[tuple[str, ...], str]] = []
    exact: dict[tuple[str, ...], str] = {}
    names: list[tuple[tuple[str, ...], str]] = []
    hosts: dict[str, frozenset[str]] = {}
    for vid, entry in entries.items():
        vid, entry = str(vid), entry or {}
        kind[vid] = str(entry.get("kind") or "conference")
        if entry.get("hosts"):
            hosts[vid] = frozenset(str(h) for h in entry["hosts"])
        for acr in entry.get("acronyms") or []:
            owner, listed = forms.setdefault(str(acr).lower(), (vid, set()))
            if owner == vid:
                listed.add(str(acr))
        for name in entry.get("names") or []:
            words = _words(str(name))
            if not words:
                continue
            # A name of three distinctive words is recognisable wherever it stands. A
            # shorter one ("Nature", "Machine Learning", "Journal of the ACM") is the
            # inside of too many other names, and counts only as the whole string.
            if len([w for w in words if w not in _STOP]) >= 3:
                phrases.append((words, vid))
            else:
                exact.setdefault(_core(words), vid)
            content = _content(words)
            if len(content) >= 2:
                names.append((content, vid))
    phrases.sort(key=lambda p: len(p[0]), reverse=True)
    acronyms = {low: (vid, frozenset(listed)) for low, (vid, listed) in forms.items()}
    return _Table(kind, acronyms, tuple(phrases), exact, tuple(names), hosts)


def _core(words: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(w for w in words if w != "the") or words


def venue_kind(venue_id: str) -> str:
    """``conference`` / ``journal`` / ``preprint`` / ``series`` for a table key (else "")."""
    return _table().kind.get(venue_id, "")


# ───────────────────────────────────────────────────────────────
# Abbreviations
# ───────────────────────────────────────────────────────────────
def _content(words: tuple[str, ...]) -> tuple[str, ...]:
    """The words that identify a name: no function words, no publisher/"Proceedings" wrapper."""
    return tuple(w for w in words if w not in _STOP and w not in _WRAPPER)


def _abbreviates(short: str, full: str) -> bool:
    """``short`` is how ``full`` is abbreviated: same first letter, letters in order.

    Truncation ("Trans" / "Transactions", "Recognit" / "Recognition") and contraction
    ("Natl" / "National", "Intl" / "International") are both that.
    """
    if short == full:
        return True
    if len(short) >= len(full) or short[0] != full[0]:
        return False
    it = iter(full)
    return all(ch in it for ch in short)


def _aligned(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Two word lists that are one name, one of them abbreviated word by word."""
    if len(a) != len(b) or len(a) < 2 or a == b:
        return False
    solid = 0
    for x, y in zip(a, b, strict=True):
        short, full = (x, y) if len(x) <= len(y) else (y, x)
        if not _abbreviates(short, full):
            return False
        solid += len(short) >= 3
    return solid * 2 >= len(a)  # single letters alone ("J. A. B.") identify nothing


def is_abbreviation_of(a: str, b: str) -> bool:
    """True when one string is the other with its words abbreviated.

    "IEEE Trans. Pattern Anal. Mach. Intell." / "IEEE Transactions on Pattern Analysis and
    Machine Intelligence"; "Proc. Int. Conf. Mach. Learn." / "International Conference on
    Machine Learning". Function words and a "Proceedings" / publisher wrapper are ignored.
    """
    return _aligned(_content(_words(a)), _content(_words(b)))


# ───────────────────────────────────────────────────────────────
# Recognising a venue
# ───────────────────────────────────────────────────────────────
def _acronym_id(tok: _Tok, alone: bool, table: _Table) -> tuple[str, bool] | None:
    """The venue an acronym-shaped token names, and whether it is the workshop form.

    An acronym is recognised written as the table lists it, in capitals, or when it is
    the whole string — so "SODA", "COLT" and "CHI" are venues and the words "soda",
    "Colt" and "chi" inside a longer name are not. "CVPRW" is CVPR's workshops.
    """
    def lookup(raw: str) -> str | None:
        hit = table.acronyms.get(raw.lower())
        if hit is None:
            return None
        vid, listed = hit
        return vid if (raw in listed or raw.isupper() or alone) else None

    vid = lookup(tok.raw)
    if vid is not None:
        return vid, False
    if tok.raw.isupper() and tok.raw.endswith("W") and len(tok.raw) >= 4:
        vid = lookup(tok.raw[:-1])
        if vid is not None:
            return vid, True
    return None


@lru_cache(maxsize=4096)
def _recognise(text: str) -> tuple[frozenset[str], bool]:
    """``(table keys named in text, whether it names a workshop)``."""
    table = _table()
    folded, toks = _tokens(text)
    if not toks:
        return frozenset(), False
    norms = [t.norm for t in toks]
    used = [False] * len(toks)
    found: set[str] = set()
    workshop = False
    pending: list[tuple[int, int, str]] = []  # names stopped only by what follows them

    # 0) a short name ("Nature", "Neural Computation", "ACL Anthology") is the WHOLE
    #    string or nothing: "Nature" must not be found in "Nature Communications", nor the
    #    conference in "ACL Anthology". "Whole" allows a "Proceedings of the" in front and
    #    a parenthesised place behind ("Science (New York, N.Y.)").
    outside = tuple(t.norm for t in _tokens(re.sub(r"\([^()]*\)", " ", text))[1])
    for words in (tuple(norms), outside):
        core = _core(words)
        for lead in ((), ("in",), ("proceedings", "of"), ("proc", "of"), ("proc",)):
            rest = core[len(lead):] if core[: len(lead)] == lead else None
            if rest and rest in table.exact:
                return frozenset({table.exact[rest]}), False

    def gap(i: int, j: int) -> str:
        return folded[toks[i].end : toks[j].start]

    def boundary_left(i: int) -> bool:
        if i == 0:
            return True
        prev = toks[i - 1]
        return bool(gap(i - 1, i).strip()) or prev.norm in _LEFT_OK or (
            prev.raw.isupper() and len(prev.raw) >= 2
        )

    def boundary_right(j: int) -> bool:  # j = index just past the match
        if j >= len(toks):
            return True
        nxt = toks[j]
        return bool(gap(j - 1, j).strip()) or nxt.norm in _RIGHT_OK or (
            nxt.raw.isupper() and len(nxt.raw) >= 2
        )

    # 1) names, longest first, each consuming its words — so a name is never found
    #    inside a longer name that is a different venue.
    for words, vid in table.phrases:
        n = len(words)
        for i in range(len(norms) - n + 1):
            if any(used[i : i + n]) or tuple(norms[i : i + n]) != words:
                continue
            if not boundary_left(i):
                continue
            if boundary_right(i + n):
                found.add(vid)
                used[i : i + n] = [True] * n
            else:
                pending.append((i, n, vid))

    # 2) acronyms among what is left.
    alone = len(toks) == 1
    for i, tok in enumerate(toks):
        if used[i]:
            continue
        hit = _acronym_id(tok, alone, table)
        if hit is not None:
            found.add(hit[0])
            workshop = workshop or hit[1]
            used[i] = True

    # 2b) a joint venue: "<name> and the 9th <name>". A name followed by a connective is
    #     that name only when what the connective leads to is ANOTHER recognised venue —
    #     "… Machine Learning and Applications" leads nowhere and stays unrecognised.
    progress = True
    while progress and pending:
        progress = False
        for cand in list(pending):
            i, n, vid = cand
            if any(used[i : i + n]):
                pending.remove(cand)
                continue
            j, linked = i + n, False
            while j < len(toks) and not used[j] and norms[j] in _JOINT:
                linked = linked or norms[j] in ("and", "with", "conjunction")
                j += 1
            if linked and j < len(toks) and used[j]:
                found.add(vid)
                used[i : i + n] = [True] * n
                pending.remove(cand)
                progress = True

    # A workshop is named by a "workshop" that is not part of a listed name (one metadata
    # source files a whole conference under "… Workshops").
    workshop = workshop or any(
        norms[i] in _WORKSHOP for i in range(len(toks)) if not used[i]
    )

    # 3) the abbreviated form of a listed name.
    if not found:
        content = _content(tuple(norms))
        for words, vid in table.names:
            if _aligned(content, words):
                found.add(vid)
                break
    return frozenset(found), workshop


def venue_ids(text: str) -> frozenset[str]:
    """The table keys a venue string names — empty when the table does not know it.

    More than one for a joint venue ("EMNLP-IJCNLP") or a string that names both the
    conference and the series it was printed in ("… ICML, PMLR 139").
    """
    return _recognise(text or "")[0]


def is_workshop(text: str) -> bool:
    """True when a venue string names a workshop ("ICML Workshop on …", "CVPRW")."""
    return _recognise(text or "")[1]


def is_preprint(text: str) -> bool:
    """True when a venue string names a preprint server and no actual venue."""
    ids = venue_ids(text)
    kinds = {_table().kind.get(i) for i in ids}
    return "preprint" in kinds and not (kinds & set(_REAL))


def compare_venues(cited: str, canonical: str) -> str:
    """How a cited venue stands against the venue of the record it resolved to.

    Returns one of:

    * ``SAME`` — both name the same known venue (or a conference and the journal that
      prints its proceedings: SIGGRAPH / ACM Transactions on Graphics).
    * ``DIFFERENT`` — both are known and they are not the same venue; or the work was
      cited as a preprint and the record gives a venue; or it was cited as a main
      conference and the record says it appeared at that conference's workshop.
    * ``UNCOMPARABLE`` — one side cannot contradict the other: a proceedings series names
      the volume and not the venue, and a preprint record says where a copy lives, not
      that the work was never published.
    * ``UNKNOWN`` — at least one string is not a venue the table knows; the caller
      decides with its own string tests.

    Asymmetric on purpose: the arguments are not interchangeable.
    """
    table = _table()
    kinds = table.kind

    def split(ids: frozenset[str]) -> tuple[set[str], set[str], set[str]]:
        real = {i for i in ids if kinds.get(i) in _REAL}
        pre = {i for i in ids if kinds.get(i) == "preprint"}
        series = {i for i in ids if kinds.get(i) == "series"}
        return real, pre, series

    def compatible(a: str, b: str) -> bool:
        return a == b or b in table.hosts.get(a, ()) or a in table.hosts.get(b, ())

    real_c, pre_c, series_c = split(venue_ids(cited))
    real_k, pre_k, series_k = split(venue_ids(canonical))

    if real_k:
        if real_c:
            if not any(compatible(a, b) for a in real_c for b in real_k):
                return DIFFERENT
            if real_c & real_k and is_workshop(canonical) and not is_workshop(cited):
                return DIFFERENT
            return SAME
        if pre_c:
            return DIFFERENT       # cited as a preprint; the record gives a venue
        return UNCOMPARABLE if series_c else UNKNOWN
    if pre_k or series_k:          # the record names only a preprint server or a series
        if pre_c & pre_k or series_c & series_k:
            return SAME
        if pre_c and not real_c and series_k and not pre_k:
            return DIFFERENT       # cited as a preprint; the record is a printed volume
        return UNCOMPARABLE
    if series_c and not real_c and not pre_c:
        return UNCOMPARABLE
    return UNKNOWN


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover — a debugging aid
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m citation_verifier.venues <venue> [<other venue>]")
        return 2
    for a in args:
        ids = ", ".join(f"{i} ({venue_kind(i)})" for i in sorted(venue_ids(a))) or "— not in the table"
        print(f"{a!r}: {ids}{'  [workshop]' if is_workshop(a) else ''}")
    if len(args) == 2:
        print(f"cited vs canonical: {compare_venues(args[0], args[1])}"
              f"   abbreviation: {is_abbreviation_of(args[0], args[1])}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
