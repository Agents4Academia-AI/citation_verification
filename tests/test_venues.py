"""The venue alias table and its matcher (``citation_verifier.venues``).

Every string below is one a bibliography or a metadata source really produces: BibTeX
``booktitle`` / ``journal`` fields, DBLP and ISO-4 abbreviations, and the venue names
Semantic Scholar, Crossref and the arXiv API return. Offline, no model.
"""

from __future__ import annotations

import pytest

from citation_verifier import venues as V
from citation_verifier.stages import correctness as C

# (cited, canonical): one venue under two spellings — never a finding.
SAME_VENUE = [
    # acronym / former acronym / full name / the name with its decorations
    ("NeurIPS", "Neural Information Processing Systems"),
    ("NIPS", "Advances in Neural Information Processing Systems 30"),
    ("neurips", "Neural Information Processing Systems"),
    ("Thirty-sixth Conference on Neural Information Processing Systems (NeurIPS 2022)", "NeurIPS"),
    ("ICML", "Proceedings of the 38th International Conference on Machine Learning, PMLR 139"),
    ("CVPR", "2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)"),
    ("IEEE Computer Society Conference on Computer Vision and Pattern Recognition", "CVPR"),
    ("ICCV", "IEEE International Conference on Computer Vision"),
    ("NAACL-HLT", "North American Chapter of the Association for Computational Linguistics"),
    ("ACL", "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)"),
    ("WWW", "Proceedings of the Web Conference 2021"),
    ("VLDB", "Proceedings of the VLDB Endowment"),
    ("Nature", "Nature volume 645, pages 633-638 (2025)"),
    ("Science", "Science (New York, N.Y.)"),
    ("Lancet", "The Lancet"),
    # word-by-word abbreviations of a listed name — none of them is in the table
    ("Adv. Neural Inf. Process. Syst.", "NeurIPS"),
    ("Proc. Int. Conf. Mach. Learn.", "International Conference on Machine Learning"),
    ("Proc. IEEE Conf. Comput. Vis. Pattern Recognit.", "Computer Vision and Pattern Recognition"),
    ("IEEE Trans. Pattern Anal. Mach. Intell.", "IEEE Transactions on Pattern Analysis and Machine Intelligence"),
    ("TPAMI", "IEEE Trans. Pattern Anal. Mach. Intell."),
    ("J. Mach. Learn. Res.", "Journal of Machine Learning Research"),
    ("Annu. Meet. Assoc. Comput. Linguist.", "Annual Meeting of the Association for Computational Linguistics"),
    ("Nat. Commun.", "Nature Communications"),
    ("Proc. Natl. Acad. Sci. U.S.A.", "Proceedings of the National Academy of Sciences of the United States of America"),
    # preprint servers
    ("CoRR", "arXiv.org"),
    ("arXiv preprint arXiv:1706.03762", "arXiv (preprint)"),
    ("Arxiv", "ArXiv"),
    # a joint venue is each of its parts
    ("EMNLP", "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing "
              "and the 9th International Joint Conference on Natural Language Processing"),
    # a conference and the journal that prints its proceedings
    ("SIGGRAPH", "ACM Transactions on Graphics"),
    ("POPL", "Proceedings of the ACM on Programming Languages"),
    # names only a metadata source uses
    ("Machine Learning", "Machine-mediated learning"),
    ("WACV", "IEEE Workshop/Winter Conference on Applications of Computer Vision"),
    ("ECML PKDD", "PKDD/ECML Workshops"),
    # a workshop cited as one, against a record that drops the detail
    ("NeurIPS Workshop on Deep Reinforcement Learning", "Neural Information Processing Systems"),
]

# (cited, canonical): the record cannot contradict the citation.
NOT_COMPARABLE = [
    ("ICLR", "arXiv (preprint)"),            # a preprint record: where a copy lives
    ("NeurIPS", "arXiv.org"),
    ("ICLR", "OpenReview.net"),
    ("AI & Society", "arXiv (preprint)"),    # …whatever the cited venue is
    ("ICML", "Proceedings of Machine Learning Research"),   # a series: names the volume
    ("ECCV", "Lecture Notes in Computer Science"),
]

# (cited, canonical): two different known venues.
DIFFERENT_VENUE = [
    ("ICML", "International Conference on Learning Representations"),
    ("NeurIPS", "2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)"),
    ("ICML", "International Conference on Machine Learning and Applications"),   # ICMLA
    ("KDD", "Pacific-Asia Conference on Knowledge Discovery and Data Mining"),   # PAKDD
    ("ICDM", "SIAM International Conference on Data Mining"),                    # SDM
    ("Nature", "Nature Communications"),
    ("ACL", "Transactions of the Association for Computational Linguistics"),
    ("JMLR", "Transactions on Machine Learning Research"),
    ("ACL", "Nature"),
]


@pytest.mark.parametrize("cited,canonical", SAME_VENUE)
def test_two_spellings_of_one_venue_are_the_same_venue(cited, canonical):
    assert V.compare_venues(cited, canonical) == V.SAME, (V.venue_ids(cited), V.venue_ids(canonical))
    assert C._venue_issue(cited, canonical) is None


@pytest.mark.parametrize("cited,canonical", NOT_COMPARABLE)
def test_a_preprint_or_series_record_contradicts_no_venue(cited, canonical):
    """The largest source of false venue findings: the record came from arXiv, so its
    "venue" is arXiv, and every published paper cited correctly was reported."""
    assert V.compare_venues(cited, canonical) == V.UNCOMPARABLE
    assert C._venue_issue(cited, canonical) is None


@pytest.mark.parametrize("cited,canonical", DIFFERENT_VENUE)
def test_two_known_venues_that_differ_are_a_mismatch(cited, canonical):
    assert V.compare_venues(cited, canonical) == V.DIFFERENT, (V.venue_ids(cited), V.venue_ids(canonical))
    assert C._venue_issue(cited, canonical).startswith("venue mismatch")


def test_a_name_is_not_found_inside_a_longer_name():
    """Longest name first, on word boundaries: the inside of another venue's name is not
    that venue."""
    assert V.venue_ids("International Conference on Machine Learning and Applications") == {"ICMLA"}
    assert V.venue_ids("Pacific-Asia Conference on Knowledge Discovery and Data Mining") == {"PAKDD"}
    assert V.venue_ids("Transactions of the Association for Computational Linguistics") == {"TACL"}
    # …and an unlisted venue that contains a listed name stays unknown rather than wrong
    for unlisted in (
        "International Conference on Machine Learning and Cybernetics",
        "International Conference on Computer Vision Theory and Applications",
        "International Conference on Pattern Recognition Applications and Methods",
        "Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support",
        "Nature Reviews Neuroscience",
        "Artificial Intelligence Review",
        "Pattern Recognition Letters",
        "BMC Bioinformatics",
    ):
        assert V.venue_ids(unlisted) == frozenset(), unlisted


def test_an_acronym_is_a_venue_only_when_written_as_one():
    assert V.venue_ids("SODA") == {"SODA"}
    assert V.venue_ids("Proc. SODA 2019") == {"SODA"}
    assert V.venue_ids("Journal of Soda Science") == frozenset()
    assert V.venue_ids("New Media & Society") == frozenset()
    assert V.venue_ids("NEURIPS") == V.venue_ids("NeurIPS") == V.venue_ids("neurips") == {"NeurIPS"}
    assert V.venue_ids("arxiv preprint") == {"arXiv"}
    # the workshop form of an acronym
    assert V.venue_ids("CVPRW") == {"CVPR"} and V.is_workshop("CVPRW")
    assert V.venue_ids("WWW") == {"WWW"} and not V.is_workshop("WWW")


def test_a_joint_venue_is_recognised_as_both():
    joint = ("Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing "
             "and the 9th International Joint Conference on Natural Language Processing")
    assert V.venue_ids(joint) == {"EMNLP", "IJCNLP"}
    assert V.venue_ids("EMNLP-IJCNLP 2019") == {"EMNLP", "IJCNLP"}
    assert V.venue_ids("ICML, PMLR 139") == {"ICML", "PMLR"}


def test_a_workshop_cited_as_the_conference_is_reported_as_that():
    for canonical in (
        "IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)",
        "CVPR Workshops",
    ):
        issue = C._venue_issue("CVPR", canonical)
        assert issue and "a workshop at that conference" in issue, canonical
    # the other way round is not a finding: sources often drop the workshop
    assert C._venue_issue("CVPR Workshops", "Computer Vision and Pattern Recognition") is None
    # and a source's odd name for the whole conference is not a workshop
    assert not V.is_workshop("PKDD/ECML Workshops")


def test_cited_as_a_preprint_when_the_record_gives_a_venue():
    for canonical in (
        "Computer Vision and Pattern Recognition",       # a venue the table knows
        "Proceedings of Machine Learning Research",      # a printed volume
        "Alexa Prize Proceedings",                       # a venue the table has never heard of
    ):
        issue = C._venue_issue("arXiv preprint arXiv:1512.03385", canonical)
        assert issue and issue.startswith("cited as a preprint"), canonical
    assert C._venue_issue("arXiv preprint arXiv:1512.03385", "arXiv (preprint)") is None
    assert C._venue_issue("arXiv", "bioRxiv") is None    # two hosts of one paper


def test_venues_the_table_does_not_list_fall_back_to_the_general_rules():
    same = [
        ("Comput. Educ. Artif. Intell.", "Computers and Education: Artificial Intelligence"),
        ("Lang. Linguistics Compass", "Language and Linguistics Compass"),
        ("Int. J. Obscure Stud.", "International Journal of Obscure Studies"),
        ("Frontiers of Computer Science", "Frontiers in Computer Science"),
        ("FSE", "Fast Software Encryption"),             # an acronym the table gives to another venue
        ("SDM", "SIAM International Conference on Data Mining"),
    ]
    for cited, canonical in same:
        assert C._venue_issue(cited, canonical) is None, (cited, canonical)
    for cited, canonical in [
        ("Journal of Obscure Results", "International Conference on Machine Learning"),
        ("Addison-Wesley Press", "Science"),
        ("New Media & Society", "Medical Image Analysis"),
    ]:
        assert C._venue_issue(cited, canonical).startswith("venue mismatch"), (cited, canonical)


def test_abbreviation_alignment_is_word_by_word():
    assert V.is_abbreviation_of("IEEE Trans. Pattern Anal. Mach. Intell.",
                                "IEEE Transactions on Pattern Analysis and Machine Intelligence")
    assert V.is_abbreviation_of("Int. J. Comput. Vis.", "International Journal of Computer Vision")
    assert not V.is_abbreviation_of("Int. J. Comput. Vis.", "International Journal of Robotics Research")
    assert not V.is_abbreviation_of("J. A. B.", "Journal of Applied Biology")   # initials identify nothing
    assert not V.is_abbreviation_of("Nature", "Nature")                          # not an abbreviation


def test_the_table_is_internally_consistent():
    table = V._table()
    assert len(table.kind) > 100
    assert set(table.kind.values()) <= {"conference", "journal", "preprint", "series"}
    # every host target is an entry
    assert all(h in table.kind for hs in table.hosts.values() for h in hs)
    # no spelling belongs to two venues: each listed name and acronym finds its own entry
    from importlib import resources

    import yaml

    raw = resources.files("citation_verifier").joinpath("data/venues.yaml").read_text("utf-8")
    for vid, entry in yaml.safe_load(raw)["venues"].items():
        for spelling in (entry.get("acronyms") or []) + (entry.get("names") or []):
            assert vid in V.venue_ids(str(spelling)), (vid, spelling, V.venue_ids(str(spelling)))


def test_empty_and_unknown_input():
    assert V.venue_ids("") == frozenset() and V.venue_ids(None) == frozenset()
    assert V.compare_venues("", "ICML") == V.UNKNOWN
    assert C._venue_issue("", "ICML") is None and C._venue_issue("ICML", "") is None
