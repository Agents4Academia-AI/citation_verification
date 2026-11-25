from unittest.mock import patch

from citation_verifier.grounding import fulltext

WRAPPER = ('% padding metadata\n' * 50 + r'\documentclass{article}\begin{document}'
           r'\includepdf[pages=1-last]{paper.pdf}\end{document}')


def test_wrapper_falls_through_to_pdf():
    with patch('citation_verifier.grounding.oa_fulltext.arxiv_html_text_with_url', return_value=('', '')), \
         patch.object(fulltext, '_http_get_bytes', return_value=b'archive'), \
         patch.object(fulltext, '_extract_tex_from_eprint', return_value=WRAPPER), \
         patch.object(fulltext, '_fetch_arxiv_pdf_text', return_value='Real paper body') as pdf:
        result = fulltext._fetch_arxiv_full_text('1602.03572', timeout=1, max_chars=200000)
    assert result.source == 'arxiv_pdf'
    pdf.assert_called_once()


def test_wrapper_never_returned_as_last_resort():
    with patch('citation_verifier.grounding.oa_fulltext.arxiv_html_text_with_url', return_value=('', '')), \
         patch.object(fulltext, '_http_get_bytes', return_value=b'archive'), \
         patch.object(fulltext, '_extract_tex_from_eprint', return_value=WRAPPER), \
         patch.object(fulltext, '_fetch_arxiv_pdf_text', return_value=''):
        assert not fulltext._fetch_arxiv_full_text('1602.03572', timeout=1, max_chars=200000).text


def test_cached_wrapper_is_refetched():
    got = fulltext.FullTextResult('Real PDF body', 'arxiv_pdf', 'https://arxiv.org/pdf/1602.03572')
    with patch.object(fulltext.fulltext_cache, 'read', return_value=(WRAPPER, 'arxiv_eprint', 'old')), \
         patch.object(fulltext, '_fetch_arxiv_full_text', return_value=got) as fetch, \
         patch.object(fulltext.fulltext_cache, 'write') as write:
        assert fulltext.fetch_full_text_with_source('1602.03572') == got
    fetch.assert_called_once()
    write.assert_called_once()


def test_real_body_can_include_pdf_appendix():
    text = r'\begin{document}' + ('This is substantive paper prose and reasoning. ' * 30)
    text += r'\includepdf{appendix.pdf}\end{document}'
    assert fulltext._usable_eprint(text)
