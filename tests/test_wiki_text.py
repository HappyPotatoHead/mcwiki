from src.wiki_text import wikitext_to_plaintext


def test_removes_templates_including_nested():
    text = "{{Infobox block\n| hardness = 2\n| nested = {{x|y}}\n}}\nThe block is hard."
    result = wikitext_to_plaintext(text)
    assert "Infobox" not in result
    assert "The block is hard." in result


def test_keeps_section_headings():
    text = "== Creation ==\nBuild a frame."
    result = wikitext_to_plaintext(text)
    assert "== Creation ==" in result
    assert "Build a frame." in result


def test_links_become_their_target_text():
    text = "Craft with [[Iron Ingot|irons]] and [[Stick]]s."
    result = wikitext_to_plaintext(text)
    assert "irons" in result
    assert "Stick" in result
    assert "[[" not in result


def test_drops_file_and_category_links():
    text = "[[File:Block.png|thumb|alt]] [[Category:Blocks]] Keep me."
    result = wikitext_to_plaintext(text)
    assert "File:" not in result
    assert "Category:" not in result
    assert "Keep me." in result


def test_drops_refs_and_tables():
    text = (
        "Intro sentence.<ref>Some book</ref>\n"
        "{| class=itemtable\n! Header\n|-\n| cell value\n|}\n"
        "Closing sentence."
    )
    result = wikitext_to_plaintext(text)
    assert "Some book" not in result
    assert "cell value" not in result
    assert "Intro sentence." in result
    assert "Closing sentence." in result


def test_strips_bold_markup_but_keeps_apostrophes():
    text = "'''Bold''' and don't worry."
    result = wikitext_to_plaintext(text)
    assert "Bold" in result
    assert "'''" not in result
    assert "don't" in result


def test_unescapes_html_entities():
    text = "5&times;5 &amp; 3&#47;8"
    result = wikitext_to_plaintext(text)
    assert "&times;" not in result
    assert "&amp;" not in result
    assert "5×5" in result


def test_collapses_excess_blank_lines():
    text = "One.\n\n\n\n\nTwo."
    result = wikitext_to_plaintext(text)
    assert "\n\n\n" not in result
    assert "One.\n\nTwo." in result
