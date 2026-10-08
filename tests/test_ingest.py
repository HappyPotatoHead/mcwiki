from src.ingest import _split_sections, chunk_record

FAKE_EXTRACT = """Lead intro paragraph.

== Creation ==
Body one about creation.

== Videos ==
Secret video text that must be dropped.

=== Screenshots ===
More dropped text.

== Behavior ==
Surviving paragraph.

=== Nested ===
Deep surviving text.
"""


def test_drops_noise_sections():
    sections = _split_sections(FAKE_EXTRACT)
    paths = [path for path, _ in sections]
    bodies = " ".join(body for _, body in sections)

    assert "(intro)" in paths
    assert "Behavior" in paths
    assert "Secret video" not in bodies
    assert "More dropped" not in bodies
    assert "Surviving paragraph" in bodies
    assert "Deep surviving text" in bodies


def test_nested_breadcrumb_paths():
    sections = _split_sections("== A ==\n\n=== B ===\n\nDeep.")
    assert sections == [("A > B", "Deep.")]


def test_intro_section_has_no_heading():
    sections = _split_sections("Just lead text.")
    assert sections == [("(intro)", "Just lead text.")]


def test_chunk_record_structure_and_ids():
    record = {
        "pageid": 42,
        "title": "Nether Portal",
        "url": "https://minecraft.wiki/wiki/Nether_Portal",
        "extract": FAKE_EXTRACT,
    }
    chunks = chunk_record(record)

    assert chunks
    assert all(chunk["id"].startswith("42:") for chunk in chunks)
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)
    assert all("Nether Portal" in chunk["embed_text"] for chunk in chunks)
    assert all(chunk["url"].startswith("https://minecraft.wiki/") for chunk in chunks)
    assert all(chunk["pageid"] == 42 for chunk in chunks)


def test_chunk_size_stays_near_limit():
    paragraphs = "\n\n".join(f"Paragraph number {i} " + "x" * 250 for i in range(30))
    record = {
        "pageid": 1,
        "title": "Big",
        "url": "https://minecraft.wiki/wiki/Big",
        "extract": f"== Big ==\n\n{paragraphs}",
    }
    chunks = chunk_record(record)

    assert len(chunks) > 1
    assert all(len(c["content"]) <= 1700 + 800 for c in chunks)
