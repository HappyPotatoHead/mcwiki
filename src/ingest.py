"""
Turn wiki articles into retrieval chunks.

Splits plaintext extracts on their section headings, keeps the heading
hierarchy as a breadcrumb path, and packs section bodies into
size-bounded chunks with paragraph-level overlap.
"""

# `re` is used to recognise headings
# == Crafting ==
# === Materials ===
# ==== Iron ====
import re

from src import config

# Does this look like a heading
# With 2 to 6 "=" wrapped around the word
# This also gives us a level
# == is level 2
# === is level 3
HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")
# this is just a label
INTRO_PATH = "(intro)"


# you put the entire wiki article
def _split_sections(text: str) -> list[tuple[str, str]]:
    """
    Return (path, body) pairs; path is a 'A > B' breadcrumb.
    sections = [
        ("(intro)", "some introduction"),
        ("Crafting", "crafting information"),
        ("Crafting > Materials", "material information"),
        ("Crafting > Tools", "tool information"),
    ]
    """
    sections: list[tuple[str, str]] = []

    # keeps track of where we are in the hierarchy
    stack: list[tuple[int, str]] = []

    # temporary holds ordinary texts, after encountering headings
    buffer: list[str] = []

    dropped = False

    def flush() -> None:
        # we're clearing the text buffers for the next sections
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body or dropped:
            return

        # this is breadcrumbing
        path = " > ".join(title for _, title in stack) or INTRO_PATH

        # subheadings
        # Crafting > Materials
        sections.append((path, body))

    # goes through the article one line at a time
    for line in text.split("\n"):
        # ignores body content, not what we want in the first place
        # for path
        match = HEADING_RE.match(line.strip())
        if not match:
            # introduction line
            buffer.append(line)
            continue

        # so that we can group subheadings with their correct headings
        level = len(match.group(1))
        title = match.group(2).strip()

        # extract the body of the text
        flush()

        # this is when you reach the same heading or when
        # there's a deeper heading
        # This is the part that tells determines which subsections belong to which subsections
        while stack and stack[-1][0] >= level:
            stack.pop()

        if level == 2:
            # we don't want sections about issues
            dropped = title in config.DROP_SECTIONS

        if not dropped:
            stack.append((level, title))

    flush()

    return sections


# this part is breaking the texts into chunks
def _pack_paragraphs(body: str) -> list[str]:
    # easier to iterate rather than blank spaces
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

    # collections of built chunks
    chunks: list[str] = []

    # keep track of current chunk
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        # we want to avoid overloading it
        if current and current_len + len(paragraph) > config.CHUNK_MAX_CHARS:
            # we join the current built chunk (we can't add anymore)
            chunks.append("\n\n".join(current))

            # this is so that we can preserve the context between chunks
            overlap = current[-1] if current_len > config.CHUNK_MAX_CHARS // 2 else None

            current = [overlap] if overlap else []
            current_len = len(overlap) if overlap else 0

        # still fits within the maximum configured chunk
        current.append(paragraph)
        current_len += len(paragraph)

    # appending any remaining leftovers
    if current:
        chunks.append("\n\n".join(current))

    return chunks


# this just assembles everything
def chunk_record(record: dict) -> list[dict]:
    title = record["title"]
    chunks: list[dict] = []

    # heading, and their information
    for path, body in _split_sections(record["extract"]):
        for _index, text in enumerate(_pack_paragraphs(body)):
            # intro snippets are usually noise
            if len(text) < 40 and path == INTRO_PATH:
                continue
            breadcrumb = "" if path == INTRO_PATH else path
            # put into pages.jsonl
            chunks.append(
                {
                    "id": f"{record['pageid']}:{len(chunks)}",
                    "title": title,
                    "section": path,
                    "content": text,
                    "embed_text": f"{title} > {breadcrumb}\n\n{text}".strip(),
                    "url": record["url"],
                    "pageid": record["pageid"],
                }
            )

    return chunks


# multiple wiki pages
def chunk_records(records: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for record in records:
        chunks.extend(chunk_record(record))
    return chunks
