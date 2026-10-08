# ingest.py

## split sections

```py
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
    buffer: list[str] = []
    dropped = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body or dropped:
            return
        path = " > ".join(title for _, title in stack) or INTRO_PATH
        sections.append((path, body))

    for line in text.split("\n"):
        match = HEADING_RE.match(line.strip())
        if not match:
            buffer.append(line)
            continue

        level = len(match.group(1))
        title = match.group(2).strip()

        flush()
        while stack and stack[-1][0] >= level:
            stack.pop()

        if level == 2:
            dropped = title in config.DROP_SECTIONS

        if not dropped:
            stack.append((level, title))

    flush()

    return sections
```

### `stack`

Keeps track of where we are in the heading hierarchy.

Assuming an article contains:

```text
== Crafting ==

blah blah

=== Materials ===

iron

=== Tools ===

pickaxe

== Enchantments ==

efficiency
```

As we walk through the wiki, the stack might look like:

```
[] -> [(2, "Crafting")] -> [(2, "Crafting"), (3, "Materials")]
```

But then we encounter Tools, which is also a level 3. So, we replace materials with tools

```
[(2, "Crafting"), (3, "Materials")] -> [(2, "Crafting"), (3, "Tools")]
```

And the same thing happens when we encounter Enchantments, level 2. We remove Tools because that's a level lower than 2. Essentially, removing any subheadings along with the headings

```
[(2, "Crafting"), (3, "Tools")] -> [(2, "Enchantments")]
```

## pack_paragraphs

```py
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

        # current chunk
        current.append(paragraph)
        current_len += len(paragraph)

    # appending any remaining leftovers
    if current:
        chunks.append("\n\n".join(current))

    return chunks
```

```py
            # this is so that we can preserve the context between chunks
            overlap = current[-1] if current_len > config.CHUNK_MAX_CHARS // 2 else None

            current = [overlap] if overlap else []
            current_len = len(overlap) if overlap else 0
```

the overlapping part is where we want to preserve the context between chunks. we take the last of the current chunk (previous chunk) -> overlap.

Under the condition where the current chunk exceeds the half of max per chunk -> why half? because that's the threshold where the information that crosses chunk boundaries become significant. Too low and there information is pointless, too high and there's barely any information.

We then add the overlap to the next current and reset our current_len to the length of the overlap

So imagine:

```
Chunk 1:

paragraph A
paragraph B
paragraph C
```

and we need to start another chunk. Instead of:

```
Chunk 2:
paragraph D
paragraph E
```

we might get:

```
Chunk 2:
paragraph C
paragraph D
paragraph E
```

## pack_paragraphs

```py

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
```

```
{
    "id": "12345:7",
    "title": "Villager",
    "section": "Breeding > Mechanics",
    "content": "Villagers can breed when...",
    "embed_text": "Villager > Breeding > Mechanics\n\nVillagers can breed when...",
    "url": "https://minecraft.wiki/...",
    "pageid": "12345"
}
```

Why separate content and embed_text?
content -> for the LLM
embed_text -> for the embedding model
