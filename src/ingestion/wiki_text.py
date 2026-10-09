"""Crude wikitext -> plaintext conversion.

Good enough for retrieval: keeps section headings and prose, drops
templates/tables/refs the same way MediaWiki's TextExtracts mostly does.
"""

import html
import re

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_RE = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(
    r"<(?:gallery|timeline|score|syntaxhighlight|nowiki|includeonly|noinclude)"
    r"[^>]*>.*?</(?:gallery|timeline|score|syntaxhighlight|nowiki|includeonly|noinclude)>",
    re.DOTALL | re.IGNORECASE,
)
_FILE_LINK_RE = re.compile(
    r"\[\[(?:File|Image|Category|Media):[^\]]*\]\]", re.IGNORECASE
)
_PLAIN_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_EXT_LINK_RE = re.compile(r"\[(?:https?://\S+?)(?:\s+([^\]]+))?\]")
_MAGIC_RE = re.compile(r"__[A-Z]+__")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_APOSTROPHE_RE = re.compile(r"'{2,5}")


def _remove_balanced(text: str, open_token: str, close_token: str) -> str:
    """Remove balanced {{...}} / {| ... |} blocks, innermost-safe."""
    out = []
    i = 0
    n = len(text)
    open_len = len(open_token)
    close_len = len(close_token)

    while i < n:
        if text.startswith(open_token, i):
            depth = 1
            j = i + open_len
            while j < n and depth:
                if text.startswith(open_token, j):
                    depth += 1
                    j += open_len
                elif text.startswith(close_token, j):
                    depth -= 1
                    j += close_len
                else:
                    j += 1
            i = j
            continue
        out.append(text[i])
        i += 1

    return "".join(out)


def wikitext_to_plaintext(text: str) -> str:
    text = _COMMENT_RE.sub(" ", text)
    text = _REF_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = _remove_balanced(text, "{{", "}}")
    text = _remove_balanced(text, "{|", "|}")
    text = _FILE_LINK_RE.sub(" ", text)
    text = _PLAIN_LINK_RE.sub(lambda m: m.group(2) or m.group(1), text)
    text = _EXT_LINK_RE.sub(lambda m: m.group(1) or " ", text)
    text = _MAGIC_RE.sub(" ", text)
    text = _APOSTROPHE_RE.sub("", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()
