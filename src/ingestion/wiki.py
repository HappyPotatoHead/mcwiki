"""MediaWiki client for minecraft.wiki.

Handles page enumeration (recursive category walk), plaintext extraction,
rate limiting, retries and a local JSONL cache so the wiki is only hit
for pages we have not seen yet.
"""

import json
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

from src import config
from src.ingestion.wiki_text import wikitext_to_plaintext

_last_request_ts = 0.0
_slot_lock = threading.Lock()
_cache_lock = threading.Lock()
_thread_local = threading.local()


def _log(message: str) -> None:
    print(f"[wiki] {message}", file=sys.stderr, flush=True)


def _session() -> requests.Session:
    """One keep-alive session per thread: few TCP connections instead of
    thousands (which gets the client rate-limited by the wiki's CDN)."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = config.USER_AGENT
        _thread_local.session = session
    return session


def _api_get(params: dict) -> dict:
    global _last_request_ts

    params = {**params, "format": "json", "formatversion": "2"}
    url = f"{config.WIKI_API}?{urllib.parse.urlencode(params)}"

    for attempt in range(config.REQUEST_RETRIES):
        with _slot_lock:
            wait = config.REQUEST_DELAY - (time.monotonic() - _last_request_ts)
            if wait > 0:
                time.sleep(wait)
            _last_request_ts = time.monotonic()

        try:
            response = _session().get(url, timeout=30)
        except requests.RequestException as e:
            if attempt < config.REQUEST_RETRIES - 1:
                _log(f"retry {attempt + 1} after {type(e).__name__}: {e}")
                time.sleep(2**attempt)
                continue
            raise

        if response.status_code >= 400:
            if (
                response.status_code in (429, 500, 502, 503)
                and attempt < config.REQUEST_RETRIES - 1
            ):
                _log(f"retry {attempt + 1} after HTTP {response.status_code}")
                time.sleep(2**attempt)
                continue
            response.raise_for_status()

        try:
            return response.json()
        except requests.RequestException as e:
            if attempt < config.REQUEST_RETRIES - 1:
                _log(f"retry {attempt + 1} after bad response: {e}")
                time.sleep(2**attempt)
                continue
            raise

    raise RuntimeError("unreachable")


def enumerate_category_pages(category: str) -> list[dict]:
    """Return main-namespace pages reachable from a category (recursive)."""

    pages: list[dict] = []
    seen_categories: set[str] = set()
    queue = [f"Category:{category}"]

    while queue:
        current = queue.pop()
        if current in seen_categories:
            continue
        seen_categories.add(current)

        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": current,
            "gcmtype": "page|subcat",
            "gcmlimit": "500",
            "prop": "info",
            "iiprop": "touched",
        }

        while True:
            data = _api_get(params)
            batch = data.get("query", {}).get("pages", [])
            for page in batch:
                if page.get("ns") == 14:
                    queue.append(page["title"])
                elif page.get("ns") == 0 and "missing" not in page:
                    pages.append(
                        {
                            "pageid": page["pageid"],
                            "title": page["title"],
                            "touched": page.get("touched"),
                        }
                    )

            if "continue" not in data:
                break
            params.update(data["continue"])

    return pages


def enumerate_pages(categories: list[str]) -> list[dict]:
    """Union of all pages across categories, deduplicated by pageid.

    Top-level categories are walked in parallel; each walk is serial.
    """
    by_id: dict[int, dict] = {}
    by_id_lock = threading.Lock()

    def walk(category: str) -> list[dict]:
        found = enumerate_category_pages(category)
        _log(f"Category:{category} -> {len(found)} pages")
        return found

    with ThreadPoolExecutor(max_workers=config.ENUM_WORKERS) as pool:
        for found in pool.map(walk, categories):
            with by_id_lock:
                for page in found:
                    by_id[page["pageid"]] = page

    return list(by_id.values())


def enumerate_all_pages() -> list[dict]:
    """Every main-namespace article on the wiki (~18k), with last-edit time."""
    pages: list[dict] = []
    params = {
        "action": "query",
        "generator": "allpages",
        "gapnamespace": "0",
        "gaplimit": "500",
        "prop": "info",
        "iiprop": "touched",
    }
    while True:
        data = _api_get(params)
        for page in data.get("query", {}).get("pages", []):
            if "missing" not in page:
                pages.append(
                    {
                        "pageid": page["pageid"],
                        "title": page["title"],
                        "touched": page.get("touched"),
                    }
                )
        if "continue" not in data:
            break
        params.update(data["continue"])
    return pages


def fetch_page(pageid: int, title: str) -> dict | None:
    """Fetch one article as plaintext via the extracts API (slow but clean)."""
    params = {
        "action": "query",
        "prop": "extracts|info",
        "inprop": "url",
        "explaintext": "1",
        "exlimit": "1",
        "pageids": str(pageid),
        "redirects": "1",
    }
    data = _api_get(params)
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None

    page = pages[0]
    extract = page.get("extract", "")
    if "missing" in page or not extract.strip():
        return None

    return {
        "pageid": page["pageid"],
        "title": page["title"],
        "url": page.get(
            "url",
            f"https://minecraft.wiki/wiki/{urllib.parse.quote(page['title'].replace(' ', '_'))}",
        ),
        "extract": extract,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def fetch_wikitext_batch(pages: list[dict]) -> list[dict]:
    """Fetch up to 50 articles as wikitext in one request, converted locally."""
    params = {
        "action": "query",
        "prop": "revisions|info",
        "inprop": "url",
        "rvprop": "content",
        "rvslots": "main",
        "titles": "|".join(page["title"] for page in pages),
        "redirects": "1",
    }
    data = _api_get(params)

    records = []
    for page in data.get("query", {}).get("pages", []):
        if page.get("missing") or page.get("invalid"):
            continue
        revisions = page.get("revisions") or []
        if not revisions:
            continue
        content = revisions[0].get("slots", {}).get("main", {}).get("content", "")
        if not content.strip():
            continue
        records.append(
            {
                "pageid": page["pageid"],
                "title": page["title"],
                "url": page.get(
                    "url",
                    f"https://minecraft.wiki/wiki/{urllib.parse.quote(page['title'].replace(' ', '_'))}",
                ),
                "extract": wikitext_to_plaintext(content),
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
    return records


def load_cache(cache_path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    if not cache_path.exists():
        return records
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[record["pageid"]] = record
    return records


def _append_cache(cache_path: Path, record: dict) -> None:
    with _cache_lock:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _batched(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def crawl(
    categories: list[str] | None = None,
    force: bool = False,
    limit: int | None = None,
    mode: str = config.FETCH_MODE,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> Iterator[dict]:
    """Yield article records, fetching only pages missing from the cache.

    categories=None crawls every main-namespace article.
    limit caps the number of yielded records (fetching included).
    mode "wikitext" fetches 50 pages per request (fast); "extracts" uses
    the per-page TextExtracts API (slower, slightly cleaner text).
    """
    cache = load_cache(config.WIKI_CACHE_PATH)

    if force:
        config.WIKI_CACHE_PATH.unlink(missing_ok=True)
        cache = {}

    pages = enumerate_all_pages() if categories is None else enumerate_pages(categories)
    page_ids = {p["pageid"] for p in pages}

    def is_stale(page: dict) -> bool:
        cached = cache.get(page["pageid"])
        if cached is None:
            return True
        fetched_at = cached.get("fetched_at")
        touched = page.get("touched")
        if not fetched_at:
            return True
        if not touched:
            return False
        try:
            return datetime.fromisoformat(fetched_at) < datetime.fromisoformat(touched)
        except ValueError:
            return True

    todo = pages if force else [p for p in pages if is_stale(p)]
    if not force:
        new_count = sum(1 for p in todo if p["pageid"] not in cache)
        _log(
            f"{len(todo)} of {len(pages)} pages to fetch "
            f"({new_count} new, {len(todo) - new_count} edited since last ingest)"
        )

    batch_size = config.WIKI_BATCH_SIZE if mode == "wikitext" else 1
    batches = _batched(todo, batch_size)

    def fetch_batch(batch: list[dict]) -> list[dict]:
        if mode == "wikitext":
            return fetch_wikitext_batch(batch)
        return [
            record
            for page in batch
            if (record := fetch_page(page["pageid"], page["title"])) is not None
        ]

    fetched: set[int] = set()
    done_count = 0

    def handle(record: dict) -> dict | None:
        nonlocal done_count
        if record["pageid"] in fetched:
            return None  # redirect resolved to an already-yielded page
        _append_cache(config.WIKI_CACHE_PATH, record)
        fetched.add(record["pageid"])
        done_count += 1
        if on_progress:
            on_progress(done_count, len(todo), record["title"])
        return record

    stop = False

    if limit is None:
        with ThreadPoolExecutor(max_workers=config.FETCH_WORKERS) as pool:
            futures = [pool.submit(fetch_batch, batch) for batch in batches]
            for future in as_completed(futures):
                for record in future.result():
                    if (record := handle(record)) is not None:
                        yield record
    else:
        for batch in batches:
            for record in fetch_batch(batch):
                if len(fetched) >= limit:
                    stop = True
                    break
                if (record := handle(record)) is not None:
                    yield record
            if stop:
                break

    for record in cache.values():
        if limit is not None and len(fetched) >= limit:
            return
        if record["pageid"] in page_ids and record["pageid"] not in fetched:
            yield record
