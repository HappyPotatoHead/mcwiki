"""Retrieval evaluation: Recall@k and MRR, cosine-only vs reranked."""

from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEVICE,
    EMBED_MODEL,
    QUERY_INSTRUCTION,
    RERANK_MODEL,
    RETRIEVE_TOP_N,
)
from .embedder import Embedder
from .retrieve import Match, Reranker
from .store import ChromaStore

TEST_CASES = [
    # question -> expected source
    ("What do I need to make a Nether portal?", "Nether Portal"),
    ("What enchantment makes tools last longer?", "Unbreaking"),
    ("How do villagers reproduce?", "Villager"),
    ("What does Mending do?", "Mending"),
    ("How does a wheat farm work?", "Wheat"),
    ("What is the End?", "The End"),
    ("What does Silk Touch do?", "Silk Touch"),
    ("What is a librarian?", "Librarian"),
    ("How do I make a furnace?", "Furnace"),
    ("What does Fortune do?", "Fortune"),
    ("What is the Nether?", "The Nether"),
    ("How do I breed villagers?", "Villager"),
    ("How do I enter the Nether?", "The Nether"),
    ("How do I defeat the Ender Dragon?", "Ender Dragon"),
    ("What does a hopper do?", "Hopper"),
    ("What is an ender chest?", "Ender Chest"),
    ("How do I enchant a weapon?", "Enchanting Table"),
    ("What tools do I need to mine diamond ore?", "Diamond"),
]


def _rank(matches: list[Match], expected: str) -> int | None:
    expected = expected.casefold()
    for rank, match in enumerate(matches, start=1):
        if match.title.casefold() == expected:
            return rank
    return None


def _metrics(ranks: list[int | None]) -> dict:
    # queries where expected page was found
    scored = [r for r in ranks if r is not None]
    total = len(ranks)
    return {
        # measures the proportion of queries where the expected page appears first
        "recall@1": (sum(1 for r in scored if r <= 1) / total) if total else 0.0,
        # measures the proportion of queries where the expected page appears in the top 5
        "recall@5": (sum(1 for r in scored if r <= 5) / total) if total else 0.0,
        # reward
        "mrr": (sum(1 / r for r in scored) / total) if total else 0.0,
        "found": len(scored),
        "total": total,
    }


def main() -> None:
    database = ChromaStore(str(CHROMA_DIR), COLLECTION_NAME)
    reranker = Reranker(RERANK_MODEL, DEVICE)
    embedder = Embedder(EMBED_MODEL, DEVICE)

    if len(database) == 0:
        print("Index is empty — run `python -m src.cli ingest` first.")
        return

    indexed = database.indexed_titles()
    cosine_ranks: list[int | None] = []
    rerank_ranks: list[int | None] = []
    skipped = 0

    print(
        f"{'Status':<4} {'Cosine rank':>5} -> {'Rerank rank':<5} {'Expected':<18} Query"
    )

    for query, expected in TEST_CASES:
        # we dont want to check non-existing sources
        # Skip unindexed title rather than calling it a failure
        if expected.casefold() not in {title.casefold() for title in indexed}:
            skipped += 1
            print(f"SKIP (not in index): {expected:<18} <- {query}")
            continue

        # shouldn't access private methods directly
        # matches have gone through cosine similarity and reranking
        matches = reranker._retrieve(
            query, RETRIEVE_TOP_N, True, embedder, database, QUERY_INSTRUCTION
        )

        # sorts via similarity score
        cosine_order = sorted(matches, key=lambda h: h.cosine, reverse=True)

        cosine_rank = _rank(cosine_order, expected)
        # already sorted with rerank
        rerank_rank = _rank(matches, expected)

        cosine_ranks.append(cosine_rank)
        rerank_ranks.append(rerank_rank)

        def fmt(rank: int | None) -> str:
            return f"#{rank}" if rank else "miss"

        status = "PASS" if rerank_rank == 1 else ("OK" if rerank_rank else "FAIL")
        print(
            f"{status:<4} {fmt(cosine_rank):>5} -> {fmt(rerank_rank):<5} "
            f"{expected:<18} <- {query}"
        )

    cosine_metrics = _metrics(cosine_ranks)
    rerank_metrics = _metrics(rerank_ranks)

    if rerank_metrics["total"] == 0:
        print("\nNo test query had its expected page in the index — nothing to score.")
        return

    print(f"\nEvaluated: {rerank_metrics['total']} queries ({skipped} skipped)")
    print(f"{'':24} {'cosine':>8} {'rerank':>8}")
    for key in ("recall@1", "recall@5", "mrr"):
        print(f"{key:24} {cosine_metrics[key]:>8.3f} {rerank_metrics[key]:>8.3f}")


if __name__ == "__main__":
    main()
