"""Two-stage retrieval: cosine top-N from Chroma, then cross-encoder rerank."""

from dataclasses import dataclass

from src import config, embed, store


@dataclass
class Hit:
    id: str
    title: str
    section: str
    url: str
    content: str
    cosine: float
    rerank_score: float | None = None


_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _reranker = CrossEncoder(config.RERANK_MODEL, device=device)
    return _reranker


def retrieve(
    query: str,
    top_n: int = config.RETRIEVE_TOP_N,
    rerank: bool = True,
) -> list[Hit]:
    """
    Return up to top_n candidates.

    With rerank=True the list is ordered by cross-encoder score (each hit
    keeps its cosine score too); otherwise it is ordered by cosine.

    Top-N -> because it's cheap
    The rerank is expensive, so we only do it for already narrowed down lists
    """

    # bi-encoder / dense retrieval
    query_embedding = embed.embed_query(query)
    # get the top 50 similar results (cosine similarity)
    raw = store.query(query_embedding, top_n)

    hits = [
        Hit(
            id=r["id"],
            title=r["title"],
            section=r["section"],
            url=r["url"],
            content=r["embed_text"],
            cosine=r["cosine"],
        )
        for r in raw
    ]

    if rerank and hits:
        """
        (
            "How do I make villagers breed?",
            "Villager > Breeding\n\nVillagers can breed when..."
        ),
        (
            "How do I make villagers breed?",
            "Villager > Trading\n\nVillagers trade..."
        )
        """
        pairs = [(query, hit.content) for hit in hits]

        # cross_encoding
        # The pair becomes the query + document
        # reranker is the cross_encoder
        scores = _get_reranker().predict(pairs)

        # associate each score with each "hit" <- retrieved result
        for hit, score in zip(hits, scores):
            hit.rerank_score = float(score)

        # sort the reranked
        hits.sort(key=lambda h: h.rerank_score, reverse=True)

    return hits


# wrap it around
def search(
    query: str,
    k: int = config.RERANK_TOP_K,
    top_n: int = config.RETRIEVE_TOP_N,
    rerank: bool = True,
) -> list[Hit]:
    return retrieve(query, top_n=top_n, rerank=rerank)[:k]
