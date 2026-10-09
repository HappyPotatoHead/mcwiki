from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from .embedder import Embedder
from .models import Model
from .store import ChromaStore


class Match(BaseModel):
    id: str
    title: str
    section: str
    url: str
    content: str
    cosine: float
    rerank_score: float | None = 0.0


class Reranker(Model):
    def search(
        self,
        query: str,
        k: int,
        top_n: int,
        rerank: bool,
        embedder: Embedder,
        database: ChromaStore,
        query_instructions: str,
    ) -> list[Match]:
        return self._retrieve(
            query, top_n, rerank, embedder, database, query_instructions
        )[:k]

    def _get_reranker(
        self,
    ) -> CrossEncoder:
        if self.model is None:
            self.model = CrossEncoder(self.model_name, device=self.device)

        return self.model

    def _rerank(self, query: str, matches: list[Match]) -> list[Match]:
        reranker = self._get_reranker()
        pairs = [(query, match.content) for match in matches]
        scores = reranker.predict(pairs)
        for match, score in zip(matches, scores):
            match.rerank_score = float(score)

        matches.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)

        return matches

    def _retrieve(
        self,
        query: str,
        # this is the cosine similarity -> im reusing it for the subsequent reranking
        top_n: int,
        rerank: bool,
        embedder: Embedder,
        database: ChromaStore,
        query_instructions: str,
    ) -> list[Match]:
        # retrieve matches based on query
        # cast a wide net with cosine_similarity top n
        # rerank boolean for cross encoding reranking

        query_embedding = embedder.embed_query(query, query_instructions)
        top_results = database.query(query_embedding, top_n)

        matches = [
            Match(
                id=str(result.get("id") or ""),
                title=str(result.get("title") or ""),
                section=str(result.get("section") or ""),
                url=str(result.get("url") or ""),
                content=str(result.get("embed_text") or ""),
                cosine=float(result.get("cosine") or 0.0),
            )
            for result in top_results
        ]

        # rerank_score: float | None = None
        # cross enccoding reranking
        if rerank and matches:
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
            matches = self._rerank(query, matches)

        return matches
