from .embedder import Embedder
from .generate import Generator
from .ingestion import ingest as ingest_mod
from .ingestion import wiki
from .retrieve import Match, Reranker
from .store import ChromaStore


class App:
    def __init__(
        self,
        generator: Generator,
        embedder: Embedder,
        reranker: Reranker,
        database: ChromaStore,
        query_instructions: str,
    ) -> None:
        self.generator = generator
        self.embedder = embedder
        self.reranker = reranker
        self.database = database
        self.query_instructions = query_instructions

    def search(self, query: str, k: int, top_n: int, rerank: bool) -> list[Match]:
        search = self.reranker.search(
            query,
            k,
            top_n,
            rerank,
            self.embedder,
            self.database,
            self.query_instructions,
        )
        return search

    def generate_answers(self, query: str, context: str):
        return self.generator.generate_answer(query, context)

    def ingest(
        self,
        cache_path: str,
        cache_name: str,
        categories: list[str] | None,
        force: bool = False,
        limit: int | None = None,
        mode: str = "wikitext",
        on_progress=None,
    ) -> dict:
        records = list(
            wiki.crawl(
                categories,
                force=force,
                limit=limit,
                mode=mode,
                on_progress=on_progress,
            )
        )

        unique = {}
        for record in records:
            unique.setdefault(record["pageid"], record)
        records = list(unique.values())

        if not records:
            return {"articles": 0, "chunks": 0}

        chunks = ingest_mod.chunk_records(records)
        embeddings = self.embedder.embed_documents(
            [chunk["embed_text"] for chunk in chunks]
        )

        self.database.reset()
        self.database.upsert(chunks, embeddings)

        return {
            "articles": len(records),
            "chunks": len(chunks),
        }
