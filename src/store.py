# uses chroma db
# The previous version uses .md files -> which is not scalable

from typing import Any, cast

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Where

# from src import config


class ChromaStore:
    def __init__(self, path: str, name: str) -> None:
        self._path = path
        self._name = name
        self._client = None
        # we dont want to keep opening connections
        self._collection: Collection | None = None

        # self._collection_path: str | None = None

    def _get_client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self._path)
        return self._client

    def _get_collection(self):
        if self._collection is None:
            self._collection = self._get_client().get_or_create_collection(
                name=self._name, metadata={"hnsw:space": "cosine"}
            )

        return self._collection

    def reset(self):
        client = self._get_client()

        try:
            client.delete_collection(self._name)
        except ValueError:
            pass
        finally:
            self._collection = None
            self._get_collection()

    def upsert(
        self, chunks: list[dict], embeddings: np.ndarray, batch_size: int = 500
    ) -> None:
        collection = self._get_collection()

        for start in range(0, len(chunks), batch_size):
            # ID | embedding | document | metadata
            batch = chunks[start : start + batch_size]

            collection.upsert(
                ids=[_["id"] for _ in batch],
                embeddings=np.asarray(
                    embeddings[start : start + batch_size], dtype=np.float32
                ),
                documents=[_["embed_text"] for _ in batch],
                metadatas=[
                    {
                        "title": _["title"],
                        "section": _["section"],
                        "url": _["url"],
                        "pageid": _["pageid"],
                        "chars": len(_["content"]),
                    }
                    for _ in batch
                ],
            )

    def query(self, embedding: np.ndarray, top_n: int) -> list[dict[str, Any]]:
        collection = self._get_collection()

        query_embedding = embedding.astype(np.float32).tolist()

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n,
            # document is the text
            include=["documents", "metadatas", "distances"],
        )

        """
                    documents:
                        [chunk A, chunk B, chunk C]

                    metadatas: <- see above
                        [metadata A, metadata B, metadata C]

                    distances:
                        [0.18, 0.31, 0.42]
                    """

        matches = []

        ids_batch = result.get("ids")
        docs_batch = result.get("documents")
        meta_batch = result.get("metadatas")
        dist_batch = result.get("distances")

        if not (ids_batch and docs_batch and meta_batch and dist_batch):
            return matches

        ids = ids_batch[0]
        documents = docs_batch[0]
        metadatas = meta_batch[0]
        distances = dist_batch[0]

        for id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            matches.append(
                {
                    "id": id,
                    "embed_text": document,
                    "title": metadata.get("title"),
                    "section": metadata.get("section"),
                    "url": metadata.get("url"),
                    "cosine": 1.0 - float(distance),
                }
            )
        return matches

    # def open(self, path: str, name: str) -> None:
    #     self._start_collection(path, name)

    def delete_pages(self, page_ids: list[int]) -> None:
        if not page_ids:
            return

        collection = self._get_collection()
        where_filter = cast(Where, {"pageid": {"$in": page_ids}})
        collection.delete(where=where_filter)

    def __len__(
        self,
    ):
        return self._get_collection().count()

    # used to check if the titles are already in the chromadb
    # fetch titles beforehand
    def indexed_titles(self) -> set[str]:
        collection = self._get_collection()

        result = collection.get(include=["metadatas"])

        meta_list = result.get("metadatas") or []

        titles: set[str] = set()
        for metadata in meta_list:
            if isinstance(metadata, dict):
                title = metadata.get("title")
                if isinstance(title, str):
                    titles.add(title)

        return titles
