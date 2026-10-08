"""ChromaDB persistence for wiki chunks (cosine distance space)."""
# uses chroma db
# The previous version uses .md files -> which is not scalable

import numpy as np

from src import config

# we dont want to keep opening connections
# like in FastAPI
_collection_cache = None
_collection_path = None


# colletion because naming convention
def _collection():
    import chromadb

    global _collection_cache, _collection_path
    path = str(config.CHROMA_DIR)
    if _collection_cache is None or _collection_path != path:
        # initialise the vector database
        # typical database things
        client = chromadb.PersistentClient(path=path)
        _collection_cache = client.get_or_create_collection(
            # by default
            # ID | embedding | document | metadata
            name=config.COLLECTION_NAME,
            # vector index should use cosine distance
            metadata={"hnsw:space": "cosine"},
        )
        _collection_path = path
    return _collection_cache


def reset() -> None:
    import chromadb

    global _collection_cache, _collection_path
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except ValueError:
        raise ValueError("Database does not exist")
    finally:
        _collection_cache = None
        _collection_path = None


def delete_pages(pageids: list[int]) -> None:
    if pageids:
        _collection().delete(where={"pageid": {"$in": pageids}})


# this is inserting information into the chromadb
def upsert(chunks: list[dict], embeddings: np.ndarray, batch_size: int = 500) -> None:
    # start connection
    collection = _collection()

    # we got the chunks from ingest.py
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        # basically update the database based on primary key
        # ID | embedding | document | metadata
        # Insert if doesnt exist
        # upsert is useful since we're updating by batch size
        # we want incremental changes, not replacing existing content
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["embed_text"] for c in batch],
            embeddings=np.asarray(
                embeddings[start : start + batch_size], dtype=np.float32
            ),
            metadatas=[
                {
                    "title": c["title"],
                    "section": c["section"],
                    "url": c["url"],
                    "pageid": c["pageid"],
                    "chars": len(c["content"]),
                }
                for c in batch
            ],
        )


def query(embedding: np.ndarray, top_n: int) -> list[dict]:
    """Return top_n neighbours sorted by cosine similarity (best first)."""
    # returns an object
    result = _collection().query(
        query_embeddings=[np.asarray(embedding, dtype=np.float32)],
        n_results=top_n,
        include=["documents", "metadatas", "distances"],
    )

    """
    documents:
        [chunk A, chunk B, chunk C]

    metadatas:
        [metadata A, metadata B, metadata C]

    distances:
        [0.18, 0.31, 0.42]
    """

    hits = []

    """
    doc = doc_A
    meta = meta_A
    distance = 0.1

    doc = doc_B
    meta = meta_B
    distance = 0.3
    """
    for doc, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "id": result["ids"][0][len(hits)],
                "embed_text": doc,
                "title": meta["title"],
                "section": meta["section"],
                "url": meta["url"],
                "cosine": 1.0 - float(distance),
            }
        )
    return hits


def count() -> int:
    return _collection().count()


def indexed_titles() -> set[str]:
    metadatas = _collection().get(include=["metadatas"])["metadatas"]
    return {m["title"] for m in metadatas if m}
