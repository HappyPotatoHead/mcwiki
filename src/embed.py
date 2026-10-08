import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src import config

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(config.EMBED_MODEL, device=device)
    return _model


def embed_documents(texts: list[str], batch_size: int = 128) -> np.ndarray:
    return get_model().encode(
        texts,
        # because the texts might just be gigantic
        batch_size=batch_size,
        # normalisation of embedding is for better performance
        normalize_embeddings=True,
        show_progress_bar=len(texts) > batch_size,
    )


def embed_query(query: str) -> np.ndarray:
    return get_model().encode(
        config.QUERY_INSTRUCTION + query,
        normalize_embeddings=True,
    )


# manual calculations
# PyTorch has their own implementation
def cosine_similarity(a, b) -> float:
    # both a and b have been normalised
    return float(np.dot(a, b))
    # return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
