import numpy as np
from sentence_transformers import SentenceTransformer

from .models import Model


class Embedder(Model):
    def embed_documents(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        return self._get_model().encode(
            texts,
            # because the texts might just be gigantic
            batch_size=batch_size,
            # normalisation of embedding is for better performance
            normalize_embeddings=True,
            show_progress_bar=len(texts) > batch_size,
        )

    def embed_query(self, query: str, query_instructions: str) -> np.ndarray:
        return self._get_model().encode(
            query_instructions + query,
            normalize_embeddings=True,
        )

    def _get_model(
        self,
    ) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model


# manual calculations
# PyTorch has their own implementation
# used for testing
def cosine_similarity(a, b) -> float:
    # both a and b have been normalised
    return float(np.dot(a, b))

    # return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
