
from cachetools import LRUCache
from sentence_transformers import SentenceTransformer

from src.config import config


class EmbeddingService:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None
        self._cache = LRUCache(maxsize=512)
        # We'll lazy load the model

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        # BGE specific prefix
        prefix = "Represent this sentence for searching relevant passages: "
        prefixed_text = prefix + text

        if prefixed_text in self._cache:
            return self._cache[prefixed_text]

        emb = self.model.encode(
            [prefixed_text], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()
        self._cache[prefixed_text] = emb
        return emb


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 384
