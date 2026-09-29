
from sentence_transformers import CrossEncoder

from src.config import config


class RerankerService:
    def __init__(self, model_name: str = config.RERANKER_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = CrossEncoder(
                self.model_name, max_length=config.RERANK_MAX_LENGTH
            )
        return self._model

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        # Return sigmoid scores
        scores = self.model.predict(
            [(query, t) for t in texts], batch_size=config.RERANK_BATCH_SIZE
        )

        # CrossEncoder typically outputs logits if num_labels=1
        import numpy as np

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        if len(scores.shape) > 0 and scores.ndim == 1:
            return [float(sigmoid(s)) for s in scores]
        return [float(s) for s in scores]


class FakeReranker:
    def rerank(self, query: str, texts: list[str]) -> list[float]:
        # Just return decreasing scores for deterministic testing
        return [max(0.1, 0.9 - i * 0.1) for i in range(len(texts))]
