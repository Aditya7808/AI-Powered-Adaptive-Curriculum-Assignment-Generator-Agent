from src.config import config
from src.retrieval.embeddings import EmbeddingService, FakeEmbeddings
from src.retrieval.reranker import FakeReranker, RerankerService

_embedder = None
_reranker = None


def get_embedder():
    global _embedder
    if _embedder is None:
        if config.LLM_PROVIDER.lower() == "fake":
            _embedder = FakeEmbeddings()
        else:
            _embedder = EmbeddingService()
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        if config.LLM_PROVIDER.lower() == "fake":
            _reranker = FakeReranker()
        else:
            _reranker = RerankerService()
    return _reranker


def warm_up_models():
    e = get_embedder()
    r = get_reranker()
    e.embed_query("warmup")
    r.rerank("warmup", ["warmup passage"])
