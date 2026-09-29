from src.config import config
from src.documents.vectorstore import VectorStore
from src.retrieval.bm25_index import BM25Index
from src.retrieval.cache import clear_caches
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.hybrid_retriever import HybridRetriever


def test_reciprocal_rank_fusion():
    dense = [{"id": "A", "document": "A doc"}, {"id": "B", "document": "B doc"}]
    bm25 = [{"id": "B", "document": "B doc"}, {"id": "C", "document": "C doc"}]

    # B is in both (rank 1 dense, rank 0 bm25)
    fused = reciprocal_rank_fusion(dense, bm25, k=1, top_n=3)

    assert len(fused) == 3
    # B should have highest score because it is in both and high rank
    assert fused[0]["id"] == "B"


def test_bm25_index_filtering():
    index = BM25Index()
    chunks = [
        {"id": "c1", "document": "apple banana", "metadata": {"doc_id": "d1"}},
        {"id": "c2", "document": "apple orange", "metadata": {"doc_id": "d2"}},
        {"id": "c3", "document": "pear grape", "metadata": {"doc_id": "d3"}},
    ]
    index.build(chunks)

    # No filter
    res = index.search("banana", n=5)
    assert len(res) == 1

    # Filter by d2
    res_filtered = index.search("orange", n=5, doc_ids=["d2"])
    assert len(res_filtered) == 1
    assert res_filtered[0]["id"] == "c2"


from src.storage import db


def test_hybrid_retriever_offline():

    config.DB_PATH = "data/app_test.db"
    db.init_db()

    config.LLM_PROVIDER = "fake"
    config.ENABLE_BM25 = True
    config.ENABLE_RERANKER = True
    config.ENABLE_NEIGHBOR_EXPANSION = True

    clear_caches()

    vs = VectorStore(use_ephemeral=True)
    vs.upsert_chunks(
        ids=["d1:1:0", "d1:1:1", "d1:1:2"],
        embeddings=[[0.1] * 384, [0.1] * 384, [0.1] * 384],
        documents=["text 0", "text 1", "text 2"],
        metadatas=[
            {"doc_id": "d1", "file_name": "f.pdf", "page": 1, "chunk_index": 0},
            {"doc_id": "d1", "file_name": "f.pdf", "page": 1, "chunk_index": 1},
            {"doc_id": "d1", "file_name": "f.pdf", "page": 1, "chunk_index": 2},
        ],
    )

    bm25 = BM25Index()
    bm25.build(vs.get_all_chunks())

    retriever = HybridRetriever(vs, bm25)

    # Retrieve
    res = retriever.retrieve("text 1", top_k=1)

    chunks = res["chunks"]
    assert not res["cache_hit"]
    assert len(res["timings"]) > 0

    # Because of neighbor expansion on chunk 1 (if it's top), we should get 0, 1, 2
    assert len(chunks) > 1
    has_neighbor = any(c.is_neighbor_expansion for c in chunks)
    assert has_neighbor

    # Cache hit check
    res2 = retriever.retrieve("text 1", top_k=1)
    assert res2["cache_hit"]
