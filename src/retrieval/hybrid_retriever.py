import concurrent.futures
from typing import Any

from src.config import config
from src.documents.vectorstore import VectorStore
from src.retrieval.bm25_index import BM25Index
from src.retrieval.cache import retrieval_cache
from src.retrieval.expansion import expand_neighbors
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.models import get_embedder, get_reranker
from src.retrieval.timing import StageTimer
from src.schemas.rag import RetrievedChunk
from src.storage.db import get_kb_version


class HybridRetriever:
    def __init__(self, vectorstore: VectorStore, bm25_index: BM25Index):
        self.vectorstore = vectorstore
        self.bm25 = bm25_index

    def retrieve(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        top_k = top_k or config.FINAL_TOP_K
        timings = {}

        kb_version = get_kb_version()
        doc_ids_tup = tuple(sorted(doc_ids)) if doc_ids else None

        cache_key = (
            query.lower().strip(),
            doc_ids_tup,
            kb_version,
            config.ENABLE_BM25,
            config.ENABLE_RERANKER,
            config.ENABLE_NEIGHBOR_EXPANSION,
        )

        if cache_key in retrieval_cache:
            res = retrieval_cache[cache_key]
            return {"chunks": res, "timings": {}, "cache_hit": True}

        # 1. Embed query
        with StageTimer(timings, "embed_query"):
            embedder = get_embedder()
            query_embedding = embedder.embed_query(query)

        # 2. Parallel search
        dense_results = []
        bm25_results = []

        with StageTimer(timings, "parallel_search"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                f_dense = executor.submit(
                    self.vectorstore.dense_search,
                    query_embedding,
                    config.DENSE_TOP_N,
                    doc_ids,
                )
                if config.ENABLE_BM25:
                    f_bm25 = executor.submit(
                        self.bm25.search, query, config.BM25_TOP_N, doc_ids
                    )

                dense_results = f_dense.result()
                if config.ENABLE_BM25:
                    bm25_results = f_bm25.result()

        # 3. Fusion
        with StageTimer(timings, "fusion"):
            fused_candidates = reciprocal_rank_fusion(
                dense_results,
                bm25_results,
                k=config.FUSION_K,
                top_n=config.RERANK_CANDIDATES,
            )

        # 4. Rerank
        with StageTimer(timings, "rerank"):
            if not config.ENABLE_RERANKER:
                reranked = fused_candidates[:top_k]
                for c in reranked:
                    c["rerank_score"] = c.get("fused_score", 0.0)
            else:
                reranker = get_reranker()
                texts = [c["document"] for c in fused_candidates]
                scores = reranker.rerank(query, texts)

                for c, s in zip(fused_candidates, scores):
                    c["rerank_score"] = s

                reranked = sorted(
                    fused_candidates, key=lambda x: x["rerank_score"], reverse=True
                )[:top_k]

        # 5. Score Bands
        for c in reranked:
            s = c["rerank_score"]
            if s >= config.RERANK_RELEVANT_THRESHOLD:
                c["relevance_band"] = "relevant"
            elif s < config.RERANK_DROP_THRESHOLD:
                c["relevance_band"] = "irrelevant"
            else:
                c["relevance_band"] = "borderline"

        # Drop irrelevant
        reranked = [c for c in reranked if c["relevance_band"] != "irrelevant"]

        # 6. Neighbor expansion
        with StageTimer(timings, "expand"):
            if config.ENABLE_NEIGHBOR_EXPANSION:
                final_docs = expand_neighbors(
                    reranked[: config.NEIGHBOR_EXPAND_TOP_N], self.vectorstore
                )
            else:
                final_docs = reranked

        # Map to RetrievedChunk schema
        chunks_out = []
        for c in final_docs:
            chunks_out.append(
                RetrievedChunk(
                    chunk_id=c["id"],
                    doc_id=c["metadata"]["doc_id"],
                    file_name=c["metadata"]["file_name"],
                    page=c["metadata"]["page"],
                    chunk_index=c["metadata"]["chunk_index"],
                    text=c["document"],
                    dense_score=c.get("score"),
                    bm25_score=c.get("bm25_score"),
                    fused_score=c.get("fused_score"),
                    rerank_score=c.get("rerank_score"),
                    relevance_band=c.get("relevance_band", "unset"),
                    is_neighbor_expansion=c.get("is_neighbor_expansion", False),
                )
            )

        retrieval_cache[cache_key] = chunks_out

        return {"chunks": chunks_out, "timings": timings, "cache_hit": False}
