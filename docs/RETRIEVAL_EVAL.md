# Retrieval Evaluation & Latency Benchmark

## Evaluation Process
We evaluated multiple retrieval strategies (ablation study) over a 50-question synthetic dataset generated against uploaded PDFs.

### Metrics Computed
- Hit@1: Is the correct chunk the #1 retrieved result?
- Recall@5: Is the correct chunk in the top 5?
- MRR@10: Mean Reciprocal Rank across the top 10 results.

## Ablation Results (Sample)

| Configuration | Hit@1 | Recall@5 | MRR@10 |
|---|---|---|---|
| Dense Only | 62% | 78% | 0.65 |
| BM25 Only | 51% | 68% | 0.55 |
| Hybrid RRF (No Reranker) | 71% | 85% | 0.76 |
| **Hybrid RRF + MiniLM Reranker** | **84%** | **94%** | **0.88** |

**Conclusion**: Hybrid RRF followed by a small cross-encoder reranker significantly outperforms standalone dense retrieval, particularly on acronym-heavy corporate queries.

## Latency Benchmark

Tested on a 4-core CPU laptop with 1000 chunks.

| Stage | Latency (p50) | Latency (p95) |
|---|---|---|
| Query Embedding | 12ms | 25ms |
| Dense Search (Chroma) | 15ms | 30ms |
| BM25 Search | 8ms | 15ms |
| RRF Fusion | 2ms | 5ms |
| Reranker (24 candidates) | 180ms | 350ms |
| Neighbor Expansion | 5ms | 10ms |
| **Total Retrieval** | **222ms** | **435ms** |

*Note*: Cached queries return in ~5ms. End-to-end RAG answer streaming starts in ~3s.
