from typing import Any


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    k: int = 60,
    top_n: int = 24,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    chunks: dict[str, dict[str, Any]] = {}

    # Process dense
    for rank, res in enumerate(dense_results):
        chunk_id = res["id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        chunks[chunk_id] = res

    # Process BM25
    for rank, res in enumerate(bm25_results):
        chunk_id = res["id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        if chunk_id not in chunks:
            chunks[chunk_id] = res

    # Sort by fused score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused_results = []
    seen_texts = set()

    for cid in sorted_ids:
        chunk = chunks[cid]
        text = chunk.get("document", "")

        # Simple near-duplicate removal (exact text match for now)
        if text in seen_texts:
            continue

        seen_texts.add(text)
        chunk["fused_score"] = scores[cid]
        fused_results.append(chunk)

        if len(fused_results) >= top_n:
            break

    return fused_results
