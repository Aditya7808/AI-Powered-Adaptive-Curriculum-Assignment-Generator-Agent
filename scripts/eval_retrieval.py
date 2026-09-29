import json
import os

from src.retrieval.hybrid_retriever import HybridRetriever


def evaluate_retrieval():
    eval_file = os.path.join("data", "eval", "eval_set.json")
    if not os.path.exists(eval_file):
        print(f"Eval set not found at {eval_file}. Run generate_eval_set.py first.")
        return

    with open(eval_file, "r") as f:
        eval_data = json.load(f)

    retriever = HybridRetriever()

    metrics = {"hit@1": 0, "recall@5": 0, "mrr@10": 0}

    print(f"Evaluating {len(eval_data)} queries...")

    for item in eval_data:
        query = item["question"]
        expected_chunks = set(item["expected_chunk_ids"])

        results = retriever.retrieve(query)
        chunks = results["chunks"]
        retrieved_ids = [c.chunk_id for c in chunks]

        # Hit@1
        if retrieved_ids and retrieved_ids[0] in expected_chunks:
            metrics["hit@1"] += 1

        # Recall@5
        hits_in_5 = len(set(retrieved_ids[:5]).intersection(expected_chunks))
        metrics["recall@5"] += (
            hits_in_5 / len(expected_chunks) if expected_chunks else 0
        )

        # MRR@10
        mrr = 0
        for i, cid in enumerate(retrieved_ids[:10]):
            if cid in expected_chunks:
                mrr = 1.0 / (i + 1)
                break
        metrics["mrr@10"] += mrr

    n = len(eval_data)
    print("\nRetrieval Evaluation Results:")
    print(f"Hit@1:  {metrics['hit@1'] / n:.2%}")
    print(f"Recall@5: {metrics['recall@5'] / n:.2%}")
    print(f"MRR@10: {metrics['mrr@10'] / n:.2%}")


if __name__ == "__main__":
    evaluate_retrieval()
