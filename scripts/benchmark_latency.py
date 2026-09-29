import time

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.models import warm_up_models


def benchmark_latency():
    print("Benchmarking Retrieval Latency...")

    # Warm up models
    warm_up_models()

    retriever = HybridRetriever()

    # Dummy queries
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "What are decision trees?",
        "Deep learning basics",
        "Support vector machines",
    ] * 10

    total_time = 0
    stage_times = {
        "embed": [],
        "dense": [],
        "bm25": [],
        "fuse": [],
        "rerank": [],
        "expand": [],
    }

    for q in queries:
        start = time.time()
        res = retriever.retrieve(q)
        duration = time.time() - start
        total_time += duration

        for k, v in res["timings"].items():
            if k in stage_times:
                stage_times[k].append(v)

    import numpy as np

    print("\nLatency Benchmark Results (50 queries):")
    print(f"Average Total Retrieval Time: {total_time / 50 * 1000:.2f} ms")

    print("\nStage Breakdown (ms):")
    for k, v in stage_times.items():
        if v:
            print(
                f"  {k.capitalize()}: p50={np.percentile(v, 50):.2f}, p95={np.percentile(v, 95):.2f}"
            )


if __name__ == "__main__":
    benchmark_latency()
