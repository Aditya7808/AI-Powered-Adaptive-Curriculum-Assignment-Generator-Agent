# Architectural Decisions

## 1. Single Vector DB Collection
**Decision**: Use a single ChromaDB collection (`workspace_kb`) instead of per-user/per-pdf collections.
**Reasoning**: Greatly simplifies the retrieval logic and makes cross-document retrieval faster. Deduplication is handled using `doc_id` filtering metadata.

## 2. Pydantic-Enforced Multi-Agent Parsing
**Decision**: Use Pydantic schemas for all LLM outputs and force retries if validation fails.
**Reasoning**: Ensures agents can reliably consume the output of prior agents (e.g. Profiler output -> Curriculum input).

## 3. Hybrid RRF Retrieval
**Decision**: Implement BM25 + Dense retrieval via Reciprocal Rank Fusion.
**Reasoning**: Dense models alone often struggle with exact keyword/acronym matches found in academic or corporate PDFs. BM25 covers these gaps without complex embedding tuning.

## 4. No In-Memory State Between Sessions
**Decision**: Streamlit session state is only used for UI flow. All core persistence goes to SQLite.
**Reasoning**: Prevents data loss if the Streamlit app restarts.

## 5. Local Sentence Transformers
**Decision**: Use `BAAI/bge-small-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2` locally via HuggingFace `sentence-transformers`.
**Reasoning**: Avoids network latency for massive batch processing during PDF ingestion. CPU-optimized models were explicitly chosen over large neural models for speed.
