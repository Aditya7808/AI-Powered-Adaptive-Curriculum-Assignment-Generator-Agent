# AI-Powered Adaptive Curriculum & Assignment Generator Agent

A multi-agent system with a separate PDF-driven Agentic RAG pipeline: users upload PDFs, the pipeline retrieves and self-verifies the best answers, and those grounded answers power a personalized competency-based curriculum, validated multi-level assignments, and performance-based adaptation.

## Workflow

1. **Document Management:** Upload your own PDFs. The Agentic RAG pipeline extracts, embeds, and indexes them using a hybrid dense (ChromaDB) + lexical (BM25) approach, fused with RRF and reranked using a cross-encoder model.
2. **Learner Profiling:** Enter the learner's skills and availability.
3. **Curriculum Generation:** Agents generate a multi-week competency-based curriculum grounded entirely on the uploaded PDFs.
4. **Assignment Studio:** Select a module and question mix, and agents will generate questions, answer keys, and a grading rubric, which are then semantically checked for duplicates and quality.
5. **Learner Assessment:** Grade submissions and view performance analytics. The adaptation agent suggests targeted curriculum revisions based on performance.

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and configure your HuggingFace API key and model.
   - For offline testing without API keys, set `LLM_PROVIDER=fake`.
4. Run the UI: `streamlit run app/Home.py`

## Architecture

The system consists of two cooperating parts:
1. **Document Pipeline**: PDF upload -> chunking -> hybrid retrieval -> Agentic RAG graph
2. **Learning Agents**: Profiling -> Curriculum -> Assignment -> Assessment -> Adaptation

All agents utilize a unified LLM provider and communicate strictly using validated Pydantic JSON schemas.