# AI-Powered Adaptive Curriculum and Assignment Generator Agent

A production-grade multi-agent system that combines an Agentic RAG (Retrieval-Augmented Generation) pipeline with a suite of specialized learning agents. Users upload course PDFs, the system retrieves and self-verifies the most relevant content, and those grounded answers power a personalized competency-based curriculum, validated multi-level assignments with answer keys and rubrics, automated grading, and performance-based adaptation.

---

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
  - [Windows](#windows)
  - [macOS](#macos)
  - [Linux (Ubuntu/Debian)](#linux-ubuntudebian)
- [Configuration](#configuration)
  - [Environment Variables Reference](#environment-variables-reference)
  - [Offline / Testing Mode](#offline--testing-mode)
- [Running the Application](#running-the-application)
  - [Web Interface (Streamlit)](#web-interface-streamlit)
  - [End-to-End Demo Script](#end-to-end-demo-script)
  - [Evaluation and Benchmarking](#evaluation-and-benchmarking)
- [Running Tests](#running-tests)
- [Module Reference](#module-reference)
  - [Document Pipeline](#document-pipeline)
  - [Learning Agents](#learning-agents)
  - [Supporting Layers](#supporting-layers)
- [How It Works: Complete Workflow](#how-it-works-complete-workflow)
- [Web UI Pages](#web-ui-pages)
- [API / Orchestrator Reference](#api--orchestrator-reference)
- [Design Decisions](#design-decisions)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Key Features

- **Agentic RAG Pipeline** -- Self-verifying retrieval graph that automatically rewrites poor queries and validates answer grounding before passing context to downstream agents.
- **Hybrid Retrieval** -- Combines dense vector search (ChromaDB) with sparse keyword search (BM25) using Reciprocal Rank Fusion, followed by Cross-Encoder reranking for maximum recall and precision.
- **Curriculum Generation** -- Analyzes learner profiles, diagnoses skill gaps against target objectives, and produces Bloom's Taxonomy-aligned, multi-module learning pathways with prerequisite ordering.
- **Multi-Format Assignment Generation** -- Creates question banks spanning MCQ, Short Answer, Coding, and Problem Solving with detailed rubrics, marking schemes, and model answer keys.
- **Automated Validation Loop** -- Checks generated assignments for duplicate questions, rubric completeness, Bloom's level alignment, and marks balance. Rejects and regenerates if quality gates fail.
- **Intelligent Grading** -- Deterministic scoring for objective questions; rubric-grounded LLM evaluation for open-ended and coding responses with misconception detection.
- **Dynamic Adaptation** -- Modifies the learning pathway based on performance: injects remedial modules for low scores, unlocks advanced challenges for high performers.
- **Multi-Format Export** -- Exports curricula, student assignment sheets, and teacher grading cards to Markdown, JSON, and Microsoft Word (.docx).
- **Professional Web Dashboard** -- Five-page Streamlit application with document management, curriculum builder, assignment studio, and learner assessment views.
- **Fully Offline Testable** -- Ships with `FakeLLM`, `FakeEmbeddings`, and `FakeReranker` test doubles so the entire system can run without any API keys.

---

## System Architecture

```
PDF Documents / Syllabus / Textbooks
                 |
                 v
+--------------------------------------------------------+
|  Document Ingestion Pipeline                           |
|  PyMuPDF Loader --> Semantic Chunker --> SHA-256 Dedup |
+--------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------+
|  Hybrid Dual-Index Storage                             |
|  Dense: ChromaDB (HNSW Vector Similarity)              |
|  Sparse: BM25 (Exact Keyword Match)                   |
|  Fusion: Reciprocal Rank Fusion + Cross-Encoder        |
+--------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------+
|  Agentic RAG (Self-Verifying State Machine)            |
|  Retrieve --> Relevance Check --> Query Rewrite (loop) |
|  Generate Answer --> Grounding Verification            |
+--------------------------------------------------------+
                 |  (Verified Domain Context)
                 v
+--------------------------------------------------------+
|  Curriculum Agent                                      |
|  Profile Analysis --> Skill Gap Diagnosis              |
|  Prerequisite Sorting --> Bloom's Taxonomy Modules     |
+--------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------+
|  Assignment Agent                                      |
|  Context-Grounded Question Generation                  |
|  MCQ + Short Answer + Coding + Problem Solving         |
|  Rubrics + Answer Keys + Point Allocations             |
+--------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------+
|  Validation Agent (Quality Gate Loop)                  |
|  Duplicate Detection + Rubric Check + Bloom's Verify   |
|  Reject --> Feedback --> Regenerate (max 2 rounds)     |
+--------------------------------------------------------+
                 |  (Approved Assignment)
                 v
+--------------------------------------------------------+
|  Evaluation Agent                                      |
|  Deterministic MCQ Grading                             |
|  Rubric-Grounded LLM Evaluation (Open/Coding)         |
|  Misconception Diagnosis + Feedback Compilation        |
+--------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------+
|  Adaptation Agent                                      |
|  Score < 60%: Insert Remedial Modules                  |
|  Score > 85%: Unlock Advanced Topics                   |
|  Dynamically Mutate Curriculum Pathway                 |
+--------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------+
|  Export Engine + Professional Web UI                    |
|  Markdown / JSON / DOCX Output                         |
|  Streamlit Dashboard (5 Pages)                         |
+--------------------------------------------------------+
```

---

## Project Structure

```
Assignment Generator Agent/
|
|-- app/                              # Streamlit Web Application
|   |-- Home.py                       # Dashboard with KPI metrics
|   |-- pages/
|       |-- 1_Document_Management.py  # PDF upload, chunking stats
|       |-- 2_Curriculum_Generator.py # Learner profiling, curriculum builder
|       |-- 3_Assignment_Studio.py    # Question generation with validation
|       |-- 4_Learner_Assessment.py   # Grading and adaptation triggers
|
|-- src/                              # Core Application Source Code
|   |-- __init__.py
|   |-- config.py                     # Pydantic Settings (reads .env)
|   |-- orchestrator.py               # Central lifecycle controller
|   |
|   |-- agents/                       # Specialized AI Agents
|   |   |-- curriculum_agent.py       # Skill gap analysis + syllabus planner
|   |   |-- assignment_agent.py       # Question bank generator
|   |   |-- evaluation_agent.py       # Submission grading + performance analysis
|   |   |-- adaptation_agent.py       # Dynamic pathway mutation
|   |   |-- rag_agent.py              # Agentic RAG state machine builder
|   |
|   |-- documents/                    # Document Processing Pipeline
|   |   |-- loader.py                 # PyMuPDF-based PDF text extractor
|   |   |-- chunker.py               # Semantic and fixed-size chunkers
|   |   |-- ingestion.py             # Hash-based dedup + metadata enrichment
|   |   |-- vectorstore.py           # ChromaDB wrapper with HNSW tuning
|   |
|   |-- retrieval/                    # Hybrid Search Engine
|   |   |-- hybrid_retriever.py       # Orchestrates dense + sparse + reranking
|   |   |-- bm25_index.py            # BM25 sparse index (rank-bm25)
|   |   |-- embeddings.py            # Sentence-Transformers + FakeEmbeddings
|   |   |-- reranker.py              # Cross-Encoder reranker + FakeReranker
|   |   |-- fusion.py                # Reciprocal Rank Fusion implementation
|   |   |-- expansion.py             # Neighbor chunk expansion
|   |   |-- cache.py                 # TTL-based query result caching
|   |   |-- models.py                # RetrievalResult data models
|   |   |-- timing.py                # Latency measurement utilities
|   |
|   |-- llm/                         # Language Model Abstraction
|   |   |-- provider.py              # LiteLLMProvider + FakeLLM with auto-repair
|   |
|   |-- schemas/                      # Pydantic V2 Data Contracts
|   |   |-- __init__.py              # Re-exports all schemas
|   |   |-- learner.py               # LearnerProfile, SkillGapReport
|   |   |-- curriculum.py            # Curriculum, Module, Milestone
|   |   |-- assignment.py            # Assignment, Question, Rubric
|   |   |-- performance.py           # SubmissionResult, PerformanceReport
|   |   |-- documents.py             # UploadedDocument
|   |   |-- rag.py                   # RAGAnswer, RetrievalState
|   |
|   |-- export/                       # Multi-Format Export Engine
|   |   |-- __init__.py
|   |   |-- exporter.py              # Markdown, JSON, and DOCX exporters
|   |
|   |-- validation/                   # Assignment Quality Validation
|   |   |-- __init__.py
|   |   |-- validator.py             # Duplicate detection, rubric checks
|   |
|   |-- storage/                      # Persistence Layer
|       |-- db.py                     # SQLite operations (documents, KB versioning)
|
|-- tests/                            # Test Suite (39 tests)
|   |-- setup_test_db.py             # Shared test database fixture
|   |-- test_adaptation_agent.py     # 3 tests
|   |-- test_assignment_agent.py     # 3 tests
|   |-- test_curriculum.py           # 2 tests
|   |-- test_documents.py            # 3 tests
|   |-- test_evaluation_agent.py     # 3 tests
|   |-- test_export.py               # 6 tests
|   |-- test_provider.py             # 4 tests
|   |-- test_rag_agent.py            # 2 tests
|   |-- test_retrieval.py            # 3 tests
|   |-- test_schemas.py              # 6 tests
|   |-- test_validation.py           # 4 tests
|
|-- scripts/                          # Utility Scripts
|   |-- demo_end_to_end.py           # Full lifecycle demo (offline)
|   |-- eval_retrieval.py            # Retrieval quality evaluation harness
|   |-- benchmark_latency.py         # Component latency benchmarking
|
|-- docs/                             # Documentation
|   |-- DECISIONS.md                 # Architectural decision records
|   |-- RETRIEVAL_EVAL.md            # Retrieval evaluation methodology
|
|-- .env.example                      # Environment variable template
|-- .gitignore                        # Git ignore rules
|-- requirements.txt                  # Python dependencies
|-- README.md                         # This file
```

---

## Prerequisites

| Requirement | Version | Notes |
| :--- | :--- | :--- |
| Python | 3.10 or higher | Required. Python 3.12 recommended. |
| pip | Latest | Comes with Python. |
| Git | Any recent version | For cloning the repository. |
| HuggingFace API Token | (Optional) | Only needed if using real LLM inference. Not required for offline/testing mode. |

---

## Installation Guide

### Windows

**Step 1: Install Python**

Download Python 3.12+ from [python.org](https://www.python.org/downloads/). During installation, check "Add Python to PATH".

Verify installation:
```powershell
python --version
pip --version
```

**Step 2: Clone the Repository**
```powershell
git clone https://github.com/Aditya7808/AI-Powered-Adaptive-Curriculum-Assignment-Generator-Agent.git
cd AI-Powered-Adaptive-Curriculum-Assignment-Generator-Agent
```

**Step 3: Create and Activate Virtual Environment**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you encounter an execution policy error, run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Step 4: Install Dependencies**
```powershell
pip install -r requirements.txt
```

**Step 5: Configure Environment**
```powershell
copy .env.example .env
```

Edit `.env` with your preferred text editor. For offline testing without any API keys:
```
LLM_PROVIDER=fake
```

**Step 6: Run the Application**
```powershell
python -m streamlit run app/Home.py
```

The app will open automatically at `http://localhost:8501`.

---

### macOS

**Step 1: Install Python**

macOS typically comes with Python 3. If not, install via Homebrew:
```bash
brew install python@3.12
```

Verify:
```bash
python3 --version
pip3 --version
```

**Step 2: Clone the Repository**
```bash
git clone https://github.com/Aditya7808/AI-Powered-Adaptive-Curriculum-Assignment-Generator-Agent.git
cd AI-Powered-Adaptive-Curriculum-Assignment-Generator-Agent
```

**Step 3: Create and Activate Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Step 4: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 5: Configure Environment**
```bash
cp .env.example .env
```

Edit `.env` with nano, vim, or any editor:
```bash
nano .env
```

For offline testing, set:
```
LLM_PROVIDER=fake
```

**Step 6: Run the Application**
```bash
streamlit run app/Home.py
```

The app will open at `http://localhost:8501`.

---

### Linux (Ubuntu/Debian)

**Step 1: Install Python and pip**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git -y
```

Verify:
```bash
python3 --version
pip3 --version
```

**Step 2: Clone the Repository**
```bash
git clone https://github.com/Aditya7808/AI-Powered-Adaptive-Curriculum-Assignment-Generator-Agent.git
cd AI-Powered-Adaptive-Curriculum-Assignment-Generator-Agent
```

**Step 3: Create and Activate Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Step 4: Install Dependencies**
```bash
pip install -r requirements.txt
```

If you encounter build errors for `sentence-transformers` or `torch`, install build tools:
```bash
sudo apt install build-essential python3-dev -y
```

**Step 5: Configure Environment**
```bash
cp .env.example .env
nano .env
```

For offline testing, set:
```
LLM_PROVIDER=fake
```

**Step 6: Run the Application**
```bash
streamlit run app/Home.py
```

The app will open at `http://localhost:8501`.

---

## Configuration

### Environment Variables Reference

Copy `.env.example` to `.env` and configure the following:

#### Core Settings

| Variable | Default | Description |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `huggingface` | LLM backend. Set to `fake` for offline testing without any API keys. |
| `HUGGINGFACEHUB_API_TOKEN` | (empty) | Your HuggingFace API token. Required only when `LLM_PROVIDER=huggingface`. |
| `HF_LLM_REPO_ID` | (empty) | HuggingFace model repository ID (e.g., `meta-llama/Llama-3-70b-chat-hf`). |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence Transformer model for dense embeddings. Runs locally. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-Encoder model for reranking. Runs locally. |
| `DB_PATH` | `data/app.db` | SQLite database file path for document and state persistence. |

#### Document Pipeline Settings

| Variable | Default | Description |
| :--- | :--- | :--- |
| `CHROMA_DIR` | `data/chroma` | ChromaDB persistent storage directory. |
| `BM25_DIR` | `data/bm25` | BM25 index storage directory. |
| `UPLOAD_DIR` | `data/uploads` | Directory for uploaded PDF files. |
| `MAX_UPLOAD_MB` | `25` | Maximum file size per PDF upload (megabytes). |
| `MAX_FILES_PER_UPLOAD` | `10` | Maximum number of PDFs per upload batch. |

#### Retrieval Tuning

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ENABLE_BM25` | `true` | Enable BM25 sparse retrieval alongside dense search. |
| `ENABLE_RERANKER` | `true` | Enable Cross-Encoder reranking of fused results. |
| `ENABLE_NEIGHBOR_EXPANSION` | `true` | Expand results with neighboring chunks for context. |
| `ENABLE_HYDE` | `false` | Enable Hypothetical Document Embeddings (experimental). |
| `DENSE_TOP_N` | `20` | Number of candidates from dense (vector) search. |
| `BM25_TOP_N` | `20` | Number of candidates from BM25 (keyword) search. |
| `FUSION_K` | `60` | RRF fusion constant (higher = smoother blending). |
| `RERANK_CANDIDATES` | `24` | Number of candidates sent to Cross-Encoder. |
| `FINAL_TOP_K` | `5` | Final number of chunks returned to the LLM. |

#### Validation and Quality

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DUPLICATE_THRESHOLD` | `0.85` | Cosine similarity threshold for duplicate question detection. |
| `QUALITY_THRESHOLD` | `0.7` | Minimum quality score for assignment approval. |
| `MAX_REGEN_ROUNDS` | `2` | Maximum validation-regeneration cycles before hard stop. |

#### Performance Tuning

| Variable | Default | Description |
| :--- | :--- | :--- |
| `CACHE_TTL_SECONDS` | `900` | Query cache time-to-live (15 minutes). |
| `CACHE_MAX_ENTRIES` | `256` | Maximum cached query results. |
| `LLM_TIMEOUT_SECONDS` | `30` | Timeout for LLM API calls. |
| `HNSW_M` | `16` | HNSW graph connectivity parameter. |
| `HNSW_CONSTRUCTION_EF` | `100` | HNSW construction search depth. |
| `HNSW_SEARCH_EF` | `64` | HNSW query-time search depth. |
| `TORCH_NUM_THREADS` | `4` | PyTorch CPU thread count for local models. |

### Offline / Testing Mode

Set `LLM_PROVIDER=fake` in your `.env` file. This activates:

- **FakeLLM** -- Returns deterministic, structurally valid JSON responses that match all Pydantic schemas.
- **FakeEmbeddings** -- Returns fixed-dimension vectors without loading any model.
- **FakeReranker** -- Assigns decreasing scores to simulate reranking without a Cross-Encoder.

This mode is ideal for:
- UI development and layout testing
- Running the full test suite without API keys
- CI/CD pipelines
- Demonstrations and walkthroughs

---

## Running the Application

### Web Interface (Streamlit)

| Platform | Command |
| :--- | :--- |
| **Windows** | `python -m streamlit run app/Home.py` |
| **macOS / Linux** | `streamlit run app/Home.py` |

Opens at: `http://localhost:8501`

### End-to-End Demo Script

Runs the complete lifecycle non-interactively using offline test doubles:

| Platform | Command |
| :--- | :--- |
| **Windows** | `python scripts/demo_end_to_end.py` |
| **macOS / Linux** | `python3 scripts/demo_end_to_end.py` |

### Evaluation and Benchmarking

| Script | Purpose | Command |
| :--- | :--- | :--- |
| `eval_retrieval.py` | Measures retrieval precision, recall, and MRR | `python scripts/eval_retrieval.py` |
| `benchmark_latency.py` | Profiles latency of each pipeline component | `python scripts/benchmark_latency.py` |

---

## Running Tests

The project includes 39 automated tests covering all agents, data models, retrieval, export, and validation. All tests run fully offline using fake providers.

| Platform | Command |
| :--- | :--- |
| **Windows** | `python -m pytest -v` |
| **macOS / Linux** | `pytest -v` |

Expected output:
```
============================= test session starts =============================
collected 39 items

tests/test_adaptation_agent.py       3 passed
tests/test_assignment_agent.py       3 passed
tests/test_curriculum.py             2 passed
tests/test_documents.py              3 passed
tests/test_evaluation_agent.py       3 passed
tests/test_export.py                 6 passed
tests/test_provider.py               4 passed
tests/test_rag_agent.py              2 passed
tests/test_retrieval.py              3 passed
tests/test_schemas.py                6 passed
tests/test_validation.py             4 passed

======================= 39 passed in ~15s ========================
```

---

## Module Reference

### Document Pipeline

| Module | File | Purpose |
| :--- | :--- | :--- |
| PDF Loader | `src/documents/loader.py` | Extracts text from PDF files using PyMuPDF (fitz). Returns per-page text with metadata. |
| Chunker | `src/documents/chunker.py` | Splits extracted text into overlapping chunks. Supports fixed-size and semantic chunking modes. |
| Ingestion Service | `src/documents/ingestion.py` | Orchestrates loading, chunking, deduplication (SHA-256 file hash), metadata tagging, and vector storage. |
| Vector Store | `src/documents/vectorstore.py` | ChromaDB wrapper with HNSW index tuning, per-document filtering, and count queries. |
| BM25 Index | `src/retrieval/bm25_index.py` | Sparse keyword index using rank-bm25 for exact term matching. |
| Hybrid Retriever | `src/retrieval/hybrid_retriever.py` | Combines dense and sparse results via Reciprocal Rank Fusion, applies Cross-Encoder reranking, and supports neighbor chunk expansion. |
| Embeddings | `src/retrieval/embeddings.py` | Sentence-Transformers embedding service with a `FakeEmbeddings` stub for offline testing. |
| Reranker | `src/retrieval/reranker.py` | Cross-Encoder reranker with a `FakeReranker` stub. |
| Fusion | `src/retrieval/fusion.py` | Reciprocal Rank Fusion (RRF) implementation for merging ranked lists. |
| Cache | `src/retrieval/cache.py` | TTL-based caching for retrieval queries to avoid redundant computation. |

### Learning Agents

| Agent | File | Purpose |
| :--- | :--- | :--- |
| RAG Agent | `src/agents/rag_agent.py` | Builds the Agentic RAG state machine graph: retrieve, grade relevance, rewrite queries, generate answers, verify grounding. |
| Curriculum Agent | `src/agents/curriculum_agent.py` | Analyzes learner profiles, identifies skill gaps, and generates Bloom's Taxonomy-aligned multi-module curricula. |
| Assignment Agent | `src/agents/assignment_agent.py` | Generates context-grounded question banks (MCQ, Short Answer, Coding, Problem Solving) with rubrics and answer keys. |
| Evaluation Agent | `src/agents/evaluation_agent.py` | Grades student submissions using deterministic scoring for MCQs and rubric-grounded LLM evaluation for open-ended responses. Performs misconception diagnosis. |
| Adaptation Agent | `src/agents/adaptation_agent.py` | Dynamically mutates the curriculum based on evaluation results. Injects remedial modules for low performers and advanced challenges for high performers. |

### Supporting Layers

| Layer | File | Purpose |
| :--- | :--- | :--- |
| LLM Provider | `src/llm/provider.py` | Unified LLM abstraction supporting HuggingFace models and FakeLLM. Includes auto-repair JSON retry logic with Pydantic schema enforcement. |
| Schemas | `src/schemas/*.py` | Pydantic V2 data contracts for all inter-agent communication: LearnerProfile, Curriculum, Assignment, PerformanceReport, RAGAnswer. |
| Validation | `src/validation/validator.py` | Assignment quality validation: duplicate question detection via cosine similarity, rubric completeness checks, Bloom's level verification. |
| Export | `src/export/exporter.py` | Multi-format export engine producing Markdown, structured JSON, and Microsoft Word (.docx) documents for curricula and assignments. |
| Database | `src/storage/db.py` | SQLite operations for document metadata, knowledge base versioning, and CRUD operations. |
| Orchestrator | `src/orchestrator.py` | Central controller (`AdaptiveLearningOrchestrator`) that wires all agents into a unified lifecycle: upload, profile, curriculum, assign, evaluate, adapt. |
| Configuration | `src/config.py` | Pydantic Settings model that loads all configuration from `.env` with sensible defaults. |

---

## How It Works: Complete Workflow

### 1. Document Upload and Ingestion
Upload one or more PDF files (textbooks, syllabi, course notes). Each file is:
- Parsed with PyMuPDF for clean text extraction
- Split into overlapping chunks with metadata (page number, file name, document ID)
- Deduplicated using SHA-256 file hashes (re-uploading the same file is a no-op)
- Indexed into both ChromaDB (dense vectors) and BM25 (sparse keywords)

### 2. Knowledge Retrieval (Agentic RAG)
When any downstream agent needs domain knowledge, the Agentic RAG pipeline:
- Queries both dense and sparse indexes simultaneously
- Fuses results using Reciprocal Rank Fusion (RRF)
- Reranks the top candidates with a Cross-Encoder model
- Grades the relevance of retrieved chunks
- Rewrites the query and retries if relevance is poor (up to a configurable limit)
- Generates a grounded answer and verifies it against source chunks
- Returns only verified, citation-backed answers

### 3. Learner Profiling and Curriculum Generation
Users provide their profile (current skills, target competencies, weekly study hours, preferred learning style). The system:
- Analyzes the gap between current skills and target objectives
- Sorts prerequisites using dependency ordering
- Generates a multi-module curriculum aligned to Bloom's Taxonomy levels
- Assigns time estimates and milestones per module

### 4. Assignment Generation and Validation
Users select a curriculum module and question mix. The system:
- Generates a question bank grounded in the retrieved domain content
- Creates detailed rubrics with point allocations for each question
- Produces model answer keys for teachers
- Runs the assignment through the Validation Agent:
  - Checks for semantically duplicate questions
  - Verifies rubric completeness
  - Validates Bloom's level alignment
  - Confirms marks distribution balance
- If validation fails, the system automatically regenerates with specific feedback (up to 2 rounds)

### 5. Submission Grading
Students submit answers. The Evaluation Agent:
- Scores MCQs deterministically against the answer key
- Evaluates open-ended and coding responses using rubric-grounded LLM analysis
- Detects specific misconceptions and knowledge gaps
- Compiles per-question feedback with improvement suggestions

### 6. Dynamic Curriculum Adaptation
Based on evaluation results, the Adaptation Agent:
- Injects remedial foundational modules if score falls below 60%
- Unlocks advanced topics and stretch challenges if score exceeds 85%
- Preserves minimum module count constraints
- Returns the mutated curriculum for the next learning cycle

---

## Web UI Pages

| Page | Route | Description |
| :--- | :--- | :--- |
| Home | `/` | Dashboard with project overview, KPI metric cards, and system status indicators. |
| Document Management | `/Document_Management` | Upload PDFs, view ingestion status, monitor chunking statistics, delete documents. |
| Curriculum Generator | `/Curriculum_Generator` | Configure learner profile, set target competencies, generate and export structured curricula. |
| Assignment Studio | `/Assignment_Studio` | Select modules, configure question mix and difficulty, generate validated assignments with answer keys. |
| Learner Assessment | `/Learner_Assessment` | Grade student submissions, view performance analytics, trigger curriculum adaptation. |

---

## API / Orchestrator Reference

The `AdaptiveLearningOrchestrator` class (`src/orchestrator.py`) provides the programmatic API:

| Method | Parameters | Returns | Description |
| :--- | :--- | :--- | :--- |
| `upload_documents()` | `files: list[str]` | `list[UploadedDocument]` | Ingests PDF files into the knowledge base. |
| `list_documents()` | (none) | `list[UploadedDocument]` | Returns all uploaded documents with metadata. |
| `delete_document()` | `doc_id: str` | `None` | Removes a document and its vector chunks. |
| `ask_documents()` | `question: str, doc_ids: list` | `RAGAnswer` | Queries the Agentic RAG pipeline for a grounded answer. |
| `onboard_learner()` | `profile, objectives` | `SkillGapReport` | Analyzes learner profile against target objectives. |
| `build_curriculum()` | `profile, objectives` | `Curriculum` | Generates a structured learning pathway. |
| `create_assignment()` | `curriculum, module_id, difficulty, question_mix, learner` | `Assignment` | Generates a validated assignment for a specific module. |
| `analyze_and_adapt()` | `learner, submissions, assignments, curriculum` | `(PerformanceReport, Recommendations, Curriculum)` | Grades, analyzes, and adapts the curriculum. |

---

## Design Decisions

| Decision | Rationale |
| :--- | :--- |
| **Single ChromaDB Collection** | Simplifies retrieval logic. Cross-document search is handled via `doc_id` metadata filtering rather than multiple collections. |
| **Pydantic-Enforced Agent Parsing** | All LLM outputs are validated against Pydantic schemas with automatic retry on parse failure. Ensures reliable inter-agent communication. |
| **Hybrid RRF Retrieval** | Dense models struggle with exact keyword/acronym matches common in academic PDFs. BM25 covers these gaps without complex embedding tuning. |
| **No In-Memory State** | All core persistence uses SQLite. Prevents data loss if the Streamlit app restarts. Session state is only used for UI flow. |
| **Local Sentence Transformers** | `BAAI/bge-small-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2` run locally to avoid network latency during batch PDF ingestion. CPU-optimized models chosen for speed. |

---

## Troubleshooting

### Common Issues

**1. "streamlit: command not found" (macOS / Linux)**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
# Or use the module syntax
python3 -m streamlit run app/Home.py
```

**2. PowerShell execution policy error (Windows)**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**3. "ModuleNotFoundError: No module named 'src'"**

Make sure you are running commands from the project root directory, not from a subdirectory:
```bash
# Correct -- run from the project root
cd AI-Powered-Adaptive-Curriculum-Assignment-Generator-Agent
python -m streamlit run app/Home.py

# Incorrect -- running from inside src/ or app/
cd src
python ../app/Home.py  # This will fail
```

**4. Slow first startup**

The first run downloads embedding and reranker models (~100MB total). Subsequent runs use cached models. If using `LLM_PROVIDER=fake`, no models are downloaded.

**5. "Permission denied" errors on data directories**

Ensure the `data/` directory is writable:
```bash
# Linux / macOS
chmod -R 755 data/

# Windows (run PowerShell as Administrator)
icacls data /grant Users:F /T
```

**6. PyTorch / CUDA warnings**

The system runs on CPU by default. GPU warnings can be safely ignored. To suppress them:
```bash
export CUDA_VISIBLE_DEVICES=""  # Linux / macOS
$env:CUDA_VISIBLE_DEVICES=""    # Windows PowerShell
```

**7. Tests fail with database errors**

Delete any stale test databases and retry:
```bash
rm -f data/app_test.db    # Linux / macOS
del data\app_test.db       # Windows
python -m pytest -v
```

---

## License

This project is intended for educational and research purposes.