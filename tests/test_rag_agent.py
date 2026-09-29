from src.agents.rag_agent import build_rag_graph
from src.config import config
from src.llm.provider import FakeLLM
from src.schemas.rag import RetrievedChunk


class DummyRetriever:
    def retrieve(self, query, doc_ids=None):
        return {
            "chunks": [
                RetrievedChunk(
                    chunk_id="c1",
                    doc_id="d1",
                    file_name="f.pdf",
                    page=1,
                    chunk_index=0,
                    text="Apple is red.",
                    dense_score=0.9,
                    bm25_score=0.0,
                    fused_score=0.9,
                    rerank_score=0.9,
                    relevance_band="relevant",
                    is_neighbor_expansion=False,
                )
            ]
        }


def test_rag_agent_success():
    config.LLM_PROVIDER = "fake"

    # Patch FakeLLM responses
    FakeLLM.canned_responses = {
        "Answer the question using only the provided context": "Apple is red.",
        "Verify if the answer is grounded in the context and complete": '{"is_grounded": true, "is_complete": true, "reasoning": "Valid"}',
    }

    graph = build_rag_graph()

    initial_state = {
        "query": "What color is an apple?",
        "doc_ids": None,
        "retrieved_chunks": [],
        "answer": "",
        "verification_attempts": 0,
        "is_verified": False,
        "retriever": DummyRetriever(),
    }

    res = graph.invoke(initial_state)

    assert res["is_verified"] is True
    assert res["verification_attempts"] == 1
    assert "Apple is red" in res["answer"]


def test_rag_agent_retry_then_max():
    config.LLM_PROVIDER = "fake"

    FakeLLM.canned_responses = {
        "Answer the question using only the provided context": "Apple is blue.",
        "Verify if the answer is grounded in the context and complete": '{"is_grounded": false, "is_complete": true, "reasoning": "Invalid"}',
    }

    graph = build_rag_graph()

    initial_state = {
        "query": "What color is an apple?",
        "doc_ids": None,
        "retrieved_chunks": [],
        "answer": "",
        "verification_attempts": 0,
        "is_verified": False,
        "retriever": DummyRetriever(),
    }

    res = graph.invoke(initial_state)

    assert res["is_verified"] is False
    assert res["verification_attempts"] == 2  # Hit max attempts
