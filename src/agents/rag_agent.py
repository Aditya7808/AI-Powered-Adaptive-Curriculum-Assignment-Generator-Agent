from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.llm.provider import generate_structured, get_llm


class RagState(TypedDict):
    query: str
    doc_ids: list[str]
    retrieved_chunks: list[dict[str, Any]]
    answer: str
    verification_attempts: int
    is_verified: bool
    retriever: Any


class VerificationResult(BaseModel):
    is_grounded: bool = Field(
        ...,
        description="Whether the answer is fully grounded in the retrieved documents.",
    )
    is_complete: bool = Field(
        ..., description="Whether the answer fully addresses the user's query."
    )
    reasoning: str = Field(..., description="Reasoning for the verification result.")


def retrieve_docs(state: RagState) -> RagState:
    retriever = state.get("retriever")
    if not retriever:
        raise ValueError("Retriever not provided in state")

    # Simplified: just retrieve
    res = retriever.retrieve(state["query"], state.get("doc_ids"))
    return {"retrieved_chunks": [c.model_dump() for c in res["chunks"]]}


def generate_answer(state: RagState) -> RagState:
    llm = get_llm()
    chunks = state.get("retrieved_chunks", [])
    context = "\n\n".join(
        [f"Source: {c['file_name']} (Page {c['page']})\n{c['text']}" for c in chunks]
    )

    prompt = f"Context:\n{context}\n\nQuestion:\n{state['query']}\n\nAnswer the question using only the provided context."

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)

    return {"answer": response.content}


def verify_answer(state: RagState) -> RagState:
    llm = get_llm()
    attempts = state.get("verification_attempts", 0) + 1

    chunks = state.get("retrieved_chunks", [])
    context = "\n\n".join([c["text"] for c in chunks])

    prompt = f"Context:\n{context}\n\nQuestion:\n{state['query']}\n\nAnswer:\n{state['answer']}\n\nVerify if the answer is grounded in the context and complete."

    # We use generate_structured to get the VerificationResult
    result = generate_structured(prompt, VerificationResult, llm)

    is_verified = result.is_grounded and result.is_complete

    return {"verification_attempts": attempts, "is_verified": is_verified}


def should_continue(state: RagState) -> str:
    if state.get("is_verified", False):
        return "end"
    if state.get("verification_attempts", 0) >= 2:
        return "end"
    return "generate"


def build_rag_graph():
    workflow = StateGraph(RagState)

    workflow.add_node("retrieve", retrieve_docs)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("verify", verify_answer)

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "verify")

    workflow.add_conditional_edges(
        "verify",
        should_continue,
        {
            "end": END,
            "generate": "generate",  # loop back if not verified and under max attempts
        },
    )

    return workflow.compile()
