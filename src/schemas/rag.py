from typing import Literal

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    file_name: str
    page: int
    chunk_index: int
    text: str
    dense_score: float | None = None
    bm25_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    relevance_band: Literal["relevant", "borderline", "irrelevant", "unset"]
    is_neighbor_expansion: bool


class Citation(BaseModel):
    file_name: str
    page: int
    snippet: str = Field(max_length=300)


class RAGStep(BaseModel):
    node: str
    summary: str
    duration_ms: float


class RAGAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    grounded: bool
    confidence: float = Field(ge=0, le=1)
    found_in_documents: bool
    rewrites_used: int
    cache_hit: bool
    trace: list[RAGStep]
    latency_ms: float
    latency_breakdown_ms: dict[str, float]
