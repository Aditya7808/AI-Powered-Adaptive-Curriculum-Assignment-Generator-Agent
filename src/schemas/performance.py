from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class QuestionResult(BaseModel):
    question_id: str
    marks_awarded: float
    max_marks: float
    time_taken_seconds: int | None = None


class SubmissionResult(BaseModel):
    submission_id: str
    learner_id: str
    assignment_id: str
    results: list[QuestionResult]
    submitted_at: datetime


class PerformanceReport(BaseModel):
    learner_id: str
    overall_score_pct: float
    score_by_competency: dict[str, float]
    score_by_difficulty: dict[str, float]
    score_by_question_type: dict[str, float]
    strengths: list[str]
    weaknesses: list[str]
    trend: Literal["improving", "flat", "declining", "insufficient_data"]


class AdjustmentRecommendation(BaseModel):
    target: Literal["curriculum", "next_assignment"]
    action: Literal[
        "add_remedial_module",
        "skip_or_compress_module",
        "increase_difficulty",
        "decrease_difficulty",
        "change_question_mix",
        "extend_timeline",
    ]
    competency: str
    reason: str
    priority: int
