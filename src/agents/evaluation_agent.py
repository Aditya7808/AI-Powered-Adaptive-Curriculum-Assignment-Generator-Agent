import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from src.llm.provider import generate_structured, get_llm
from src.schemas.assignment import Assignment, Question
from src.schemas.learner import QuestionType
from src.schemas.performance import (
    AdjustmentRecommendation,
    PerformanceReport,
    QuestionResult,
    SubmissionResult,
)


class SubjectiveGrade(BaseModel):
    marks_awarded: float = Field(ge=0)
    feedback: str


class StrengthsWeaknessesTrend(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    trend: str = Field(
        description="'improving', 'flat', 'declining', or 'insufficient_data'"
    )


class AdaptationList(BaseModel):
    recommendations: list[AdjustmentRecommendation]


def grade_submission(
    assignment: Assignment,
    learner_id: str,
    answers: dict[str, Any],
    time_taken_seconds: dict[str, int] | None = None,
    llm=None,
) -> SubmissionResult:
    """
    Grades an assignment submission.
    - Evaluates objective question types (mcq, true_false) directly against answer keys.
    - Uses LLM structured evaluation for subjective/open-ended questions.
    """
    llm = llm or get_llm()
    results: list[QuestionResult] = []
    times = time_taken_seconds or {}

    for question in assignment.questions:
        q_id = question.question_id
        user_answer = answers.get(q_id, "")
        max_marks = float(question.marks)
        q_time = times.get(q_id)

        # Objective auto-grading
        if question.type in (QuestionType.mcq, QuestionType.true_false):
            correct = question.answer_key.correct_answer
            if isinstance(correct, list):
                is_correct = str(user_answer).strip().lower() in [
                    str(c).strip().lower() for c in correct
                ]
            else:
                is_correct = (
                    str(user_answer).strip().lower() == str(correct).strip().lower()
                )

            marks = max_marks if is_correct else 0.0
            results.append(
                QuestionResult(
                    question_id=q_id,
                    marks_awarded=marks,
                    max_marks=max_marks,
                    time_taken_seconds=q_time,
                )
            )

        elif question.type == QuestionType.multi_select:
            correct = question.answer_key.correct_answer
            correct_set = {
                str(c).strip().lower()
                for c in (correct if isinstance(correct, list) else [correct])
            }
            user_set = {
                str(u).strip().lower()
                for u in (
                    user_answer if isinstance(user_answer, list) else [user_answer]
                )
            }

            # Simple Jaccard or exact match
            marks = max_marks if user_set == correct_set else 0.0
            results.append(
                QuestionResult(
                    question_id=q_id,
                    marks_awarded=marks,
                    max_marks=max_marks,
                    time_taken_seconds=q_time,
                )
            )

        else:
            # Subjective / Open-ended / Coding / Scenario
            prompt = f"""Evaluate this learner response against the question, answer key, and rubric.

Question:
{question.text}

Max Marks: {max_marks}

Reference Answer Key:
{question.answer_key.model_dump_json(indent=2)}

Learner's Answer:
{user_answer}

Provide the marks awarded (from 0 to {max_marks}) and constructive feedback."""

            try:
                grade = generate_structured(prompt, SubjectiveGrade, llm)
                awarded = min(max(grade.marks_awarded, 0.0), max_marks)
            except Exception:
                # Fallback heuristic if generation fails
                awarded = max_marks * 0.5

            results.append(
                QuestionResult(
                    question_id=q_id,
                    marks_awarded=awarded,
                    max_marks=max_marks,
                    time_taken_seconds=q_time,
                )
            )

    return SubmissionResult(
        submission_id=f"sub-{uuid.uuid4().hex[:8]}",
        learner_id=learner_id,
        assignment_id=assignment.assignment_id,
        results=results,
        submitted_at=datetime.now(timezone.utc),
    )


def analyze_performance(
    learner_id: str,
    assignment: Assignment,
    submission: SubmissionResult,
    history: list[SubmissionResult] | None = None,
    llm=None,
) -> PerformanceReport:
    """
    Computes performance metrics across competencies, difficulties, and question types.
    Identifies strengths, weaknesses, and trend using LLM reasoning.
    """
    llm = llm or get_llm()
    q_map: dict[str, Question] = {q.question_id: q for q in assignment.questions}

    total_awarded = sum(r.marks_awarded for r in submission.results)
    total_max = sum(r.max_marks for r in submission.results)
    overall_pct = (total_awarded / total_max * 100.0) if total_max > 0 else 0.0

    score_by_comp: dict[str, list[float]] = {}
    score_by_diff: dict[str, list[float]] = {}
    score_by_type: dict[str, list[float]] = {}

    for res in submission.results:
        q = q_map.get(res.question_id)
        if not q:
            continue
        pct = (res.marks_awarded / res.max_marks * 100.0) if res.max_marks > 0 else 0.0

        score_by_comp.setdefault(q.competency, []).append(pct)
        score_by_diff.setdefault(q.difficulty.value, []).append(pct)
        score_by_type.setdefault(q.type.value, []).append(pct)

    avg_by_comp = {k: round(sum(v) / len(v), 2) for k, v in score_by_comp.items()}
    avg_by_diff = {k: round(sum(v) / len(v), 2) for k, v in score_by_diff.items()}
    avg_by_type = {k: round(sum(v) / len(v), 2) for k, v in score_by_type.items()}

    # Determine default trend if not provided by LLM
    computed_trend = "insufficient_data"
    if history and len(history) >= 2:
        prev_scores = []
        for h in history:
            h_awarded = sum(r.marks_awarded for r in h.results)
            h_max = sum(r.max_marks for r in h.results)
            if h_max > 0:
                prev_scores.append(h_awarded / h_max * 100.0)
        if prev_scores:
            avg_prev = sum(prev_scores) / len(prev_scores)
            if overall_pct > avg_prev + 5:
                computed_trend = "improving"
            elif overall_pct < avg_prev - 5:
                computed_trend = "declining"
            else:
                computed_trend = "flat"

    prompt = f"""Analyze the learner's assessment performance and identify strengths, weaknesses, and trend.

Overall Score: {overall_pct:.1f}%
Performance by Competency: {avg_by_comp}
Performance by Difficulty: {avg_by_diff}
Performance by Question Type: {avg_by_type}
Default Computed Trend: {computed_trend}

Extract:
- strengths (list of specific competency or skill areas where learner excelled)
- weaknesses (list of areas requiring improvement)
- trend ('improving', 'flat', 'declining', or 'insufficient_data')"""

    try:
        analysis = generate_structured(prompt, StrengthsWeaknessesTrend, llm)
        trend = (
            analysis.trend
            if analysis.trend in ("improving", "flat", "declining", "insufficient_data")
            else computed_trend
        )
        strengths = analysis.strengths
        weaknesses = analysis.weaknesses
    except Exception:
        trend = computed_trend
        strengths = [comp for comp, sc in avg_by_comp.items() if sc >= 75.0]
        weaknesses = [comp for comp, sc in avg_by_comp.items() if sc < 70.0]

    return PerformanceReport(
        learner_id=learner_id,
        overall_score_pct=round(overall_pct, 2),
        score_by_competency=avg_by_comp,
        score_by_difficulty=avg_by_diff,
        score_by_question_type=avg_by_type,
        strengths=strengths,
        weaknesses=weaknesses,
        trend=trend,
    )


def recommend_adaptations(
    report: PerformanceReport, llm=None
) -> list[AdjustmentRecommendation]:
    """
    Generates actionable curriculum and assignment adaptation recommendations based on performance report.
    """
    llm = llm or get_llm()

    prompt = f"""Based on the following learner performance report, recommend adaptations for the curriculum or next assignment.

Learner ID: {report.learner_id}
Overall Score: {report.overall_score_pct}%
Competency Scores: {report.score_by_competency}
Strengths: {report.strengths}
Weaknesses: {report.weaknesses}
Trend: {report.trend}

Allowed actions:
- 'add_remedial_module'
- 'skip_or_compress_module'
- 'increase_difficulty'
- 'decrease_difficulty'
- 'change_question_mix'
- 'extend_timeline'

Allowed target: 'curriculum' or 'next_assignment'

Generate prioritized adjustment recommendations."""

    try:
        rec_list = generate_structured(prompt, AdaptationList, llm)
        return rec_list.recommendations
    except Exception:
        # Rule-based fallback recommendations
        recs = []
        if report.overall_score_pct >= 85.0:
            recs.append(
                AdjustmentRecommendation(
                    target="next_assignment",
                    action="increase_difficulty",
                    competency="General",
                    reason="High performance indicates readiness for more challenging material.",
                    priority=1,
                )
            )
        elif report.overall_score_pct < 60.0:
            recs.append(
                AdjustmentRecommendation(
                    target="curriculum",
                    action="add_remedial_module",
                    competency=report.weaknesses[0] if report.weaknesses else "General",
                    reason="Low overall score indicates need for foundational reinforcement.",
                    priority=1,
                )
            )
        return recs
