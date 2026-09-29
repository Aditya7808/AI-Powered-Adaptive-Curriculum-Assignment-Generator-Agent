import json

import pytest

from src.agents.evaluation_agent import (
    analyze_performance,
    grade_submission,
    recommend_adaptations,
)
from src.config import config
from src.llm.provider import FakeLLM
from src.schemas.assignment import (
    AnswerKey,
    Assignment,
    Question,
    RubricCriterion,
    RubricLevel,
)
from src.schemas.learner import BloomLevel, DifficultyLevel, QuestionType
from src.schemas.performance import (
    AdjustmentRecommendation,
    PerformanceReport,
    SubmissionResult,
)

config.LLM_PROVIDER = "fake"


@pytest.fixture
def sample_assignment():
    return Assignment(
        assignment_id="assign-test-1",
        curriculum_id="curr-1",
        module_id="mod-1",
        title="Python Data Types & Logic",
        difficulty=DifficultyLevel.medium,
        instructions="Answer all questions carefully.",
        questions=[
            Question(
                question_id="q1",
                text="Which of the following is immutable in Python?",
                type=QuestionType.mcq,
                difficulty=DifficultyLevel.easy,
                bloom_level=BloomLevel.remember,
                competency="Data Types",
                options=["list", "dict", "tuple", "set"],
                answer_key=AnswerKey(
                    correct_answer="tuple",
                    explanation="Tuples cannot be altered after creation.",
                    common_mistakes=["Choosing list"],
                ),
                marks=5,
            ),
            Question(
                question_id="q2",
                text="Write a short explanation of how recursion works and why a base case is necessary.",
                type=QuestionType.short_answer,
                difficulty=DifficultyLevel.hard,
                bloom_level=BloomLevel.understand,
                competency="Algorithms",
                options=None,
                answer_key=AnswerKey(
                    correct_answer="Recursion involves a function calling itself until a base case stops the execution.",
                    explanation="Without a base case, recursion leads to stack overflow.",
                    common_mistakes=["Omitting base case"],
                ),
                marks=10,
            ),
        ],
        rubric=[
            RubricCriterion(
                criterion="Clarity",
                description="Clarity of explanation",
                levels=[
                    RubricLevel(label="High", points=5, descriptor="Very clear"),
                    RubricLevel(label="Low", points=1, descriptor="Unclear"),
                ],
                max_points=5,
            )
        ],
        learning_outcomes=["Understand immutability", "Understand recursion mechanics"],
        total_marks=15,
        estimated_time_minutes=30,
        status="draft",
    )


def test_grade_submission_objective_and_subjective(sample_assignment):
    # Canned response for the subjective question grading
    FakeLLM.canned_responses = {
        "Evaluate this learner response against the question": json.dumps(
            {
                "marks_awarded": 8.5,
                "feedback": "Great explanation, but could detail stack frame behavior slightly more.",
            }
        )
    }

    # q1 is correct ("tuple"), q2 has user response
    answers = {
        "q1": "tuple",
        "q2": "A function calls itself repeatedly until it hits a base condition.",
    }

    result = grade_submission(
        assignment=sample_assignment,
        learner_id="learner-42",
        answers=answers,
        time_taken_seconds={"q1": 25, "q2": 180},
    )

    assert isinstance(result, SubmissionResult)
    assert result.learner_id == "learner-42"
    assert result.assignment_id == "assign-test-1"
    assert len(result.results) == 2

    # Check objective auto-grade
    q1_res = next(r for r in result.results if r.question_id == "q1")
    assert q1_res.marks_awarded == 5.0
    assert q1_res.max_marks == 5.0
    assert q1_res.time_taken_seconds == 25

    # Check subjective LLM grade
    q2_res = next(r for r in result.results if r.question_id == "q2")
    assert q2_res.marks_awarded == 8.5
    assert q2_res.max_marks == 10.0
    assert q2_res.time_taken_seconds == 180


def test_analyze_performance(sample_assignment):
    FakeLLM.canned_responses = {
        "Analyze the learner's assessment performance": json.dumps(
            {
                "strengths": ["Data Types"],
                "weaknesses": ["Advanced Recursion"],
                "trend": "improving",
            }
        )
    }

    from datetime import datetime, timezone

    from src.schemas.performance import QuestionResult

    sub = SubmissionResult(
        submission_id="sub-test",
        learner_id="learner-42",
        assignment_id="assign-test-1",
        results=[
            QuestionResult(question_id="q1", marks_awarded=5.0, max_marks=5.0),
            QuestionResult(question_id="q2", marks_awarded=7.0, max_marks=10.0),
        ],
        submitted_at=datetime.now(timezone.utc),
    )

    report = analyze_performance(
        learner_id="learner-42", assignment=sample_assignment, submission=sub
    )

    assert isinstance(report, PerformanceReport)
    assert report.learner_id == "learner-42"
    # Total awarded = 12 / 15 = 80.0%
    assert report.overall_score_pct == 80.0
    assert report.score_by_competency["Data Types"] == 100.0
    assert report.score_by_competency["Algorithms"] == 70.0
    assert report.score_by_difficulty["easy"] == 100.0
    assert report.score_by_difficulty["hard"] == 70.0
    assert "Data Types" in report.strengths
    assert report.trend == "improving"


def test_recommend_adaptations():
    report = PerformanceReport(
        learner_id="learner-42",
        overall_score_pct=92.5,
        score_by_competency={"Data Types": 100.0, "Algorithms": 85.0},
        score_by_difficulty={"easy": 100.0, "medium": 85.0},
        score_by_question_type={"mcq": 100.0, "short_answer": 85.0},
        strengths=["Data Types", "Algorithms"],
        weaknesses=[],
        trend="improving",
    )

    FakeLLM.canned_responses = {
        "recommend adaptations for the curriculum": json.dumps(
            {
                "recommendations": [
                    {
                        "target": "next_assignment",
                        "action": "increase_difficulty",
                        "competency": "Data Types",
                        "reason": "Mastered foundational topics with 100% score.",
                        "priority": 1,
                    }
                ]
            }
        )
    }

    recs = recommend_adaptations(report)
    assert len(recs) == 1
    assert isinstance(recs[0], AdjustmentRecommendation)
    assert recs[0].action == "increase_difficulty"
    assert recs[0].target == "next_assignment"
    assert recs[0].priority == 1
