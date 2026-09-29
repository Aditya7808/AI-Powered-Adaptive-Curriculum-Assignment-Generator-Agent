import pytest
from pydantic import ValidationError

from src.schemas.assignment import AnswerKey, Assignment, BloomLevel, Question
from src.schemas.curriculum import (
    Curriculum,
    DifficultyLevel,
)
from src.schemas.learner import LearnerProfile, QuestionType, SkillGap


def test_learner_profile_valid():
    profile = LearnerProfile(
        name="Test",
        background="None",
        current_skills={"python": 1},
        prior_experience_years=2.0,
        available_hours_per_week=10,
        domain="software",
    )
    assert profile.name == "Test"


def test_learner_profile_invalid_hours():
    with pytest.raises(ValidationError):
        LearnerProfile(
            name="Test",
            background="None",
            current_skills={"python": 1},
            prior_experience_years=2.0,
            available_hours_per_week=100,  # Invalid > 60
            domain="software",
        )


def test_skill_gap_computed():
    gap = SkillGap(
        skill="python", current_level=2, target_level=4, priority=1, rationale="test"
    )
    assert gap.gap_size == 2


def test_curriculum_min_modules():
    with pytest.raises(ValidationError):
        Curriculum(
            curriculum_id="c1",
            learner_id="l1",
            version=1,
            title="Test Curr",
            modules=[],  # Invalid, needs 3
            alignment_notes="none",
            created_at="2023-10-10T10:00:00",
        )


def test_question_options_validation():
    # Valid MCQ
    q = Question(
        question_id="q1",
        text="What?",
        type=QuestionType.mcq,
        difficulty=DifficultyLevel.easy,
        bloom_level=BloomLevel.remember,
        competency="test",
        options=["A", "B", "C", "D"],
        answer_key=AnswerKey(
            correct_answer="A", explanation="test", common_mistakes=[]
        ),
        marks=10,
    )
    assert q.options is not None

    # Invalid MCQ (missing correct answer in options)
    with pytest.raises(ValidationError):
        Question(
            question_id="q2",
            text="What?",
            type=QuestionType.mcq,
            difficulty=DifficultyLevel.easy,
            bloom_level=BloomLevel.remember,
            competency="test",
            options=["A", "B", "C", "D"],
            answer_key=AnswerKey(
                correct_answer="E", explanation="test", common_mistakes=[]
            ),
            marks=10,
        )


def test_assignment_total_marks():
    q = Question(
        question_id="q1",
        text="What?",
        type=QuestionType.short_answer,
        difficulty=DifficultyLevel.easy,
        bloom_level=BloomLevel.remember,
        competency="test",
        answer_key=AnswerKey(
            correct_answer="A", explanation="test", common_mistakes=[]
        ),
        marks=5,
    )
    with pytest.raises(ValidationError):
        Assignment(
            assignment_id="a1",
            curriculum_id="c1",
            module_id="m1",
            title="Assig 1",
            difficulty=DifficultyLevel.easy,
            instructions="Do it",
            questions=[q],
            rubric=[],
            learning_outcomes=[],
            total_marks=10,  # Invalid, should be 5
            estimated_time_minutes=30,
            status="draft",
        )
