import json

from src.agents.assignment_agent import (
    generate_assignment,
    generate_targeted_questions,
    retrieve_competency_context,
)
from src.config import config
from src.llm.provider import FakeLLM
from src.schemas.assignment import (
    Assignment,
    Question,
)
from src.schemas.learner import (
    BloomLevel,
    DifficultyLevel,
    LearnerProfile,
    QuestionType,
)

config.LLM_PROVIDER = "fake"


class MockRetriever:
    def retrieve(self, query: str, top_k: int = 3):
        return [
            {"text": f"Found content for {query}", "file_name": "guide.pdf", "page": 2},
            {
                "text": f"Additional notes on {query}",
                "file_name": "guide.pdf",
                "page": 3,
            },
        ]


def test_retrieve_competency_context():
    # When no retriever
    assert retrieve_competency_context(["Python Basics"], retriever=None) == ""

    # With mock retriever
    retriever = MockRetriever()
    context = retrieve_competency_context(
        ["Python Basics", "Data Structures"], retriever=retriever
    )
    assert "[guide.pdf p.2] Found content for Python Basics" in context
    assert "[guide.pdf p.3] Additional notes on Data Structures" in context


def test_generate_assignment():
    valid_assignment_dict = {
        "assignment_id": "assign-123",
        "curriculum_id": "curr-1",
        "module_id": "mod-1",
        "title": "Module 1 Assessment: Python Foundations",
        "difficulty": "medium",
        "instructions": "Complete all questions within 45 minutes.",
        "questions": [
            {
                "question_id": "q1",
                "text": "What is the output of print(2 ** 3)?",
                "type": "mcq",
                "difficulty": "easy",
                "bloom_level": "remember",
                "competency": "Python Operators",
                "options": ["6", "8", "9", "5"],
                "answer_key": {
                    "correct_answer": "8",
                    "explanation": "2 raised to the power of 3 equals 8.",
                    "common_mistakes": ["Multiplying 2 by 3 to get 6"],
                },
                "marks": 5,
            },
            {
                "question_id": "q2",
                "text": "Explain the difference between a list and a tuple in Python.",
                "type": "short_answer",
                "difficulty": "medium",
                "bloom_level": "understand",
                "competency": "Data Structures",
                "options": None,
                "answer_key": {
                    "correct_answer": "Lists are mutable whereas tuples are immutable.",
                    "explanation": "Lists can be modified in-place; tuples cannot.",
                    "common_mistakes": ["Thinking tuples use square brackets"],
                },
                "marks": 5,
            },
        ],
        "rubric": [
            {
                "criterion": "Conceptual Accuracy",
                "description": "Demonstrates clear understanding of Python syntax and structures.",
                "levels": [
                    {
                        "label": "Proficient",
                        "points": 5,
                        "descriptor": "Complete and accurate explanations.",
                    },
                    {
                        "label": "Novice",
                        "points": 2,
                        "descriptor": "Partial understanding with minor errors.",
                    },
                ],
                "max_points": 5,
            }
        ],
        "learning_outcomes": [
            "Understand basic operators",
            "Distinguish mutable and immutable types",
        ],
        "total_marks": 10,
        "estimated_time_minutes": 30,
        "status": "draft",
    }

    FakeLLM.canned_responses = {
        "Generate a comprehensive, structured Assignment": json.dumps(
            valid_assignment_dict
        )
    }

    profile = LearnerProfile(
        name="Alex",
        background="Beginner CS Student",
        current_skills={"Python": 1},
        prior_experience_years=0.5,
        available_hours_per_week=10,
        domain="Software Development",
    )

    assignment = generate_assignment(
        curriculum_id="curr-100",
        module_id="mod-101",
        title="Intro to Python",
        competencies=["Python Operators", "Data Structures"],
        difficulty=DifficultyLevel.medium,
        learner_profile=profile,
        retriever=MockRetriever(),
    )

    assert isinstance(assignment, Assignment)
    assert assignment.curriculum_id == "curr-100"
    assert assignment.module_id == "mod-101"
    assert assignment.total_marks == 10
    assert len(assignment.questions) == 2
    assert assignment.questions[0].answer_key.correct_answer == "8"
    assert assignment.status == "draft"


def test_generate_targeted_questions():
    valid_question_dict = {
        "question_id": "q-target-1",
        "text": "Which method is used to add an element at the end of a list?",
        "type": "mcq",
        "difficulty": "easy",
        "bloom_level": "remember",
        "competency": "List Operations",
        "options": ["append()", "extend()", "insert()", "add()"],
        "answer_key": {
            "correct_answer": "append()",
            "explanation": "append() appends an element to the end of the list.",
            "common_mistakes": ["Confusing extend() with append()"],
        },
        "marks": 5,
    }

    FakeLLM.canned_responses = {
        "Generate a single question for": json.dumps(valid_question_dict)
    }

    question = generate_targeted_questions(
        competency="List Operations",
        question_type=QuestionType.mcq,
        difficulty=DifficultyLevel.easy,
        bloom_level=BloomLevel.remember,
        marks=5,
    )

    assert isinstance(question, Question)
    assert question.question_id == "q-target-1"
    assert question.answer_key.correct_answer == "append()"
    assert "append()" in question.options
