import json

from src.agents.curriculum_agent import analyze_profile, generate_curriculum
from src.config import config
from src.llm.provider import FakeLLM
from src.schemas.learner import LearnerProfile


def test_analyze_profile():
    config.LLM_PROVIDER = "fake"

    FakeLLM.canned_responses["Analyze the following raw text"] = json.dumps(
        {
            "learner_id": "L1",
            "name": "Test User",
            "background": "Some background",
            "current_skills": {"Python": 3},
            "prior_experience_years": 2.0,
            "available_hours_per_week": 10,
            "domain": "Computer Science",
            "preferred_formats": ["mcq"],
        }
    )

    profile = analyze_profile("The user is an adult who likes AI.", "")
    assert profile.learner_id == "L1"
    assert profile.name == "Test User"
    assert profile.domain == "Computer Science"


def test_generate_curriculum():
    config.LLM_PROVIDER = "fake"

    profile = LearnerProfile(
        learner_id="L1",
        name="Test User",
        background="Some background",
        current_skills={"Python": 3},
        prior_experience_years=2.0,
        available_hours_per_week=10,
        domain="Computer Science",
        preferred_formats=["mcq"],
    )

    module_template = {
        "module_id": "M1",
        "title": "Intro",
        "week_range": "Week 1",
        "competencies": ["basics"],
        "learning_outcomes": ["outcome1"],
        "topics": ["topic1"],
        "activities": ["activity1"],
        "resources": [],
        "prerequisites": [],
        "difficulty": "medium",
    }

    FakeLLM.canned_responses["Generate a personalized curriculum"] = json.dumps(
        {
            "curriculum_id": "C1",
            "learner_id": "L1",
            "version": 1,
            "title": "Basic AI",
            "modules": [module_template for _ in range(3)],  # Requires 3 modules
            "alignment_notes": "Good notes",
            "created_at": "2023-01-01T00:00:00Z",
        }
    )

    curriculum = generate_curriculum(profile, "Learn basic agents")
    assert curriculum.curriculum_id == "C1"
    assert len(curriculum.modules) == 3
    assert curriculum.modules[0].title == "Intro"
