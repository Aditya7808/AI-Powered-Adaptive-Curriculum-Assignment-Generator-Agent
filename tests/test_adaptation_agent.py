from datetime import datetime, timezone

import pytest

from src.agents.adaptation_agent import adapt_curriculum
from src.config import config
from src.schemas.curriculum import Curriculum, Module
from src.schemas.learner import DifficultyLevel
from src.schemas.performance import AdjustmentRecommendation, PerformanceReport

config.LLM_PROVIDER = "fake"


@pytest.fixture
def base_curriculum():
    mods = [
        Module(
            module_id=f"mod-{i}",
            title=f"Module {i}: Topic {i}",
            week_range=f"Week {i}",
            competencies=[f"Skill {i}"],
            learning_outcomes=[f"Outcome {i}"],
            topics=[f"Topic {i}"],
            activities=[f"Activity {i}"],
            resources=[],
            prerequisites=[],
            difficulty=DifficultyLevel.easy,
        )
        for i in range(1, 4)
    ]
    return Curriculum(
        curriculum_id="curr-test-1",
        learner_id="learner-99",
        version=1,
        title="Python Data Engineering Path",
        modules=mods,
        alignment_notes="Initial baseline curriculum.",
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_report():
    return PerformanceReport(
        learner_id="learner-99",
        overall_score_pct=65.0,
        score_by_competency={"Skill 1": 45.0, "Skill 2": 85.0},
        score_by_difficulty={"easy": 65.0},
        score_by_question_type={"mcq": 70.0},
        strengths=["Skill 2"],
        weaknesses=["Skill 1"],
        trend="flat",
    )


def test_adapt_curriculum_add_remedial(base_curriculum, sample_report):
    recs = [
        AdjustmentRecommendation(
            target="curriculum",
            action="add_remedial_module",
            competency="Skill 1",
            reason="Low score on Skill 1 indicates remediation needed.",
            priority=1,
        )
    ]

    adapted = adapt_curriculum(base_curriculum, sample_report, recs)

    assert adapted.version == 2
    assert len(adapted.modules) == 4
    assert any("Skill 1" in mod.competencies for mod in adapted.modules)
    assert "[v2 Update]" in adapted.alignment_notes
    assert "Added remedial module for 'Skill 1'" in adapted.alignment_notes


def test_adapt_curriculum_increase_difficulty(base_curriculum, sample_report):
    recs = [
        AdjustmentRecommendation(
            target="curriculum",
            action="increase_difficulty",
            competency="General",
            reason="Learner mastered baseline concepts rapidly.",
            priority=1,
        )
    ]

    adapted = adapt_curriculum(base_curriculum, sample_report, recs)

    assert adapted.version == 2
    assert len(adapted.modules) == 3
    # Difficulty should be bumped from easy to medium
    assert all(mod.difficulty == DifficultyLevel.medium for mod in adapted.modules)
    assert "Increased difficulty" in adapted.alignment_notes


def test_adapt_curriculum_skip_preserves_min_modules(base_curriculum, sample_report):
    # Base curriculum has exactly 3 modules, so skip/compress must NOT reduce below 3
    recs = [
        AdjustmentRecommendation(
            target="curriculum",
            action="skip_or_compress_module",
            competency="Skill 2",
            reason="Already mastered Skill 2.",
            priority=2,
        )
    ]

    adapted = adapt_curriculum(base_curriculum, sample_report, recs)

    assert adapted.version == 2
    assert len(adapted.modules) >= 3
    assert "Retained module count (minimum 3 required)" in adapted.alignment_notes
