import uuid
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class DifficultyLevel(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class QuestionType(str, Enum):
    mcq = "mcq"
    multi_select = "multi_select"
    true_false = "true_false"
    short_answer = "short_answer"
    coding = "coding"
    case_study = "case_study"
    scenario_based = "scenario_based"
    fill_in_the_blank = "fill_in_the_blank"


class BloomLevel(str, Enum):
    remember = "remember"
    understand = "understand"
    apply = "apply"
    analyze = "analyze"
    evaluate = "evaluate"
    create = "create"


class LearningPace(str, Enum):
    slow = "slow"
    moderate = "moderate"
    fast = "fast"


class LearnerProfile(BaseModel):
    learner_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    background: str
    current_skills: dict[str, int]
    prior_experience_years: float
    available_hours_per_week: int
    preferred_formats: list[QuestionType] | None = None
    domain: str

    @field_validator("prior_experience_years")
    @classmethod
    def check_experience_years(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Prior experience years cannot be negative")
        return v

    @field_validator("available_hours_per_week")
    @classmethod
    def check_hours(cls, v: int) -> int:
        if not 1 <= v <= 60:
            raise ValueError("Available hours per week must be between 1 and 60")
        return v


class LearningObjective(BaseModel):
    objective_id: str
    description: str
    target_skill: str
    target_level: int = Field(ge=1, le=5)
    bloom_level: BloomLevel


class SkillGap(BaseModel):
    skill: str
    current_level: int = Field(ge=0, le=5)
    target_level: int = Field(ge=0, le=5)
    gap_size: int = 0
    priority: int = Field(ge=1, le=5)
    rationale: str

    @model_validator(mode="after")
    def compute_gap_size(self) -> "SkillGap":
        self.gap_size = self.target_level - self.current_level
        return self


class SkillGapReport(BaseModel):
    learner_id: str
    gaps: list[SkillGap]
    recommended_starting_level: DifficultyLevel
    recommended_pace: LearningPace
    estimated_total_weeks: int
    summary: str
