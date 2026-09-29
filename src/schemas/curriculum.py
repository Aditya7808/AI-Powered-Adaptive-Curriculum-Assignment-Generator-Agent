from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from .learner import DifficultyLevel


class ResourceReference(BaseModel):
    title: str
    file_name: str
    page: int
    chunk_id: str
    snippet: str = Field(max_length=300)
    standard_or_framework: str | None = None


class Module(BaseModel):
    module_id: str
    title: str
    week_range: str
    competencies: list[str]
    learning_outcomes: list[str]
    topics: list[str]
    activities: list[str]
    resources: list[ResourceReference]
    prerequisites: list[str]
    difficulty: DifficultyLevel


class Curriculum(BaseModel):
    curriculum_id: str
    learner_id: str
    version: int
    title: str
    modules: list[Module]
    alignment_notes: str
    created_at: datetime

    @model_validator(mode="after")
    def check_min_modules(self) -> "Curriculum":
        if len(self.modules) < 3:
            raise ValueError("Curriculum must have at least 3 modules")
        return self
