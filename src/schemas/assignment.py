from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .learner import BloomLevel, DifficultyLevel, QuestionType


class AnswerKey(BaseModel):
    correct_answer: str | list[str]
    explanation: str
    common_mistakes: list[str]


class RubricLevel(BaseModel):
    label: str
    points: int
    descriptor: str


class RubricCriterion(BaseModel):
    criterion: str
    description: str
    levels: list[RubricLevel]
    max_points: int


class Question(BaseModel):
    question_id: str
    text: str
    type: QuestionType
    difficulty: DifficultyLevel
    bloom_level: BloomLevel
    competency: str
    options: list[str] | None = None
    scenario_context: str | None = None
    answer_key: AnswerKey
    marks: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_options_and_context(self) -> "Question":
        if self.type in (
            QuestionType.mcq,
            QuestionType.multi_select,
            QuestionType.true_false,
        ):
            if not self.options or len(self.options) < 2 or len(self.options) > 6:
                raise ValueError(f"{self.type.value} requires 2 to 6 options")

            # Correct answer must be in options for mcq
            if self.type == QuestionType.mcq:
                correct = self.answer_key.correct_answer
                if isinstance(correct, str) and correct not in self.options:
                    raise ValueError("Correct answer must be present in options")

        if self.type in (QuestionType.scenario_based, QuestionType.case_study):
            if not self.scenario_context or len(self.scenario_context.split()) < 40:
                raise ValueError(
                    f"{self.type.value} requires scenario_context of at least 40 words"
                )
        return self


class DuplicatePair(BaseModel):
    q1: str
    q2: str
    similarity: float


class ValidationReport(BaseModel):
    quality_score: float = Field(ge=0, le=1)
    checks_passed: list[str]
    checks_failed: list[str]
    duplicate_pairs: list[DuplicatePair]
    approved: bool


class Assignment(BaseModel):
    assignment_id: str
    curriculum_id: str
    module_id: str
    title: str
    difficulty: DifficultyLevel
    instructions: str
    questions: list[Question]
    rubric: list[RubricCriterion]
    learning_outcomes: list[str]
    total_marks: int
    estimated_time_minutes: int
    validation_report: ValidationReport | None = None
    status: Literal["draft", "approved", "rejected"]

    @model_validator(mode="after")
    def validate_total_marks(self) -> "Assignment":
        sum_marks = sum(q.marks for q in self.questions)
        if self.total_marks != sum_marks:
            raise ValueError(
                f"total_marks ({self.total_marks}) must equal sum of question marks ({sum_marks})"
            )
        return self
