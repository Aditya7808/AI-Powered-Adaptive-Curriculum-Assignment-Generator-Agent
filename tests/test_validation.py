from src.schemas.assignment import (
    AnswerKey,
    Assignment,
    Question,
    RubricCriterion,
    RubricLevel,
    ValidationReport,
)
from src.schemas.learner import BloomLevel, DifficultyLevel, QuestionType
from src.validation.validator import (
    compute_text_similarity,
    validate_assignment,
)


def make_valid_assignment() -> Assignment:
    return Assignment(
        assignment_id="assign-val-1",
        curriculum_id="curr-1",
        module_id="mod-1",
        title="Intro to Data Structures",
        difficulty=DifficultyLevel.medium,
        instructions="Complete within 45 minutes.",
        questions=[
            Question(
                question_id="q1",
                text="Which data structure operates on a Last In First Out (LIFO) basis?",
                type=QuestionType.mcq,
                difficulty=DifficultyLevel.easy,
                bloom_level=BloomLevel.remember,
                competency="Data Structures",
                options=["Queue", "Stack", "Array", "Tree"],
                answer_key=AnswerKey(
                    correct_answer="Stack",
                    explanation="A stack operates using LIFO ordering.",
                    common_mistakes=["Choosing Queue"],
                ),
                marks=5,
            ),
            Question(
                question_id="q2",
                text="What is the average time complexity of searching in a balanced Binary Search Tree?",
                type=QuestionType.mcq,
                difficulty=DifficultyLevel.medium,
                bloom_level=BloomLevel.understand,
                competency="Algorithms",
                options=["O(1)", "O(log n)", "O(n)", "O(n log n)"],
                answer_key=AnswerKey(
                    correct_answer="O(log n)",
                    explanation="Halving the search space at each node results in O(log n) average time complexity.",
                    common_mistakes=["Choosing O(n)"],
                ),
                marks=5,
            ),
        ],
        rubric=[
            RubricCriterion(
                criterion="Accuracy",
                description="Correctness of responses",
                levels=[
                    RubricLevel(
                        label="Exemplary", points=5, descriptor="All answers correct"
                    ),
                    RubricLevel(
                        label="Developing", points=2, descriptor="Partial understanding"
                    ),
                ],
                max_points=5,
            )
        ],
        learning_outcomes=["Understand Stacks", "Understand BST Complexity"],
        total_marks=10,
        estimated_time_minutes=25,
        status="draft",
    )


def test_compute_text_similarity():
    s1 = "What is the time complexity of quicksort?"
    s2 = "What is the time complexity of quicksort?"
    assert compute_text_similarity(s1, s2) == 1.0

    s3 = "How does binary search work on an array?"
    sim = compute_text_similarity(s1, s3)
    assert 0.0 <= sim < 0.5


def test_validate_assignment_approval():
    assignment = make_valid_assignment()
    report = validate_assignment(assignment)

    assert isinstance(report, ValidationReport)
    assert report.approved is True
    assert report.quality_score >= 0.70
    assert len(report.duplicate_pairs) == 0
    assert assignment.status == "approved"
    assert "marks_consistency_check" in report.checks_passed
    assert "minimum_questions_check" in report.checks_passed
    assert "objective_options_integrity_check" in report.checks_passed


def test_validate_assignment_detects_duplicates():
    assignment = make_valid_assignment()
    # Add a duplicate question
    dup_question = Question(
        question_id="q3_dup",
        text="Which data structure operates on a Last In First Out (LIFO) basis?",
        type=QuestionType.mcq,
        difficulty=DifficultyLevel.easy,
        bloom_level=BloomLevel.remember,
        competency="Data Structures",
        options=["Queue", "Stack", "Array", "Tree"],
        answer_key=AnswerKey(
            correct_answer="Stack",
            explanation="A stack operates using LIFO ordering.",
            common_mistakes=["Choosing Queue"],
        ),
        marks=5,
    )
    assignment.questions.append(dup_question)
    assignment.total_marks = 15

    report = validate_assignment(assignment, duplicate_threshold=0.85)

    assert len(report.duplicate_pairs) > 0
    assert report.approved is False
    assert assignment.status == "rejected"
    assert any("duplicate_questions_check" in f for f in report.checks_failed)


def test_validate_assignment_missing_rubric():
    assignment = make_valid_assignment()
    assignment.rubric = []

    report = validate_assignment(assignment)
    assert any("rubric_completeness_check" in f for f in report.checks_failed)
