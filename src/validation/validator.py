import numpy as np

from src.schemas.assignment import Assignment, DuplicatePair, Question, ValidationReport
from src.schemas.learner import QuestionType


def compute_text_similarity(t1: str, t2: str, embeddings=None) -> float:
    """
    Computes similarity between two question texts.
    Uses embeddings cosine similarity if provided, else falls back to word Jaccard similarity.
    """
    s1, s2 = t1.strip(), t2.strip()
    if s1.lower() == s2.lower():
        return 1.0

    if embeddings:
        try:
            vecs = embeddings.embed_documents([s1, s2])
            v1, v2 = np.array(vecs[0]), np.array(vecs[1])
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom > 0:
                return float(np.dot(v1, v2) / denom)
        except Exception:
            pass

    # Jaccard word-set similarity fallback
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())
    if not words1 or not words2:
        return 0.0
    return float(len(words1 & words2) / len(words1 | words2))


def find_duplicate_questions(
    questions: list[Question], embeddings=None, threshold: float = 0.85
) -> list[DuplicatePair]:
    """
    Detects pairs of questions whose text similarity meets or exceeds the threshold.
    """
    duplicates: list[DuplicatePair] = []
    n = len(questions)

    for i in range(n):
        for j in range(i + 1, n):
            q1 = questions[i]
            q2 = questions[j]
            sim = compute_text_similarity(q1.text, q2.text, embeddings=embeddings)
            if sim >= threshold:
                duplicates.append(
                    DuplicatePair(
                        q1=q1.question_id, q2=q2.question_id, similarity=round(sim, 3)
                    )
                )
    return duplicates


def validate_assignment(
    assignment: Assignment,
    embeddings=None,
    duplicate_threshold: float = 0.85,
    quality_threshold: float = 0.70,
) -> ValidationReport:
    """
    Runs multi-point validation on an Assignment:
    1. Total marks match sum of question marks.
    2. Duplicate question check.
    3. Option integrity for MCQs (correct answer included in options).
    4. Scenario context minimum length.
    5. Answer key explanations present and non-empty.
    6. Rubric presence and completeness.
    7. Question count adequacy (at least 2 questions).
    """
    checks_passed: list[str] = []
    checks_failed: list[str] = []

    # Check 1: Question Count
    if len(assignment.questions) >= 2:
        checks_passed.append("minimum_questions_check")
    else:
        checks_failed.append(
            "minimum_questions_check: Assignment has fewer than 2 questions"
        )

    # Check 2: Total Marks Consistency
    sum_marks = sum(q.marks for q in assignment.questions)
    if assignment.total_marks == sum_marks and sum_marks > 0:
        checks_passed.append("marks_consistency_check")
    else:
        checks_failed.append(
            f"marks_consistency_check: total_marks ({assignment.total_marks}) != sum of question marks ({sum_marks})"
        )

    # Check 3: Answer Key Explanations
    all_answers_explained = all(
        bool(q.answer_key.explanation and len(q.answer_key.explanation.strip()) > 5)
        for q in assignment.questions
    )
    if all_answers_explained:
        checks_passed.append("answer_key_explanations_check")
    else:
        checks_failed.append(
            "answer_key_explanations_check: One or more questions lack adequate explanations in answer_key"
        )

    # Check 4: MCQ and Multi-Select Options Integrity
    mcq_integrity = True
    for q in assignment.questions:
        if q.type in (
            QuestionType.mcq,
            QuestionType.multi_select,
            QuestionType.true_false,
        ):
            if not q.options or len(q.options) < 2:
                mcq_integrity = False
                break
            if (
                q.type == QuestionType.mcq
                and q.answer_key.correct_answer not in q.options
            ):
                mcq_integrity = False
                break
    if mcq_integrity:
        checks_passed.append("objective_options_integrity_check")
    else:
        checks_failed.append(
            "objective_options_integrity_check: Invalid options or missing correct answer in options for objective question"
        )

    # Check 5: Scenario Context Length
    scenario_ok = True
    for q in assignment.questions:
        if q.type in (QuestionType.scenario_based, QuestionType.case_study):
            if not q.scenario_context or len(q.scenario_context.split()) < 40:
                scenario_ok = False
                break
    if scenario_ok:
        checks_passed.append("scenario_context_length_check")
    else:
        checks_failed.append(
            "scenario_context_length_check: Scenario or case study question has context fewer than 40 words"
        )

    # Check 6: Rubric Completeness
    rubric_complete = False
    if assignment.rubric and len(assignment.rubric) > 0:
        rubric_complete = all(
            bool(r.criterion and r.levels and len(r.levels) >= 2 and r.max_points > 0)
            for r in assignment.rubric
        )
    if rubric_complete:
        checks_passed.append("rubric_completeness_check")
    else:
        checks_failed.append(
            "rubric_completeness_check: Rubric criteria missing or insufficient levels"
        )

    # Check 7: Duplicate Questions Check
    duplicate_pairs = find_duplicate_questions(
        assignment.questions, embeddings=embeddings, threshold=duplicate_threshold
    )
    if not duplicate_pairs:
        checks_passed.append("duplicate_questions_check")
    else:
        checks_failed.append(
            f"duplicate_questions_check: Found {len(duplicate_pairs)} duplicate or highly similar question pairs"
        )

    # Calculate Quality Score
    total_checks = len(checks_passed) + len(checks_failed)
    raw_score = (len(checks_passed) / total_checks) if total_checks > 0 else 0.0

    # Penalize if duplicates found
    if duplicate_pairs:
        raw_score = max(0.0, raw_score - 0.25 * len(duplicate_pairs))

    quality_score = round(min(max(raw_score, 0.0), 1.0), 2)
    approved = (
        (quality_score >= quality_threshold)
        and (len(duplicate_pairs) == 0)
        and ("marks_consistency_check" in checks_passed)
    )

    report = ValidationReport(
        quality_score=quality_score,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        duplicate_pairs=duplicate_pairs,
        approved=approved,
    )

    # Update assignment
    assignment.validation_report = report
    assignment.status = "approved" if approved else "rejected"

    return report
