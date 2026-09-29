from src.llm.provider import generate_structured, get_llm
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


def retrieve_competency_context(
    competencies: list[str], retriever=None, top_k: int = 3
) -> str:
    """
    Retrieve grounded context from the knowledge base for a list of competencies.
    If no retriever is passed, returns an empty string or fallback note.
    """
    if not retriever or not competencies:
        return ""

    collected_snippets = []
    for comp in competencies:
        try:
            results = retriever.retrieve(comp, top_k=top_k)
            for res in results:
                # Handle DocumentChunk or dict
                if hasattr(res, "text"):
                    text = res.text
                    fname = getattr(res, "file_name", "unknown")
                    page = getattr(res, "page", 1)
                elif isinstance(res, dict):
                    text = res.get("text", "")
                    fname = res.get("file_name", "unknown")
                    page = res.get("page", 1)
                else:
                    text = str(res)
                    fname = "doc"
                    page = 1
                collected_snippets.append(f"[{fname} p.{page}] {text}")
        except Exception:
            continue

    return "\n\n".join(collected_snippets)


def generate_assignment(
    curriculum_id: str,
    module_id: str,
    title: str,
    competencies: list[str],
    difficulty: DifficultyLevel = DifficultyLevel.medium,
    learner_profile: LearnerProfile | None = None,
    context: str | None = None,
    retriever=None,
    llm=None,
) -> Assignment:
    """
    Generates a full Assignment with questions, answer keys, and grading rubrics.
    Grounded with context from RAG if a retriever or context is provided.
    """
    llm = llm or get_llm()

    rag_context = context or ""
    if not rag_context and retriever:
        rag_context = retrieve_competency_context(competencies, retriever=retriever)

    profile_info = ""
    if learner_profile:
        profile_info = (
            f"\nLearner Profile:\n{learner_profile.model_dump_json(indent=2)}"
        )

    prompt = f"""Generate a comprehensive, structured Assignment for the following module.

Curriculum ID: {curriculum_id}
Module ID: {module_id}
Module Title: {title}
Target Competencies: {", ".join(competencies)}
Difficulty Level: {difficulty.value}
{profile_info}

Grounded Reference Context:
{rag_context if rag_context else "None provided (use domain standard knowledge)"}

Requirements:
1. Provide clear instructions and learning outcomes.
2. Generate multi-level questions spanning the target competencies and appropriate Bloom levels.
3. For MCQ questions, provide 2 to 6 options and ensure the correct answer exactly matches one of the options.
4. For scenario_based or case_study questions, ensure the scenario_context has at least 40 words.
5. Provide a complete AnswerKey for each question with explanation and common mistakes.
6. Provide a grading rubric with RubricCriterion and RubricLevel entries.
7. CRITICAL: The assignment total_marks MUST exactly equal the sum of marks of all questions.
8. Status should be 'draft'.

Return a valid, fully populated Assignment object matching the schema."""

    assignment = generate_structured(prompt, Assignment, llm)

    # Ensure IDs and total_marks align with inputs if LLM used placeholders
    assignment.curriculum_id = curriculum_id
    assignment.module_id = module_id
    if not assignment.title:
        assignment.title = title
    assignment.difficulty = difficulty

    return assignment


def generate_targeted_questions(
    competency: str,
    question_type: QuestionType,
    difficulty: DifficultyLevel,
    bloom_level: BloomLevel,
    marks: int = 5,
    context: str | None = None,
    llm=None,
) -> Question:
    """
    Generate a single targeted question for a specific competency and question type.
    """
    llm = llm or get_llm()
    prompt = f"""Generate a single question for:
Competency: {competency}
Question Type: {question_type.value}
Difficulty: {difficulty.value}
Bloom Level: {bloom_level.value}
Marks: {marks}

Reference Context:
{context or "General curriculum standard"}

Requirements:
- If MCQ, provide 2 to 6 options and the correct answer must be one of the options.
- If scenario_based or case_study, scenario_context must contain at least 40 words.
- Provide explanation and common mistakes in the answer_key.

Return a valid Question object matching the schema."""

    return generate_structured(prompt, Question, llm)
