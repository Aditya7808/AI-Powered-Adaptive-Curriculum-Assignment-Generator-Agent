from src.llm.provider import generate_structured, get_llm
from src.schemas.curriculum import Curriculum
from src.schemas.learner import LearnerProfile


def analyze_profile(raw_text: str, constraints: str = "") -> LearnerProfile:
    llm = get_llm()
    prompt = f"""Analyze the following raw text about a learner and extract their profile.

Raw Text:
{raw_text}

Constraints or additional info:
{constraints}

Return a structured LearnerProfile."""
    return generate_structured(prompt, LearnerProfile, llm)


def generate_curriculum(
    profile: LearnerProfile, learning_objectives: str
) -> Curriculum:
    llm = get_llm()
    prompt = f"""Generate a personalized curriculum based on the learner's profile and the learning objectives.

Learner Profile:
{profile.model_dump_json(indent=2)}

Learning Objectives:
{learning_objectives}

Return a structured Curriculum that fits the learner's background, learning style, and goals."""
    return generate_structured(prompt, Curriculum, llm)
