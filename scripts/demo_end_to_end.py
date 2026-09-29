from src.orchestrator import AdaptiveLearningOrchestrator
from src.schemas.assignment import DifficultyLevel
from src.schemas.learner import LearnerProfile, QuestionType


def run_demo():
    print("Starting End-to-End Demo...")

    orch = AdaptiveLearningOrchestrator()

    print("\n1. Onboarding Learner...")
    profile = LearnerProfile(
        name="Test Learner",
        background="Beginner in Python",
        current_skills={"Python": 1},
        prior_experience_years=0.5,
        available_hours_per_week=10,
        domain="Data Science",
    )

    print("\n2. Building Curriculum...")
    curr = orch.build_curriculum(profile, "Learn Python for Data Science")
    print(f"Generated curriculum: {curr.title} with {len(curr.modules)} modules.")

    print("\n3. Creating Assignment...")
    first_module = curr.modules[0].module_id
    assignment = orch.create_assignment(
        curriculum=curr,
        module_id=first_module,
        difficulty=DifficultyLevel.easy,
        question_mix={QuestionType.mcq: 2},
        learner=profile,
    )
    print(
        f"Generated assignment: {assignment.title} with {len(assignment.questions)} questions."
    )

    print("\n4. Analyzing Performance & Adapting...")
    report, recs, new_curr = orch.analyze_and_adapt(profile, [], [assignment], curr)

    print("Performance Report:", report.trend)
    if new_curr:
        print(f"Curriculum adapted to version {new_curr.version}.")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
