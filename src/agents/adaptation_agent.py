from datetime import datetime, timezone

from src.llm.provider import generate_structured, get_llm
from src.schemas.curriculum import Curriculum, Module
from src.schemas.learner import DifficultyLevel
from src.schemas.performance import AdjustmentRecommendation, PerformanceReport


def generate_remedial_module(competency: str, module_index: int, llm=None) -> Module:
    """
    Generates a structured remedial module addressing a specific weak competency.
    """
    llm = llm or get_llm()
    prompt = f"""Generate a focused remedial module to help a learner master the following competency.

Competency: {competency}
Target Module Index: {module_index}

Requirements:
- title: Remedial: Mastering {competency}
- week_range: e.g. "Week {module_index} (Remedial)"
- difficulty: easy or medium
- Clear topics, activities, and foundational learning outcomes.
- resources: can be empty list or standard references.

Return a valid Module object matching the schema."""

    try:
        return generate_structured(prompt, Module, llm)
    except Exception:
        # Fallback deterministic remedial module
        return Module(
            module_id=f"mod-remedial-{competency.lower().replace(' ', '-')[:10]}",
            title=f"Remedial Reinforcement: {competency}",
            week_range=f"Week {module_index} (Remedial)",
            competencies=[competency],
            learning_outcomes=[
                f"Reinforce fundamental concepts of {competency}",
                f"Resolve common misconceptions in {competency}",
            ],
            topics=[f"{competency} Fundamentals", f"{competency} Practice Problems"],
            activities=[
                f"Guided review of {competency}",
                "Hands-on diagnostic exercises",
            ],
            resources=[],
            prerequisites=[],
            difficulty=DifficultyLevel.easy,
        )


def adapt_curriculum(
    curriculum: Curriculum,
    report: PerformanceReport,
    recommendations: list[AdjustmentRecommendation],
    llm=None,
) -> Curriculum:
    """
    Evolves a curriculum based on performance report and adaptation recommendations:
    - Increments curriculum version.
    - Applies remedial insertions, difficulty scaling, or timeline extensions.
    - Preserves curriculum invariants (>= 3 modules).
    - Logs modifications in alignment_notes.
    """
    llm = llm or get_llm()
    adapted = curriculum.model_copy(deep=True)
    adapted.version += 1
    adapted.created_at = datetime.now(timezone.utc)

    mod_notes = []

    for rec in recommendations:
        if rec.target not in ("curriculum", "next_assignment"):
            continue

        if rec.action == "add_remedial_module":
            # Generate and insert remedial module
            new_idx = len(adapted.modules) + 1
            remedial_mod = generate_remedial_module(rec.competency, new_idx, llm=llm)
            adapted.modules.append(remedial_mod)
            mod_notes.append(
                f"Added remedial module for '{rec.competency}': {rec.reason}"
            )

        elif rec.action == "increase_difficulty":
            # Upgrade subsequent modules if not already hard
            diff_updated = False
            for mod in adapted.modules:
                if mod.difficulty == DifficultyLevel.easy:
                    mod.difficulty = DifficultyLevel.medium
                    diff_updated = True
                elif mod.difficulty == DifficultyLevel.medium:
                    mod.difficulty = DifficultyLevel.hard
                    diff_updated = True
            if diff_updated:
                mod_notes.append(
                    f"Increased difficulty for upcoming modules: {rec.reason}"
                )

        elif rec.action == "decrease_difficulty":
            # Downgrade modules if too difficult
            diff_updated = False
            for mod in adapted.modules:
                if mod.difficulty == DifficultyLevel.hard:
                    mod.difficulty = DifficultyLevel.medium
                    diff_updated = True
                elif mod.difficulty == DifficultyLevel.medium:
                    mod.difficulty = DifficultyLevel.easy
                    diff_updated = True
            if diff_updated:
                mod_notes.append(
                    f"Decreased difficulty for upcoming modules: {rec.reason}"
                )

        elif rec.action == "skip_or_compress_module":
            # Can compress if more than 3 modules
            if len(adapted.modules) > 3:
                # Find module matching competency and remove/condense
                removed = False
                for i, mod in enumerate(adapted.modules):
                    if any(
                        rec.competency.lower() in c.lower() for c in mod.competencies
                    ):
                        del adapted.modules[i]
                        removed = True
                        break
                if removed:
                    mod_notes.append(
                        f"Compressed mastered module for '{rec.competency}': {rec.reason}"
                    )
            else:
                mod_notes.append(
                    f"Retained module count (minimum 3 required) for '{rec.competency}'"
                )

        elif rec.action == "extend_timeline":
            # Extend module week ranges
            for i, mod in enumerate(adapted.modules):
                mod.week_range = f"Week {i * 2 + 1}-{i * 2 + 2} (Extended)"
            mod_notes.append(f"Extended module timelines: {rec.reason}")

    # Re-index week ranges if modules were added
    for i, mod in enumerate(adapted.modules, start=1):
        if "(Remedial)" not in mod.week_range and "(Extended)" not in mod.week_range:
            mod.week_range = f"Week {i}"

    adaptation_summary = (
        "; ".join(mod_notes)
        if mod_notes
        else "Performance reviewed; no structure changes required."
    )
    adapted.alignment_notes = f"{curriculum.alignment_notes}\n[v{adapted.version} Update]: {adaptation_summary}".strip()

    return adapted
