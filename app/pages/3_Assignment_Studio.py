import os
import sys

import streamlit as st

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.agents.assignment_agent import generate_assignment
from src.export.exporter import export_assignment_markdown
from src.schemas.assignment import DifficultyLevel, QuestionType

st.set_page_config(page_title="Assignment Studio", layout="wide")

st.markdown("# Assignment Studio")
st.markdown("Create, validate, and export assignments tailored to curriculum modules.")

st.divider()

if "curriculum" not in st.session_state or st.session_state.curriculum is None:
    st.warning("Please generate a curriculum first on the Curriculum Generator page.")
    st.stop()

curriculum = st.session_state.curriculum
st.info(f"Active Curriculum: {curriculum.title}")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Configuration")
    module_options = {mod.module_id: mod.title for mod in curriculum.modules}
    selected_module_id = st.selectbox(
        "Select Module",
        options=list(module_options.keys()),
        format_func=lambda x: module_options[x],
    )

    difficulty_mapping = {
        "Easy": DifficultyLevel.easy,
        "Medium": DifficultyLevel.medium,
        "Hard": DifficultyLevel.hard,
    }
    selected_difficulty_str = st.selectbox(
        "Difficulty", options=list(difficulty_mapping.keys())
    )
    selected_difficulty = difficulty_mapping[selected_difficulty_str]

with col2:
    st.markdown("### Question Mix")
    st.markdown("Specify the number of questions per type.")

    mcq_count = st.number_input("Multiple Choice", min_value=0, max_value=10, value=2)
    short_ans_count = st.number_input(
        "Short Answer", min_value=0, max_value=10, value=1
    )
    scenario_count = st.number_input(
        "Scenario Based", min_value=0, max_value=5, value=1
    )

    question_mix = {}
    if mcq_count > 0:
        question_mix[QuestionType.mcq] = mcq_count
    if short_ans_count > 0:
        question_mix[QuestionType.short_answer] = short_ans_count
    if scenario_count > 0:
        question_mix[QuestionType.scenario_based] = scenario_count

if st.button("Generate Assignment", type="primary"):
    total_q = sum(question_mix.values())
    if total_q == 0:
        st.error("Please request at least one question.")
    else:
        with st.spinner("Generating and validating assignment..."):
            selected_module = next(
                (m for m in curriculum.modules if m.module_id == selected_module_id),
                None,
            )

            if selected_module:
                try:
                    assignment = generate_assignment(
                        module=selected_module,
                        difficulty=selected_difficulty,
                        question_mix=question_mix,
                        learner=st.session_state.learner_profile,
                    )

                    st.session_state.current_assignment = assignment
                    st.success("Assignment generated successfully.")
                except Exception as e:
                    st.error(f"Failed to generate assignment: {e}")

if "current_assignment" in st.session_state and st.session_state.current_assignment:
    assignment = st.session_state.current_assignment

    st.divider()
    st.markdown(f"## {assignment.title}")

    if assignment.validation_report:
        vr = assignment.validation_report
        st.markdown(
            f"**Validation Score**: {vr.quality_score:.2f} | **Status**: {'Approved' if vr.approved else 'Rejected'}"
        )
        if not vr.approved:
            st.warning("Validation checks failed. Review required.")

    tab_q, tab_a, tab_r, tab_e = st.tabs(
        ["Questions", "Answer Key", "Rubric", "Export"]
    )

    with tab_q:
        for idx, q in enumerate(assignment.questions, 1):
            st.markdown(f"**Q{idx}. {q.text}** ({q.marks} marks)")
            if q.options:
                for opt in q.options:
                    st.markdown(f"- {opt}")
            if q.scenario_context:
                st.markdown(f"> Context: {q.scenario_context}")
            st.markdown("---")

    with tab_a:
        for idx, q in enumerate(assignment.questions, 1):
            st.markdown(f"**Q{idx}. Answer Key**")
            st.markdown(f"**Correct Answer**: {q.answer_key.correct_answer}")
            st.markdown(f"**Explanation**: {q.answer_key.explanation}")
            st.markdown("---")

    with tab_r:
        if assignment.rubric:
            for crit in assignment.rubric:
                st.markdown(f"**{crit.criterion}** (Max Points: {crit.max_points})")
                st.markdown(f"*{crit.description}*")
                for level in crit.levels:
                    st.markdown(
                        f"- **{level.label}** ({level.points} pts): {level.descriptor}"
                    )
                st.markdown("---")
        else:
            st.info("No rubric generated (likely MCQ only).")

    with tab_e:
        st.markdown("Download Assignment")

        md_student = export_assignment_markdown(assignment, include_answers=False)
        st.download_button(
            "Download Student Version (Markdown)",
            md_student,
            file_name=f"assignment_{assignment.assignment_id}_student.md",
        )

        md_instructor = export_assignment_markdown(assignment, include_answers=True)
        st.download_button(
            "Download Instructor Version (Markdown)",
            md_instructor,
            file_name=f"assignment_{assignment.assignment_id}_instructor.md",
        )
