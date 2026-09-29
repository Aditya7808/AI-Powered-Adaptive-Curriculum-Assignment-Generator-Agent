import json
import os
import sys

import streamlit as st

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.agents.curriculum_agent import generate_curriculum
from src.export.exporter import (
    export_curriculum_docx,
    export_curriculum_json,
    export_curriculum_markdown,
)
from src.schemas.learner import LearnerProfile, QuestionType
from src.storage import db

st.set_page_config(page_title="Curriculum Generator", layout="wide")
db.init_db()

st.markdown("# Curriculum Generator")
st.markdown(
    "Create personalized, competency-based curricula grounded in your uploaded knowledge base."
)

st.divider()

# --- Session State ---
if "learner_profile" not in st.session_state:
    st.session_state.learner_profile = None
if "curriculum" not in st.session_state:
    st.session_state.curriculum = None

# --- Tab Layout ---
tab1, tab2, tab3 = st.tabs(["Learner Profile", "Generate Curriculum", "Export"])

# ==========================
# TAB 1: Learner Profile
# ==========================
with tab1:
    st.markdown("### Build Learner Profile")

    with st.form("learner_form"):
        name = st.text_input("Learner Name", value="Alex Johnson")
        background = st.text_area(
            "Background",
            value="Computer Science undergraduate with interest in AI and data science.",
            height=80,
        )
        domain = st.text_input("Target Domain", value="Machine Learning & Data Science")

        col1, col2 = st.columns(2)
        with col1:
            experience = st.slider("Years of Experience", 0.0, 20.0, 1.0, 0.5)
            hours = st.slider("Available Hours/Week", 1, 60, 15)
        with col2:
            st.markdown("**Current Skills** (skill: level 1-5)")
            skills_text = st.text_area(
                "One per line (e.g., Python: 3)",
                value="Python: 3\nStatistics: 2\nSQL: 2",
                height=100,
            )

        preferred = st.multiselect(
            "Preferred Question Formats",
            ["mcq", "short_answer", "coding", "case_study", "scenario_based"],
            default=["mcq", "coding"],
        )

        submitted = st.form_submit_button("Create Profile", type="primary")

    if submitted:
        # Parse skills
        skills = {}
        for line in skills_text.strip().split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                try:
                    skills[k.strip()] = int(v.strip())
                except ValueError:
                    pass

        if not skills:
            st.error(
                "Please add at least one skill with a valid level (e.g., Python: 3)"
            )
        else:
            try:
                profile = LearnerProfile(
                    name=name,
                    background=background,
                    current_skills=skills,
                    prior_experience_years=experience,
                    available_hours_per_week=hours,
                    preferred_formats=[QuestionType(p) for p in preferred]
                    if preferred
                    else None,
                    domain=domain,
                )
                st.session_state.learner_profile = profile
                st.success(
                    f"Profile created for **{profile.name}** (ID: {profile.learner_id[:8]}...)"
                )

                with st.expander("View Profile JSON"):
                    st.json(json.loads(profile.model_dump_json()))
            except Exception as e:
                st.error(f"Profile creation failed: {e}")

# ==========================
# TAB 2: Generate Curriculum
# ==========================
with tab2:
    st.markdown("### Generate Personalized Curriculum")

    if not st.session_state.learner_profile:
        st.warning("⚠️ Please create a learner profile first (Tab 1).")
    else:
        profile = st.session_state.learner_profile
        st.info(
            f"**Active Learner**: {profile.name} | Domain: {profile.domain} | Skills: {', '.join(profile.current_skills.keys())}"
        )

        objectives = st.text_area(
            "Learning Objectives & Goals",
            value="Master foundational machine learning concepts including supervised learning, unsupervised learning, and model evaluation. Build practical skills in Python-based ML libraries.",
            height=120,
        )

        if st.button("Generate Curriculum", type="primary"):
            with st.spinner("Generating personalized curriculum using AI..."):
                try:
                    curriculum = generate_curriculum(profile, objectives)
                    st.session_state.curriculum = curriculum
                    st.success(
                        f"Curriculum **{curriculum.title}** generated with {len(curriculum.modules)} modules!"
                    )
                except Exception as e:
                    st.error(f"Curriculum generation failed: {e}")

        if st.session_state.curriculum:
            curriculum = st.session_state.curriculum
            st.divider()
            st.markdown(f"## {curriculum.title}")
            st.caption(
                f"Version {curriculum.version} | {len(curriculum.modules)} modules | Created {curriculum.created_at.strftime('%Y-%m-%d')}"
            )

            for i, mod in enumerate(curriculum.modules, start=1):
                with st.expander(
                    f"Module {i}: {mod.title} ({mod.week_range})", expanded=(i == 1)
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Difficulty**: {mod.difficulty.value.title()}")
                        st.markdown("**Competencies:**")
                        for c in mod.competencies:
                            st.markdown(f"  - {c}")
                    with col2:
                        st.markdown("**Learning Outcomes:**")
                        for lo in mod.learning_outcomes:
                            st.markdown(f"  - {lo}")

                    st.markdown("**Topics:**")
                    for t in mod.topics:
                        st.markdown(f"  - {t}")

                    st.markdown("**Activities:**")
                    for a in mod.activities:
                        st.markdown(f"  - {a}")

                    if mod.resources:
                        st.markdown("**Referenced Resources:**")
                        for r in mod.resources:
                            st.markdown(f"  - [{r.file_name} p.{r.page}] {r.title}")

            with st.expander("Alignment Notes"):
                st.markdown(curriculum.alignment_notes)

# ==========================
# TAB 3: Export
# ==========================
with tab3:
    st.markdown("### Export Curriculum")

    if not st.session_state.curriculum:
        st.warning("⚠️ Generate a curriculum first (Tab 2).")
    else:
        curriculum = st.session_state.curriculum

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Markdown")
            md_content = export_curriculum_markdown(curriculum)
            st.download_button(
                "Download Markdown",
                md_content,
                file_name=f"curriculum_{curriculum.curriculum_id}.md",
                mime="text/markdown",
            )

        with col2:
            st.markdown("#### JSON")
            json_content = export_curriculum_json(curriculum)
            st.download_button(
                "Download JSON",
                json_content,
                file_name=f"curriculum_{curriculum.curriculum_id}.json",
                mime="application/json",
            )

        with col3:
            st.markdown("#### Word (.docx)")
            docx_path = os.path.join(
                "data", f"curriculum_{curriculum.curriculum_id}.docx"
            )
            os.makedirs("data", exist_ok=True)
            export_curriculum_docx(curriculum, docx_path)
            with open(docx_path, "rb") as f:
                st.download_button(
                    "Download DOCX",
                    f.read(),
                    file_name=f"curriculum_{curriculum.curriculum_id}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )

        st.divider()
        with st.expander("Preview Markdown"):
            st.markdown(md_content)
