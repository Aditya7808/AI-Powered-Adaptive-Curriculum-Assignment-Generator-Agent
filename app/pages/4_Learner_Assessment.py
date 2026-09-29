import os
import sys
from datetime import datetime

import streamlit as st

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.agents.adaptation_agent import adapt_curriculum
from src.agents.evaluation_agent import analyze_performance
from src.schemas.performance import QuestionResult, SubmissionResult

st.set_page_config(page_title="Learner Assessment", layout="wide")

st.markdown("# Learner Assessment")
st.markdown(
    "Record assignment scores, analyze performance trends, and adapt the curriculum."
)
st.divider()

if (
    "current_assignment" not in st.session_state
    or st.session_state.current_assignment is None
):
    st.warning("Please generate an assignment in the Assignment Studio first.")
    st.stop()

if (
    "learner_profile" not in st.session_state
    or st.session_state.learner_profile is None
):
    st.warning("Please create a learner profile first.")
    st.stop()

assignment = st.session_state.current_assignment
learner = st.session_state.learner_profile
curriculum = st.session_state.curriculum

st.markdown(f"### Record Scores for: {assignment.title}")

with st.form("score_form"):
    st.markdown("Enter marks awarded for each question.")

    score_data = {}
    for q in assignment.questions:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Q. {q.text}** (Max: {q.marks})")
        with col2:
            score_data[q.question_id] = st.number_input(
                f"Marks for Q{q.question_id[:4]}",
                min_value=0.0,
                max_value=float(q.marks),
                value=float(q.marks),
                key=q.question_id,
            )

    submitted = st.form_submit_button("Submit Assessment")

if submitted:
    results = []
    for q in assignment.questions:
        results.append(
            QuestionResult(
                question_id=q.question_id,
                marks_awarded=score_data[q.question_id],
                max_marks=float(q.marks),
            )
        )

    submission = SubmissionResult(
        submission_id=f"sub_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        learner_id=learner.learner_id,
        assignment_id=assignment.assignment_id,
        results=results,
        submitted_at=datetime.utcnow(),
    )

    if "submissions" not in st.session_state:
        st.session_state.submissions = []
    st.session_state.submissions.append(submission)

    st.success("Scores recorded successfully!")

if "submissions" in st.session_state and st.session_state.submissions:
    st.divider()
    st.markdown("### Performance Analytics")

    if st.button("Analyze Performance & Get Recommendations"):
        with st.spinner("Analyzing data..."):
            report, recommendations = analyze_performance(
                learner_id=learner.learner_id,
                submissions=st.session_state.submissions,
                assignments=[assignment],  # In a full system, fetch all from DB
                curriculum=curriculum,
            )
            st.session_state.performance_report = report
            st.session_state.recommendations = recommendations

if "performance_report" in st.session_state:
    report = st.session_state.performance_report
    recommendations = st.session_state.recommendations

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Score", f"{report.overall_score_pct:.1f}%")
    with col2:
        st.metric("Performance Trend", report.trend.title())
    with col3:
        st.metric(
            "Strongest Competency",
            max(report.score_by_competency, key=report.score_by_competency.get)
            if report.score_by_competency
            else "N/A",
        )

    st.markdown("#### Recommendations")
    for rec in recommendations:
        st.info(
            f"**{rec.action}** for **{rec.target}** (Competency: {rec.competency})\n\nReason: {rec.reason}"
        )

    if st.button("Apply Recommendations to Curriculum", type="primary"):
        with st.spinner("Adapting curriculum..."):
            try:
                new_curr = adapt_curriculum(curriculum, learner, recommendations)
                st.session_state.curriculum = new_curr
                st.success(
                    f"Curriculum adapted to version {new_curr.version}! Check the Curriculum page."
                )
            except Exception as e:
                st.error(f"Adaptation failed: {e}")
