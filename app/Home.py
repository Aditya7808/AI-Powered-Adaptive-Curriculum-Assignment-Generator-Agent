import os
import sys

import streamlit as st

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.storage import db

st.set_page_config(
    page_title="AI Curriculum & Assignment Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #3d3d5c;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #aaa;
        margin-top: 0.25rem;
    }
    .status-badge-ok {
        background: #22c55e22;
        color: #22c55e;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-badge-warn {
        background: #eab30822;
        color: #eab308;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Initialize Database ---
db.init_db()

# --- Header ---
st.markdown(
    '<div class="main-header">AI-Powered Curriculum & Assignment Generator</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Upload PDFs → Generate Grounded Curricula → Create Validated Assignments → Assess & Adapt</div>',
    unsafe_allow_html=True,
)

# --- System Status Metrics ---
st.markdown("### System Dashboard")

try:
    docs = db.get_all_documents()
    total_docs = len(docs)
    ready_docs = sum(1 for d in docs if d.status == "ready")
    total_pages = sum(d.num_pages for d in docs)
    total_chunks = sum(d.num_chunks for d in docs)
    kb_version = db.get_kb_version()
except Exception:
    total_docs = 0
    ready_docs = 0
    total_pages = 0
    total_chunks = 0
    kb_version = 1

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Documents Ingested", total_docs)
with col2:
    st.metric("Ready Documents", ready_docs)
with col3:
    st.metric("Total Pages", total_pages)
with col4:
    st.metric("Total Chunks", total_chunks)
with col5:
    st.metric("KB Version", kb_version)

st.divider()

# --- Pipeline Status ---
st.markdown("### Pipeline Component Status")

llm_status = "Active" if config.LLM_PROVIDER else "Not Configured"
embedding_status = "Enabled: " + config.EMBEDDING_MODEL
reranker_status = "Enabled" if config.ENABLE_RERANKER else "Disabled"
bm25_status = "Enabled" if config.ENABLE_BM25 else "Disabled"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info(f"**LLM Provider**\n\n{config.LLM_PROVIDER.title()}")
with col2:
    st.info(f"**Embedding Model**\n\n{config.EMBEDDING_MODEL.split('/')[-1]}")
with col3:
    st.info(f"**Reranker**\n\n{'Enabled' if config.ENABLE_RERANKER else 'Disabled'}")
with col4:
    st.info(f"**BM25 Hybrid**\n\n{'Enabled' if config.ENABLE_BM25 else 'Disabled'}")

st.divider()

# --- Quick Navigation ---
st.markdown("### Quick Navigation")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.page_link("pages/1_Document_Management.py", label="Document Management")
    st.caption("Upload PDFs, manage knowledge base")

with col2:
    st.page_link("pages/2_Curriculum_Generator.py", label="Curriculum Generator")
    st.caption("Create personalized learning paths")

with col3:
    st.page_link("pages/3_Assignment_Studio.py", label="Assignment Studio")
    st.caption("Generate & validate assignments")

with col4:
    st.page_link("pages/4_Learner_Assessment.py", label="Learner Assessment")
    st.caption("Grade, analyze & adapt")

st.divider()

# --- Recent Documents ---
st.markdown("### Recent Documents")
if total_docs > 0:
    for doc in docs[:5]:
        status_text = (
            "Ready"
            if doc.status == "ready"
            else ("Processing" if doc.status == "processing" else "Failed")
        )
        st.markdown(
            f"- **{doc.file_name}** — {doc.num_pages} pages, {doc.num_chunks} chunks — *{status_text}*"
        )
else:
    st.info("No documents uploaded yet. Go to **Document Management** to upload PDFs.")

# --- Sidebar ---
with st.sidebar:
    st.markdown("### System Configuration")
    st.text(f"LLM Provider: {config.LLM_PROVIDER}")
    st.text(f"DB Path: {config.DB_PATH}")
    st.text(f"ChromaDB: {config.CHROMA_DIR}")
    st.text(f"Max Upload: {config.MAX_UPLOAD_MB} MB")
    st.divider()
    st.caption("AI Curriculum & Assignment Generator Agent v1.3")
