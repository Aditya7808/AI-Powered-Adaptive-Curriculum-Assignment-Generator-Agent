import os
import sys

import streamlit as st

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.config import config
from src.documents.ingestion import DocumentIngestionService
from src.documents.vectorstore import VectorStore
from src.retrieval.embeddings import EmbeddingService
from src.storage import db

st.set_page_config(page_title="Document Management", layout="wide")

db.init_db()

st.markdown("# Document Management")
st.markdown(
    "Upload PDFs to build the knowledge base that grounds your curricula and assignments."
)

st.divider()

# --- Upload Section ---
st.markdown("### Upload New Documents")

uploaded_files = st.file_uploader(
    "Choose PDF files to upload",
    type=["pdf"],
    accept_multiple_files=True,
    help=f"Max {config.MAX_UPLOAD_MB} MB per file, up to {config.MAX_FILES_PER_UPLOAD} files at once.",
)

if uploaded_files:
    if len(uploaded_files) > config.MAX_FILES_PER_UPLOAD:
        st.error(
            f"⚠️ Maximum {config.MAX_FILES_PER_UPLOAD} files per upload. You selected {len(uploaded_files)}."
        )
    else:
        if st.button("Ingest Selected Documents", type="primary"):
            os.makedirs(config.UPLOAD_DIR, exist_ok=True)

            with st.spinner("Initializing embedding model and vector store..."):
                try:
                    emb = EmbeddingService()
                    vs = VectorStore()
                    service = DocumentIngestionService(emb, vs)
                except Exception as e:
                    st.error(f"Failed to initialize services: {e}")
                    st.stop()

            progress_bar = st.progress(0)
            results = []

            for i, uploaded_file in enumerate(uploaded_files):
                # Size check
                file_size_mb = uploaded_file.size / (1024 * 1024)
                if file_size_mb > config.MAX_UPLOAD_MB:
                    results.append(
                        (
                            uploaded_file.name,
                            "rejected",
                            f"File too large ({file_size_mb:.1f} MB)",
                        )
                    )
                    continue

                # Save to upload dir
                save_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    try:
                        doc, is_new = service.ingest(save_path)
                        if is_new:
                            results.append(
                                (
                                    uploaded_file.name,
                                    doc.status,
                                    f"{doc.num_pages} pages, {doc.num_chunks} chunks",
                                )
                            )
                        else:
                            results.append(
                                (
                                    uploaded_file.name,
                                    "duplicate",
                                    "Already in knowledge base",
                                )
                            )
                    except Exception as e:
                        results.append((uploaded_file.name, "failed", str(e)))

                progress_bar.progress((i + 1) / len(uploaded_files))

            st.divider()
            st.markdown("### Ingestion Results")
            for name, status, detail in results:
                if status == "ready":
                    st.success(f"Success: **{name}** — {detail}")
                elif status == "duplicate":
                    st.info(f"Duplicate: **{name}** — {detail}")
                elif status == "no_text_found":
                    st.warning(
                        f"Warning: **{name}** — Scanned/image PDF, no extractable text"
                    )
                else:
                    st.error(f"Error: **{name}** — {detail}")

st.divider()

# --- Existing Documents ---
st.markdown("### Knowledge Base Documents")

docs = db.get_all_documents()

if not docs:
    st.info("No documents in the knowledge base yet. Upload PDFs above to get started.")
else:
    # Summary row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", len(docs))
    with col2:
        st.metric("Total Pages", sum(d.num_pages for d in docs))
    with col3:
        st.metric("KB Version", db.get_kb_version())

    st.divider()

    for doc in docs:
        status_text = {
            "ready": "Ready",
            "processing": "Processing",
            "failed": "Failed",
            "no_text_found": "No Text",
        }.get(doc.status, "Unknown")

        with st.expander(
            f"{status_text}: {doc.file_name} — {doc.num_pages} pages, {doc.num_chunks} chunks"
        ):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.text(f"Doc ID: {doc.doc_id[:12]}...")
            with col2:
                st.text(f"Status: {doc.status}")
            with col3:
                st.text(f"Pages: {doc.num_pages}")
            with col4:
                st.text(f"Chunks: {doc.num_chunks}")

            if doc.error_message:
                st.warning(f"Error: {doc.error_message}")

            if st.button(f"Delete {doc.file_name}", key=f"del_{doc.doc_id}"):
                try:
                    vs = VectorStore()
                    service = DocumentIngestionService(EmbeddingService(), vs)
                    service.delete(doc.doc_id)
                    st.success(f"Deleted {doc.file_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete: {e}")
