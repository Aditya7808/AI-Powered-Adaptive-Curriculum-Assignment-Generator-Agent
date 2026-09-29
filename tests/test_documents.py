import os
import shutil

import fitz
import pytest

from src.config import config
from src.documents.chunker import DocumentChunker
from src.documents.ingestion import DocumentIngestionService
from src.documents.loader import PDFLoader
from src.documents.vectorstore import VectorStore
from src.retrieval.embeddings import FakeEmbeddings
from src.storage import db


@pytest.fixture(scope="module")
def setup_db_and_paths():
    # Make sure we use an ephemeral chromadb by patching it or passing use_ephemeral=True
    os.makedirs("data/sample_pdfs", exist_ok=True)
    os.makedirs("data/chroma_test", exist_ok=True)
    config.DB_PATH = "data/app_test.db"

    # Init DB
    db.init_db()

    # Create a real dummy PDF using pymupdf
    pdf_path = "data/sample_pdfs/dummy.pdf"
    doc = fitz.open()
    page = doc.new_page()
    long_text = "This is a dummy PDF file for testing. It has some text. " * 5
    page.insert_text(fitz.Point(100, 100), long_text)
    page2 = doc.new_page()
    page2.insert_text(fitz.Point(100, 100), "Page 2 text. " * 5)
    doc.save(pdf_path)
    doc.close()

    yield pdf_path

    # Teardown
    if os.path.exists("data/app_test.db"):
        try:
            os.remove("data/app_test.db")
        except PermissionError:
            pass
    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except PermissionError:
            pass
    if os.path.exists("data/chroma_test"):
        shutil.rmtree("data/chroma_test", ignore_errors=True)


def test_loader(setup_db_and_paths):
    pdf_path = setup_db_and_paths
    loader = PDFLoader(pdf_path)
    pages = loader.load()
    assert len(pages) == 2
    assert "dummy PDF" in pages[0]["text"]
    assert "Page 2" in pages[1]["text"]


def test_chunker(setup_db_and_paths):
    pdf_path = setup_db_and_paths
    loader = PDFLoader(pdf_path)
    pages = loader.load()
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_pages(pages, "doc1", "dummy.pdf")

    assert len(chunks) > 0
    assert chunks[0]["page"] == 1
    assert "dummy.pdf" in chunks[0]["file_name"]
    assert "chunk_id" in chunks[0]


def test_ingestion_and_vectorstore(setup_db_and_paths):
    pdf_path = setup_db_and_paths
    emb = FakeEmbeddings()
    vs = VectorStore(use_ephemeral=True)

    service = DocumentIngestionService(emb, vs)

    initial_kb = db.get_kb_version()

    # Ingest
    doc, is_new = service.ingest(pdf_path)
    assert is_new
    assert doc.status == "ready"
    assert doc.num_pages == 2

    new_kb = db.get_kb_version()
    assert new_kb > initial_kb

    # Vector store count
    assert vs.count() > 0
    assert vs.count(doc.doc_id) > 0

    # Idempotent (same hash)
    doc2, is_new2 = service.ingest(pdf_path)
    assert not is_new2
    assert doc.doc_id == doc2.doc_id

    # Delete removes chunks
    service.delete(doc.doc_id)
    assert vs.count(doc.doc_id) == 0
    assert db.get_kb_version() > new_kb
