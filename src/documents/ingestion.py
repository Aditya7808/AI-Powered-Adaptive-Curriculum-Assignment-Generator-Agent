import hashlib
import os
import uuid
from datetime import datetime

from src.documents.chunker import DocumentChunker
from src.documents.loader import PDFLoader
from src.documents.vectorstore import VectorStore
from src.retrieval.embeddings import EmbeddingService
from src.schemas.documents import UploadedDocument
from src.storage import db


class DocumentIngestionService:
    def __init__(self, embedding_service: EmbeddingService, vectorstore: VectorStore):
        self.embedding_service = embedding_service
        self.vectorstore = vectorstore
        self.chunker = DocumentChunker()

    def _compute_hash(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def ingest(self, file_path: str) -> tuple[UploadedDocument, bool]:
        """Ingests a document. Returns (doc, is_new)."""
        file_hash = self._compute_hash(file_path)
        file_name = os.path.basename(file_path)

        # Check if hash already exists
        existing_doc = db.get_document_by_hash(file_hash)
        if existing_doc:
            return existing_doc, False

        doc_id = str(uuid.uuid4())
        doc = UploadedDocument(
            doc_id=doc_id,
            file_name=file_name,
            file_hash=file_hash,
            num_pages=0,
            num_chunks=0,
            status="processing",
            uploaded_at=datetime.utcnow(),
        )
        db.add_document(doc)

        try:
            loader = PDFLoader(file_path)
            pages = loader.load()
            doc.num_pages = len(pages)

            chunks = self.chunker.chunk_pages(pages, doc_id, file_name)
            doc.num_chunks = len(chunks)

            if len(chunks) == 0:
                doc.status = "no_text_found"
                db.update_document(doc)
                return doc, True

            # Embed
            texts_to_embed = [
                f"{file_name} | page {c['page']}\n{c['text']}" for c in chunks
            ]
            embeddings = self.embedding_service.embed_documents(texts_to_embed)

            # Upsert to vectorstore
            ids = [c["chunk_id"] for c in chunks]
            documents = [c["text"] for c in chunks]
            metadatas = [
                {
                    "doc_id": c["doc_id"],
                    "file_name": c["file_name"],
                    "page": c["page"],
                    "chunk_index": c["chunk_index"],
                }
                for c in chunks
            ]

            self.vectorstore.upsert_chunks(ids, embeddings, documents, metadatas)

            doc.status = "ready"
            db.update_document(doc)
            db.bump_kb_version()

        except Exception as e:
            doc.status = "failed"
            doc.error_message = str(e)
            db.update_document(doc)
            if "Scanned or image-only PDF detected" in str(e):
                doc.status = "no_text_found"
                db.update_document(doc)

        return doc, True

    def delete(self, doc_id: str):
        self.vectorstore.delete_by_doc(doc_id)
        db.delete_document(doc_id)
        db.bump_kb_version()
