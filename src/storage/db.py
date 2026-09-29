import sqlite3
from datetime import datetime

from src.config import config
from src.schemas.documents import UploadedDocument


def get_connection():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                file_name TEXT,
                file_hash TEXT UNIQUE,
                num_pages INTEGER,
                num_chunks INTEGER,
                status TEXT,
                error_message TEXT,
                uploaded_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kb_state (
                id INTEGER PRIMARY KEY,
                kb_version INTEGER,
                embedding_model TEXT,
                updated_at TEXT
            )
        """)
        # Initialize kb_state if empty
        row = conn.execute("SELECT id FROM kb_state WHERE id=1").fetchone()
        if not row:
            conn.execute(
                "INSERT INTO kb_state (id, kb_version, embedding_model, updated_at) VALUES (1, 1, ?, ?)",
                (config.EMBEDDING_MODEL, datetime.utcnow().isoformat()),
            )


def bump_kb_version():
    with get_connection() as conn:
        conn.execute(
            "UPDATE kb_state SET kb_version = kb_version + 1, updated_at = ? WHERE id = 1",
            (datetime.utcnow().isoformat(),),
        )


def get_kb_version() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT kb_version FROM kb_state WHERE id = 1").fetchone()
        return row["kb_version"] if row else 1


def add_document(doc: UploadedDocument):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO documents (doc_id, file_name, file_hash, num_pages, num_chunks, status, error_message, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                doc.doc_id,
                doc.file_name,
                doc.file_hash,
                doc.num_pages,
                doc.num_chunks,
                doc.status,
                doc.error_message,
                doc.uploaded_at.isoformat(),
            ),
        )


def get_document_by_hash(file_hash: str) -> UploadedDocument | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        if row:
            return UploadedDocument(
                doc_id=row["doc_id"],
                file_name=row["file_name"],
                file_hash=row["file_hash"],
                num_pages=row["num_pages"],
                num_chunks=row["num_chunks"],
                status=row["status"],
                error_message=row["error_message"],
                uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
            )
    return None


def update_document(doc: UploadedDocument):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE documents 
            SET num_pages=?, num_chunks=?, status=?, error_message=?
            WHERE doc_id=?
        """,
            (doc.num_pages, doc.num_chunks, doc.status, doc.error_message, doc.doc_id),
        )


def get_all_documents() -> list[UploadedDocument]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM documents").fetchall()
        return [UploadedDocument(**dict(row)) for row in rows]


def delete_document(doc_id: str):
    with get_connection() as conn:
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
