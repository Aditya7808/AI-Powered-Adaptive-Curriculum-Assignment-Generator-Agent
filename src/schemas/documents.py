from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class UploadedDocument(BaseModel):
    doc_id: str
    file_name: str
    file_hash: str
    num_pages: int
    num_chunks: int
    status: Literal["processing", "ready", "failed", "no_text_found"]
    uploaded_at: datetime
    error_message: str | None = None
