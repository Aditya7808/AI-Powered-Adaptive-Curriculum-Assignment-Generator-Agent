import re
from typing import Any

import fitz


class PDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[dict[str, Any]]:
        doc = fitz.open(self.file_path)
        pages = []
        total_chars = 0

        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text("text")

            # Basic cleaning
            text = re.sub(r"\s+", " ", text)
            text = text.replace("-\n", "")
            text = text.strip()

            total_chars += len(text)

            pages.append({"page": i + 1, "text": text})

        doc.close()

        # Detect scanned PDF (less than 50 chars per page on average)
        if len(pages) > 0 and (total_chars / len(pages)) < 50:
            raise ValueError("Scanned or image-only PDF detected. No text found.")

        return pages
