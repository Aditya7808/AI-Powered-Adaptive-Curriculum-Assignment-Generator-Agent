from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def chunk_pages(
        self, pages: list[dict[str, Any]], doc_id: str, file_name: str
    ) -> list[dict[str, Any]]:
        chunks = []
        chunk_index = 0

        for page in pages:
            text = page["text"]
            if not text:
                continue

            page_chunks = self.splitter.split_text(text)
            for chunk_text in page_chunks:
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}:{page['page']}:{chunk_index}",
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "page": page["page"],
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                    }
                )
                chunk_index += 1

        return chunks
