import re
from typing import Any

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.corpus_chunks = []

    def build(self, chunks: list[dict[str, Any]]):
        if not chunks:
            self.bm25 = None
            self.corpus_chunks = []
            return

        self.corpus_chunks = chunks
        tokenized_corpus = [_tokenize(c["document"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self, query: str, n: int, doc_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        if not self.bm25 or not self.corpus_chunks:
            return []

        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        results = []
        for i, score in enumerate(scores):
            if score <= 0:
                continue
            chunk = self.corpus_chunks[i]
            if doc_ids and chunk["metadata"]["doc_id"] not in doc_ids:
                continue

            c_copy = dict(chunk)
            c_copy["score"] = score
            results.append(c_copy)

        # Sort
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:n]
