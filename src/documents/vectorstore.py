from typing import Any

import chromadb

from src.config import config


class VectorStore:
    def __init__(self, use_ephemeral: bool = False):
        if use_ephemeral:
            self.client = chromadb.EphemeralClient()
        else:
            self.client = chromadb.PersistentClient(path=config.CHROMA_DIR)

        self.collection_name = "workspace_kb"

        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "embedding_model": config.EMBEDDING_MODEL,
                    "hnsw:space": "cosine",
                    "hnsw:M": config.HNSW_M,
                    "hnsw:construction_ef": config.HNSW_CONSTRUCTION_EF,
                    "hnsw:search_ef": config.HNSW_SEARCH_EF,
                },
            )

    def upsert_chunks(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ):
        if not ids:
            return

        # Batch size
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

    def delete_by_doc(self, doc_id: str):
        self.collection.delete(where={"doc_id": doc_id})

    def dense_search(
        self, query_embedding: list[float], n: int, doc_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        where = None
        if doc_ids:
            if len(doc_ids) == 1:
                where = {"doc_id": doc_ids[0]}
            else:
                where = {"doc_id": {"$in": doc_ids}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        out = []
        if not results["ids"] or not results["ids"][0]:
            return out

        for i in range(len(results["ids"][0])):
            # Distance is cosine distance, convert to similarity
            sim = 1.0 - results["distances"][0][i]
            out.append(
                {
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": sim,
                }
            )
        return out

    def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        res = self.collection.get(ids=ids)
        out = []
        for i in range(len(res["ids"])):
            out.append(
                {
                    "id": res["ids"][i],
                    "document": res["documents"][i],
                    "metadata": res["metadatas"][i],
                }
            )
        return out

    def get_all_chunks(
        self, doc_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        where = None
        if doc_ids:
            if len(doc_ids) == 1:
                where = {"doc_id": doc_ids[0]}
            else:
                where = {"doc_id": {"$in": doc_ids}}

        res = self.collection.get(where=where)
        out = []
        for i in range(len(res["ids"])):
            out.append(
                {
                    "id": res["ids"][i],
                    "document": res["documents"][i],
                    "metadata": res["metadatas"][i],
                }
            )
        return out

    def count(self, doc_id: str | None = None) -> int:
        if doc_id:
            return len(self.collection.get(where={"doc_id": doc_id})["ids"])
        return self.collection.count()

    def reset(self):
        try:
            self.client.delete_collection(self.collection_name)
        except ValueError:
            pass
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "embedding_model": config.EMBEDDING_MODEL,
                "hnsw:space": "cosine",
                "hnsw:M": config.HNSW_M,
                "hnsw:construction_ef": config.HNSW_CONSTRUCTION_EF,
                "hnsw:search_ef": config.HNSW_SEARCH_EF,
            },
        )
