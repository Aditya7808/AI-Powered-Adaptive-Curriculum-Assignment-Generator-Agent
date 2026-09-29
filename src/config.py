from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    LLM_PROVIDER: str = "huggingface"
    HUGGINGFACEHUB_API_TOKEN: str = ""
    HF_LLM_REPO_ID: str = ""
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    DB_PATH: str = "data/app.db"
    DUPLICATE_THRESHOLD: float = 0.85
    QUALITY_THRESHOLD: float = 0.7
    MAX_REGEN_ROUNDS: int = 2
    CHROMA_DIR: str = "data/chroma"
    UPLOAD_DIR: str = "data/uploads"
    MAX_UPLOAD_MB: int = 25
    MAX_FILES_PER_UPLOAD: int = 10
    MIN_RELEVANT_CHUNKS: int = 1
    MAX_REWRITES: int = 1
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ENABLE_BM25: bool = True
    ENABLE_RERANKER: bool = True
    ENABLE_NEIGHBOR_EXPANSION: bool = True
    ENABLE_HYDE: bool = False
    LLM_GRADER: str = "off"
    DENSE_TOP_N: int = 20
    BM25_TOP_N: int = 20
    FUSION_K: int = 60
    RERANK_CANDIDATES: int = 24
    FINAL_TOP_K: int = 5
    RERANK_MAX_LENGTH: int = 256
    RERANK_BATCH_SIZE: int = 16
    RERANK_RELEVANT_THRESHOLD: float = 0.5
    RERANK_DROP_THRESHOLD: float = 0.1
    CONFIDENT_THRESHOLD: float = 0.75
    NEIGHBOR_EXPAND_TOP_N: int = 2
    MAX_CONTEXT_TOKENS: int = 3000
    SELF_CHECK_MODE: str = "low_confidence"
    CACHE_TTL_SECONDS: int = 900
    CACHE_MAX_ENTRIES: int = 256
    LLM_TIMEOUT_SECONDS: int = 30
    HNSW_M: int = 16
    HNSW_CONSTRUCTION_EF: int = 100
    HNSW_SEARCH_EF: int = 64
    TORCH_NUM_THREADS: int = 4
    BM25_DIR: str = "data/bm25"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


config = Settings()
