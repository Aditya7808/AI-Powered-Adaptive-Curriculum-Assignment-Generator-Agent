from cachetools import TTLCache

from src.config import config

# We use global TTLCaches
retrieval_cache = TTLCache(
    maxsize=config.CACHE_MAX_ENTRIES, ttl=config.CACHE_TTL_SECONDS
)
answer_cache = TTLCache(maxsize=config.CACHE_MAX_ENTRIES, ttl=config.CACHE_TTL_SECONDS)


def clear_caches():
    retrieval_cache.clear()
    answer_cache.clear()
