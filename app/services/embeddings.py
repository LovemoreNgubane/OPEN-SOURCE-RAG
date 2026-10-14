"""Local embeddings via sentence-transformers.

Open-source replacement for OpenAI's text-embedding-3-small.
Uses Meta's PyTorch under the hood.
"""
import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the model once per process. First call downloads ~90 MB."""
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    return SentenceTransformer(settings.embedding_model)


def embed_text(text: str) -> list[float]:
    """Embed a single string. Used for search queries."""
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings in one pass. Used for bulk inserts."""
    model = _get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )
    return [v.tolist() for v in vectors]