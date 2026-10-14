"""Central configuration for the RAG system.

Reads from .env file. All other modules import `settings` from here.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag"
    postgres_user: str = "rag"
    postgres_password: str = "ragpassword"

    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # LLM
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    top_k_results: int = 5
    bm25_weight: float = 0.3
    vector_weight: float = 0.7

    # Paths
    data_dir: Path = Path("./data")
    raw_data_dir: Path = Path("./data/raw")
    processed_data_dir: Path = Path("./data/processed")

    @property
    def database_url(self) -> str:
        """SQLAlchemy-compatible connection URL for PostgreSQL."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# Single shared instance — import this everywhere
settings = Settings()