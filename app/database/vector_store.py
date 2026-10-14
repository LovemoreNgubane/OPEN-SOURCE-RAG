"""PostgreSQL + pgvector vector store.

Open-source replacement for Timescale Vector.
Embeddings come from sentence-transformers (see app/services/embeddings.py).
"""
import logging
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from app.config.settings import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages a PostgreSQL table with a pgvector embedding column."""

    def __init__(self) -> None:
        self.conninfo = (
            f"host={settings.postgres_host} "
            f"port={settings.postgres_port} "
            f"dbname={settings.postgres_db} "
            f"user={settings.postgres_user} "
            f"password={settings.postgres_password}"
        )
        self.dimension = settings.embedding_dimension

    @contextmanager
    def _connect(self) -> Iterator[psycopg.Connection]:
        conn = psycopg.connect(self.conninfo, autocommit=True)
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)
            yield conn
        finally:
            conn.close()

    def create_tables(self) -> None:
        """Create the documents table if it doesn't exist."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id         TEXT PRIMARY KEY,
                    contents   TEXT NOT NULL,
                    embedding  vector({self.dimension}),
                    metadata   JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
        logger.info("documents table ready")

    def create_index(self) -> None:
        """Create HNSW index for fast cosine similarity search."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS documents_embedding_idx
                ON documents USING hnsw (embedding vector_cosine_ops)
                """
            )
        logger.info("HNSW index ready")

    def upsert(
        self,
        doc_id: str,
        contents: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (id, contents, embedding, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET contents  = EXCLUDED.contents,
                    embedding = EXCLUDED.embedding,
                    metadata  = EXCLUDED.metadata
                """,
                (doc_id, contents, embedding, Jsonb(metadata or {})),
            )

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Cosine similarity search. Returns list of dicts (id, contents, metadata, similarity)."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, contents, metadata,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, limit),
            )
            rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "contents": row[1],
                "metadata": row[2],
                "similarity": float(row[3]),
            }
            for row in rows
        ]

    def delete_all(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM documents")

    def count(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            return cur.fetchone()[0]