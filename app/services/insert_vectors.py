"""Load a CSV of question/answer pairs and insert them into PostgreSQL + pgvector.

Open-source equivalent of the tutorial's insert_vectors.py.
Reads: data/faq_dataset.csv (columns: question, answer, category)
Writes: documents table (id, contents, embedding, metadata)
"""
import logging
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.database.vector_store import VectorStore
from app.services.embeddings import embed_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CSV location
DATA_FILE = Path("data/faq_dataset.csv")


def prepare_records(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw CSV rows into rows ready for insertion.

    - Combines question + answer into one 'contents' field
    - Generates a UUID v1 (time-based) for each row
    - Builds the metadata JSON
    - Leaves 'embedding' as a placeholder - filled later in batch
    """
    def _row(row: pd.Series) -> pd.Series:
        content = f"Question: {row['question']}\nAnswer: {row['answer']}"
        return pd.Series(
            {
                "id": str(uuid.uuid1()),
                "metadata": {
                    "category": row.get("category", "general"),
                    "created_at": datetime.now().isoformat(),
                },
                "contents": content,
            }
        )

    return df.apply(_row, axis=1)


def main() -> None:
    if not DATA_FILE.exists():
        raise SystemExit(
            f"Data file not found: {DATA_FILE}\n"
            f"Place a CSV with columns 'question', 'answer', 'category' at that path."
        )

    logger.info(f"Reading {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, sep=";")
    logger.info(f"Loaded {len(df)} rows")

    # Build id / metadata / contents
    records_df = prepare_records(df)

    # Embed all 'contents' in one batch (much faster than one-by-one)
    logger.info("Generating embeddings (first run downloads ~90 MB model)...")
    embeddings = embed_batch(records_df["contents"].tolist())
    records_df["embedding"] = embeddings

    # Prepare DB
    store = VectorStore()
    store.create_tables()
    store.create_index()

    # Insert row by row
    logger.info(f"Inserting {len(records_df)} records...")
    for _, row in records_df.iterrows():
        store.upsert(
            doc_id=row["id"],
            contents=row["contents"],
            embedding=row["embedding"],
            metadata=row["metadata"],
        )

    logger.info(f"Done. Total documents in store: {store.count()}")


if __name__ == "__main__":
    main()