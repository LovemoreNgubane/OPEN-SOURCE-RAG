"""Ingest PDFs into the vector store.

Reads all PDFs from data/, splits each into chunks,
embeds each chunk, and inserts into pgvector.

Usage:
    uv run python -m app.services.ingest_pdf
"""
import logging
import uuid
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader

from app.config.settings import settings
from app.database.vector_store import VectorStore
from app.services.embeddings import embed_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
        logger.info(f"Extracted page {page_num}/{len(reader.pages)}")
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    if not words:
        return []

    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return chunks


def ingest_pdf(pdf_path: Path) -> int:
    """Extract, chunk, embed, and store a single PDF."""
    logger.info(f"Processing: {pdf_path.name}")

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        logger.warning(f"No text extracted from {pdf_path.name}")
        return 0

    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    logger.info(f"Split into {len(chunks)} chunks")

    logger.info("Generating embeddings...")
    embeddings = embed_batch(chunks)

    store = VectorStore()
    store.create_tables()
    store.create_index()

    for chunk, embedding in zip(chunks, embeddings):
        doc_id = str(uuid.uuid1())
        store.upsert(
            doc_id=doc_id,
            contents=chunk,
            embedding=embedding,
            metadata={
                "source": pdf_path.name,
                "ingested_at": datetime.now().isoformat(),
            },
        )

    logger.info(f"Inserted {len(chunks)} chunks from {pdf_path.name}")
    return len(chunks)


def main() -> None:
    data_dir = Path(settings.data_dir)
    pdfs = list(data_dir.glob("*.pdf"))

    if not pdfs:
        raise SystemExit(f"No PDFs found in {data_dir}. Add one and re-run.")

    total = 0
    for pdf in pdfs:
        total += ingest_pdf(pdf)

    store = VectorStore()
    logger.info(f"Done. Inserted {total} chunks. Total documents: {store.count()}")


if __name__ == "__main__":
    main()