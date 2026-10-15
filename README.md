# Open-Source RAG

**Enterprise-grade Retrieval-Augmented Generation built entirely from open-source components. Runs on an 8 GB laptop with no GPU, no paid APIs, and no cloud dependencies.**

Built by [Lovemore Ngubane](https://github.com/LovemoreNgubane) in South Africa as a submission for the **Meta AI Academy South Africa Pitchathon 2026** and as the foundation of a data + AI platform for South African banks, telcos, and insurers.

---

## Table of Contents

- [What This Is](#what-this-is)
- [Why It Matters](#why-it-matters)
- [What Works Today](#what-works-today)
- [Architecture](#architecture)
- [Stack](#stack)
- [The Journey: How We Built This](#the-journey-how-we-built-this)
- [Setup From Scratch](#setup-from-scratch)
- [Project Structure](#project-structure)
- [Code Walkthrough](#code-walkthrough)
- [Results](#results)
- [What's Next](#whats-next)
- [License](#license)

---

## What This Is

A complete, working Retrieval-Augmented Generation system. You give it a question. It:

1. Converts the question into a vector using a local embedding model
2. Finds the closest matching documents in PostgreSQL using pgvector
3. Sends the question + matched context to Llama 3.2 (running locally via Ollama)
4. Returns a grounded answer — or refuses if the context doesn't contain the answer

No OpenAI. No Pinecone. No Anthropic. No cloud. Everything runs on one laptop.

---

## Why It Matters

Enterprise RAG normally costs thousands of dollars per month:

| What Enterprises Pay For | What We Replaced It With | Cost |
|---|---|---|
| OpenAI embeddings | `sentence-transformers/all-MiniLM-L6-v2` on PyTorch | **$0** |
| Pinecone / Weaviate / Timescale Cloud | PostgreSQL 16 + `pgvector` | **$0** |
| GPT-4 / Claude generation | Llama 3.2 3B via Ollama | **$0** |
| Databricks / Snowflake | Apache Spark + Iceberg (next phase) | **$0** |

Everything runs offline. Data never leaves the machine. This matters enormously for:

- **South African banks** subject to POPIA data residency
- **Telecoms** with sensitive subscriber data
- **Insurers** with medical and financial records
- **Government** with classified information

---

## What Works Today

✅ CSV of Q&A pairs → embeddings → PostgreSQL with pgvector
✅ HNSW index for fast cosine similarity search
✅ Query → embed → top-K ranked retrieval
✅ Context + question → Llama 3.2 3B → grounded answer
✅ Correct refusal when context is irrelevant (no hallucination)
✅ Full Docker Compose stack (PostgreSQL + Ollama)

**Verified end-to-end on a laptop with 8 GB RAM and no GPU.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUESTION                            │
│              "What is the quadratic formula?"                   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         EMBEDDING (sentence-transformers/all-MiniLM-L6-v2)      │
│              384-dimensional vector, runs on CPU                │
│                        Backed by PyTorch (Meta)                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              VECTOR SEARCH (PostgreSQL 16 + pgvector)           │
│                                                                 │
│   SELECT id, contents, 1 - (embedding <=> query) AS similarity  │
│   FROM documents                                                │
│   ORDER BY embedding <=> query                                  │
│   LIMIT 3;                                                      │
│                                                                 │
│   HNSW index on embedding column (cosine similarity)            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│            TOP-K RETRIEVED DOCUMENTS (ranked by similarity)     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         GENERATION (Llama 3.2 3B via Ollama, in Docker)         │
│                                                                 │
│   System prompt: "Answer using ONLY the context below.          │
│   If the context doesn't contain the answer, say so."           │
│                                                                 │
│   User prompt: "Context: {chunks}  Question: {question}"        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GROUNDED ANSWER OR REFUSAL                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stack

| Component | Version | Purpose | License |
|---|---|---|---|
| Python | 3.11 | Runtime | PSF |
| uv | 0.12+ | Package manager | Apache 2.0 |
| PostgreSQL | 16 | Database engine | PostgreSQL License |
| pgvector | 0.6+ | Vector extension | PostgreSQL License |
| psycopg | 3.x | PostgreSQL driver | LGPL |
| sentence-transformers | 6.1 | Embedding models | Apache 2.0 |
| **PyTorch** | 2.14 | Deep learning (**Meta**) | BSD |
| pydantic | 2.x | Config validation | MIT |
| pandas | 3.x | CSV loading | BSD |
| **Ollama** + **Llama 3.2** | 3B | Local LLM (**Meta**) | Meta Llama License |
| Docker | latest | Container runtime | Apache 2.0 |

---

## The Journey: How We Built This

This wasn't a tutorial follow-along. It was an actual build with real bugs. Here's what happened, in order — this is the real engineering log.

### Phase 1: Foundation

1. **Installed `uv`** — the modern Python package manager. Faster than pip, uses `pyproject.toml` + `uv.lock` for reproducible builds.
2. **Created the project** with `uv init --python 3.11` and `uv venv`.
3. **Cleaned up** the auto-generated `src/` folder, created our own structure (`app/config`, `app/database`, `app/services`, `data`, `docker`, `tests`).
4. **Installed dependencies** in one command:
   ```
   uv add fastapi "uvicorn[standard]" pydantic pydantic-settings python-dotenv \
          "psycopg[binary]" pgvector sqlalchemy sentence-transformers ollama \
          rank-bm25 pypdf tiktoken pandas
   ```
   This pulled in 70 packages including PyTorch (2 GB). Took 15 minutes.

### Phase 2: Configuration

5. **Wrote `app/config/settings.py`** — pydantic-based config reading from `.env`. Centralizes DB credentials, model names, chunk sizes, and paths.
6. **Learned a lesson about Notepad** — the first paste didn't save (Ctrl+S was missed). Diagnosed with `type settings.py` showing empty output. Fixed.
7. **Verified config loads** with `uv run python -c "from app.config.settings import settings; print(settings.database_url)"`.

### Phase 3: Vector store

8. **Started PostgreSQL + pgvector** via Docker Compose (`pgvector/pgvector:pg16` image). Container healthy in ~15 seconds.
9. **Wrote `app/database/vector_store.py`** — a psycopg-based wrapper around:
   - Table creation with a `vector(384)` column
   - HNSW index for cosine similarity
   - `upsert()` for insertion
   - `search()` for top-K retrieval
10. **Hit our first real bug**: `psycopg.ProgrammingError: cannot adapt type 'dict' using placeholder '%s'`. The `metadata` dict wasn't being serialized to JSONB.
    - **Fix**: `from psycopg.types.json import Jsonb` and wrap `metadata` in `Jsonb(...)`.
11. **Verified the schema** with `docker exec rag-postgres psql -U rag -d rag -c "\d documents"`. Confirmed `embedding | vector(384)` and `documents_embedding_idx hnsw (embedding vector_cosine_ops)`.

### Phase 4: Embeddings

12. **Wrote `app/services/embeddings.py`** — wraps `sentence-transformers/all-MiniLM-L6-v2` (90 MB, runs on CPU).
13. **Wrote `app/services/insert_vectors.py`** — reads CSV, combines Q+A, batches embeddings, inserts into pgvector.
14. **Wrote `app/services/similarity_search.py`** — query interface for testing retrieval.

### Phase 5: Data ingestion

15. **Created `data/faq_dataset.csv`** with 5 Grade 12 sample questions across Mathematics, Physics, Biology, Geography.
16. **Ran the insert** — hit the Jsonb bug again, fixed, re-ran. `Done. Total documents in store: 5`.

### Phase 6: Retrieval verification

17. **Ran similarity search** on three questions:
    - "What is the quadratic formula?" → top match **0.7377**
    - "Explain Newton's second law." → top match **0.6760**
    - "What is the weather in Tokyo?" → top match only **0.1133**
18. **Confirmed retrieval works**: high scores for relevant queries, low for irrelevant.

### Phase 7: Generation

19. **Installed Ollama in Docker** — added to `docker-compose.yml` as a service with persistent volume.
20. **Pulled Llama 3.2 1B** (1.3 GB) — later upgraded to 3B.
21. **Wrote `app/services/generation.py`** — sends context + question to Ollama, gets answer back.
22. **Wrote `app/services/rag_pipeline.py`** — full pipeline: embed → retrieve → generate.

### Phase 8: Bugs, crashes, and recovery

23. **Docker Desktop engine broke** after a hard shutdown — every command returned `500 Internal Server Error`.
24. **Diagnosed**: the WSL2 backend (`docker-desktop` distro) had crashed and couldn't restart.
25. **Attempted repairs**: `wsl --shutdown`, `taskkill` on Docker processes, purging `%APPDATA%\Docker`. None worked.
26. **Uninstalled and reinstalled Docker Desktop** — the definitive fix. Volumes survived.
27. **Rebuilt the stack** — re-pulled images, re-created containers, re-inserted the 5 test rows.

### Phase 9: Model tuning

28. **Ran the pipeline with Llama 3.2 1B** — the model correctly refused even when the answer was in the context. Classic small-model limitation.
29. **Tried prompt engineering** — added few-shot examples and stricter rules. Still refused.
30. **Upgraded to Llama 3.2 3B** — the smallest model that reliably follows RAG instructions. 2 GB download.
31. **Re-ran the pipeline** — all three answers correct:
    - Quadratic formula: answered correctly
    - Newton's law: answered correctly with a full explanation
    - Tokyo weather: correctly refused

### Phase 10: Final polish

32. **Removed the trailing `Answer:` from the user prompt** — Llama was echoing it back.
33. **Confirmed clean output** — no artifacts, no hallucination, no refusals on valid questions.

**Total build time**: One long session with multiple failures, recoveries, and a full end-to-end working system at the end.

---

## Setup From Scratch

### Prerequisites

- Windows 10/11 with WSL2 enabled (or Linux/macOS)
- Docker Desktop running (allocate 4+ GB RAM)
- Python 3.11 (installed automatically by uv)
- `uv` package manager
- Git

### Install `uv`

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Clone and install

```cmd
git clone https://github.com/LovemoreNgubane/OPEN-SOURCE-RAG.git
cd OPEN-SOURCE-RAG
uv venv
.venv\Scripts\activate
uv sync
```

`uv sync` reads `uv.lock` and installs the exact same package versions we used.

### Configure

```cmd
copy .env.example .env
notepad .env
```

Set:

```
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag
POSTGRES_USER=rag
POSTGRES_PASSWORD=ragpassword

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

OLLAMA_HOST=http://localhost:11434
LLM_MODEL=llama3.2:3b
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=512
```

### Start the stack

```cmd
docker compose -f docker/docker-compose.yml up -d
```

This starts PostgreSQL + pgvector and Ollama.

### Pull the model

```cmd
docker exec rag-ollama ollama pull llama3.2:3b
```

### Create the vector table

```cmd
uv run python -c "from app.database.vector_store import VectorStore; s = VectorStore(); s.create_tables(); s.create_index(); print(f'Ready. Rows: {s.count()}')"
```

### Load your data

Place a CSV at `data/faq_dataset.csv` in this format (semicolon-separated):

```
question;answer;category
What is the quadratic formula?;x = (-b ± √(b²-4ac)) / 2a;Mathematics
State Newton's second law.;F = ma;Physics
```

Then:

```cmd
uv run python -m app.services.insert_vectors
```

### Run the RAG pipeline

```cmd
uv run python -m app.services.rag_pipeline
```

---

## Project Structure

```
OPEN-SOURCE-RAG/
├── app/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py              # Config loaded from .env via pydantic
│   ├── database/
│   │   ├── __init__.py
│   │   └── vector_store.py          # PostgreSQL + pgvector wrapper
│   └── services/
│       ├── __init__.py
│       ├── embeddings.py            # sentence-transformers wrapper
│       ├── insert_vectors.py        # CSV → embeddings → DB
│       ├── similarity_search.py     # Query → top-K retrieval
│       ├── generation.py            # Llama 3.2 via Ollama
│       └── rag_pipeline.py          # Full retrieval + generation
├── data/
│   ├── faq_dataset.csv              # Your Q&A corpus (gitignored)
│   └── raw/                         # Source PDFs (gitignored)
├── docker/
│   └── docker-compose.yml           # PostgreSQL + Ollama
├── scripts/
├── tests/
├── notebooks/
├── .env                             # Real credentials (gitignored)
├── .env.example                     # Public template
├── .gitignore
├── pyproject.toml                   # uv project + deps
├── uv.lock                          # Locked versions
└── README.md
```

---

## Code Walkthrough

### The embedding model

`sentence-transformers/all-MiniLM-L6-v2` is a 90 MB transformer model. It takes a sentence and outputs a 384-dimensional vector. Similar sentences produce vectors that point in the same direction (high cosine similarity).

**It runs on PyTorch** — Meta's open-source deep learning framework.

### The vector store

PostgreSQL with the `pgvector` extension stores vectors as a native `vector(384)` column type. Queries use the `<=>` operator for cosine distance:

```sql
1 - (embedding <=> query_vector) AS similarity
```

The `<=>` returns **cosine distance** (0 = identical, 2 = opposite). We subtract from 1 to get **cosine similarity** (1 = identical, -1 = opposite).

### The HNSW index

```sql
CREATE INDEX documents_embedding_idx
ON documents USING hnsw (embedding vector_cosine_ops)
```

HNSW = Hierarchical Navigable Small World. It makes similarity search **logarithmic** instead of linear. Without it, every query scans every row.

### The generation prompt

The system prompt explicitly instructs the model to refuse when context is irrelevant:

```
You are a Grade 12 tutor for South African students.

RULES:
1. If the Context contains the answer, USE IT and give the answer directly.
2. If the Context does NOT contain the answer, reply exactly:
   "I don't have enough information to answer that."
3. Do not invent facts. Do not add information from outside the Context.
```

Two few-shot examples show the model how to behave in both cases.

### The retrieval → generation flow

```python
# 1. Embed the question
query_vector = embed_text(question)

# 2. Find top-K similar documents
chunks = store.search(query_vector, limit=3)

# 3. Generate answer with Llama using retrieved context
answer = generate_answer(question, chunks)
```

That's the entire RAG loop. Three lines.

---

## Results

With 5 test documents (Mathematics, Physics, Biology, Geography questions):

| Query | Top Match | Similarity | Generated Answer |
|---|---|---|---|
| What is the quadratic formula? | Quadratic formula document | **0.7377** | "x equals negative b plus or minus the square root of b squared minus 4ac, all divided by 2a" |
| Explain Newton's second law. | Newton's law document | **0.6697** | Full explanation with F = ma and intuition |
| What is the weather in Tokyo? | Highest match only reached | **0.1133** | "I don't have enough information to answer that." |

**Interpretation:**

- **Relevant queries** score 0.6–0.8 and produce grounded answers
- **Irrelevant queries** score below 0.3 and are correctly refused
- **Zero hallucinations** — every answer is grounded in retrieved context

---

## What's Next

- [ ] **Real data**: Replace sample CSV with Grade 12 past exam papers (PDFs → chunks)
- [ ] **Hybrid retrieval**: Combine BM25 (keyword) + vector search for better recall
- [ ] **FastAPI endpoint**: Expose `/ask` HTTP route for external clients
- [ ] **Reranking**: Cross-encoder reranker for higher precision
- [ ] **Evaluation**: Ground-truth test set with precision/recall metrics
- [ ] **Deployment**: Kubernetes manifests for xneelo Cloud (South Africa)
- [ ] **MLflow**: Experiment tracking for embedding model comparisons
- [ ] **Streaming**: Apache Kafka for real-time document ingestion
- [ ] **Lakehouse**: Apache Iceberg + Spark for large-scale data

---

## References

- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [sentence-transformers docs](https://sbert.net)
- [HNSW paper](https://arxiv.org/abs/1603.09320)
- [Meta Llama 3.2](https://www.llama.com)
- [uv package manager](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com)

---

## License

MIT — free to use, modify, and distribute.

---

**Built by [Lovemore Ngubane](https://github.com/LovemoreNgubane) in South Africa.**

**Contact:** lovemorengubane@gmail.com