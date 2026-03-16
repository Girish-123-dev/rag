"""
Dementia App - RAG Pipeline
============================
FR-204: System Administrator uploads PDF documents to update questionnaire
        content and voice agent responses dynamically.

Embedding model : neuml/pubmedbert-base-embeddings (medical-grade, 438MB)
Vector store    : pgvector via PostgreSQL (production default)
LLM             : OpenAI GPT-4o — see llm_factory.py

Backend options (set via environment variable):
    VECTOR_BACKEND=pgvector   → PostgreSQL + pgvector (default, production)
    VECTOR_BACKEND=faiss      → local file-based index (offline dev/testing)

pgvector setup (run ONCE in your PostgreSQL database):
------------------------------------------------------------
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS dementia_documents (
        id        BIGSERIAL PRIMARY KEY,
        content   TEXT,
        metadata  JSONB,
        embedding VECTOR(768)    -- 768 dims for pubmedbert-base-embeddings
    );

    CREATE INDEX IF NOT EXISTS dementia_documents_embedding_idx
        ON dementia_documents
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
------------------------------------------------------------
"""

import os
import logging
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "neuml/pubmedbert-base-embeddings"
EMBED_DIMENSIONS = 768          # pubmedbert output dimensions
CHUNK_SIZE       = 300
CHUNK_OVERLAP    = 150
FAISS_INDEX_PATH = "output/dementia_rag_index"

# Vector backend: "pgvector" (default/production) or "faiss" (local dev)
VECTOR_BACKEND   = os.getenv("VECTOR_BACKEND", "pgvector").lower()

# PostgreSQL connection string for pgvector
# Format: postgresql://user:password@host:port/database
# Example: postgresql://postgres:mypassword@localhost:5432/dementia_db
PG_CONNECTION    = os.getenv(
    "PG_CONNECTION",
    "postgresql://postgres:password@localhost:5432/dementia_db"
)
PG_COLLECTION    = os.getenv("PG_COLLECTION", "dementia_documents")

# ── Embedding Model ────────────────────────────────────────────────────────────
logger.info(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
logger.info("Embedding model loaded.")


# ── Embedding Wrapper (LangChain-compatible) ───────────────────────────────────
class _EmbedWrapper:
    """
    Wraps SentenceTransformer to satisfy LangChain's Embeddings interface.
    Used by both FAISS and pgvector backends.
    """
    def embed_documents(self, texts: list) -> list:
        vecs = embed_model.encode(texts, normalize_embeddings=True, batch_size=64)
        return vecs.tolist()

    def embed_query(self, text: str) -> list:
        vec = embed_model.encode([text], normalize_embeddings=True)[0]
        return vec.tolist()


# ── Raw Embedding Function (used internally) ───────────────────────────────────
def embed_text(texts) -> np.ndarray:
    """
    Embed a string or list of strings.
    Returns 1D array for single string, 2D for a list.
    """
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    embeddings = embed_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )
    return embeddings[0] if single else embeddings


# ── PDF Processing ─────────────────────────────────────────────────────────────
def extract_text_from_page(page) -> str:
    """Extract clean text from a single PDF page."""
    blocks = page.get_text("blocks")
    return "\n".join(b[4].strip() for b in blocks if b[4].strip())


def process_pdf(pdf_path: str) -> tuple:
    """
    Extract text from PDF, split into chunks, embed each chunk.

    Returns:
        docs       — list of LangChain Document objects
        embeddings — list of numpy arrays (one per doc)
    """
    logger.info(f"Processing PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    text_docs = []
    for i, page in enumerate(doc):
        text = extract_text_from_page(page)
        if not text:
            continue
        base_doc = Document(
            page_content=text,
            metadata={"page": i, "type": "text", "source": pdf_path},
        )
        text_docs.extend(splitter.split_documents([base_doc]))

    if not text_docs:
        logger.warning("No text content found in PDF.")
        return [], []

    logger.info(f"Embedding {len(text_docs)} text chunks...")
    raw = embed_text([d.page_content for d in text_docs])
    embeddings = [raw] if raw.ndim == 1 else list(raw)

    logger.info(f"PDF processed: {len(text_docs)} chunks embedded.")
    return text_docs, embeddings


# ── pgvector Backend ───────────────────────────────────────────────────────────
def _get_pgvector_store(create: bool = False):
    """
    Connect to pgvector store.

    Args:
        create: If True, creates a new collection (use when inserting new docs).
                If False, connects to existing collection (use when querying).

    Requires:
        pip install langchain-postgres psycopg2-binary
    """
    try:
        from langchain_postgres.vectorstores import PGVector
    except ImportError:
        raise ImportError(
            "pgvector packages not installed. Run:\n"
            "  pip install langchain-postgres psycopg2-binary"
        )

    store = PGVector(
        embeddings=_EmbedWrapper(),
        collection_name=PG_COLLECTION,
        connection=PG_CONNECTION,
        use_jsonb=True,
        create_extension=True,   # auto-runs CREATE EXTENSION IF NOT EXISTS vector
    )
    return store


# ── FAISS Backend (local dev) ──────────────────────────────────────────────────
def _load_faiss_store():
    from langchain_community.vectorstores import FAISS
    if Path(FAISS_INDEX_PATH).exists():
        logger.info("Loading existing FAISS index...")
        store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings=_EmbedWrapper(),
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS index loaded.")
        return store
    logger.info("No existing FAISS index found. Will create on first ingest.")
    return None


def _build_faiss_store(docs, embeddings):
    from langchain_community.vectorstores import FAISS
    pairs     = [(doc.page_content, emb.tolist()) for doc, emb in zip(docs, embeddings)]
    metadatas = [doc.metadata for doc in docs]
    return FAISS.from_embeddings(
        text_embeddings=pairs,
        embedding=_EmbedWrapper(),
        metadatas=metadatas,
    )


# ── DementiaRAGStore ───────────────────────────────────────────────────────────
class DementiaRAGStore:
    """
    Unified vector store manager for the dementia RAG system.

    Backend is controlled by VECTOR_BACKEND environment variable:
        pgvector  — PostgreSQL + pgvector (default, production)
        faiss     — local file on disk (offline dev/testing)

    All other modules (assessment_generator, voice_agent_context, admin_upload)
    call only ingest_pdf() and retrieve() — they never see which backend is active.
    """

    def __init__(self):
        self.vector_store = None
        self.image_store  = {}   # kept for API compatibility, not used
        self.backend      = VECTOR_BACKEND
        logger.info(f"Vector backend: {self.backend.upper()}")
        self._load_existing()

    def _load_existing(self):
        if self.backend == "pgvector":
            try:
                self.vector_store = _get_pgvector_store()
                logger.info("pgvector store connected.")
            except Exception as e:
                logger.warning(f"Could not connect to pgvector: {e}")
                logger.warning("Check PG_CONNECTION in your .env file.")
        else:
            self.vector_store = _load_faiss_store()

    def ingest_pdf(self, pdf_path: str):
        """
        FR-204: Ingest a PDF and store its embeddings in the vector store.
        Works identically for both pgvector and FAISS.
        """
        os.makedirs("output", exist_ok=True)
        docs, embeddings = process_pdf(pdf_path)

        if not docs:
            logger.warning("No content extracted — nothing ingested.")
            return

        if self.backend == "pgvector":
            # pgvector: add documents directly — they persist in PostgreSQL
            store = _get_pgvector_store()
            texts     = [d.page_content for d in docs]
            metadatas = [d.metadata for d in docs]
            store.add_texts(texts=texts, metadatas=metadatas)
            self.vector_store = store
            logger.info(f"Inserted {len(docs)} chunks into pgvector (table: {PG_COLLECTION})")

        else:
            # FAISS: build index and save to disk
            if self.vector_store is None:
                self.vector_store = _build_faiss_store(docs, embeddings)
            else:
                new_store = _build_faiss_store(docs, embeddings)
                self.vector_store.merge_from(new_store)
            self.vector_store.save_local(FAISS_INDEX_PATH)
            logger.info(f"FAISS index saved to {FAISS_INDEX_PATH}")

    def retrieve(self, query: str, k: int = 5) -> list:
        """
        Retrieve top-k most relevant chunks for a query.
        Works identically for both pgvector and FAISS.
        """
        if self.vector_store is None:
            logger.warning("Vector store is empty. Ingest a PDF first.")
            return []

        # Both backends support similarity_search with the same interface
        return self.vector_store.similarity_search(query, k=k)


# ── Message Builder ────────────────────────────────────────────────────────────
def build_multimodal_message(
    query: str,
    retrieved_docs: list,
    image_store: dict = None,
    domain: str = "dementia assessment",
) -> HumanMessage:
    """
    Build a LangChain HumanMessage from retrieved context chunks.
    Used by assessment_generator and voice_agent_context.
    """
    excerpts = "\n\n".join(
        f"[Page {d.metadata.get('page', '?')}]: {d.page_content}"
        for d in retrieved_docs
        if d.metadata.get("type") == "text"
    )

    prompt = (
        f"You are an AI assistant specialised in {domain}.\n\n"
        f"Task: {query}\n\n"
        f"Relevant clinical context:\n{excerpts}\n\n"
        "Answer using only the provided context. Be concise and clinically accurate."
    )

    return HumanMessage(content=prompt)


# ── Singleton ──────────────────────────────────────────────────────────────────
_store = None

def get_store() -> DementiaRAGStore:
    global _store
    if _store is None:
        _store = DementiaRAGStore()
    return _store