"""
Wraps Chroma so the rest of the codebase never touches embedding/vector
details directly — only `build_vector_store` (write path, ingestion) and
`load_vector_store` (read path, agent runtime).
"""
import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import get_settings

logger = logging.getLogger(__name__)


def _embeddings() -> HuggingFaceEmbeddings:
    """Runs entirely on your machine — first call downloads the model
    (~90MB) from HuggingFace once and caches it locally; no API key,
    no per-call cost after that."""
    settings = get_settings()
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def build_vector_store(chunks: list[Document]) -> Chroma:
    """Embed and persist chunks. Idempotent: re-running ingestion on the
    same persist_dir adds duplicates, so callers doing a full re-index
    should clear chroma_persist_dir first."""
    settings = get_settings()
    store = Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        persist_directory=settings.chroma_persist_dir,
        collection_name="research_papers",
    )
    logger.info("Persisted %d chunks to %s", len(chunks), settings.chroma_persist_dir)
    return store


def load_vector_store() -> Chroma:
    """Load an already-built store for querying at agent runtime."""
    settings = get_settings()
    return Chroma(
        persist_directory=settings.chroma_persist_dir,
        embedding_function=_embeddings(),
        collection_name="research_papers",
    )
