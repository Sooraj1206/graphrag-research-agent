"""
Loads research papers (PDFs) from disk, splits them into retrieval-sized
chunks, and attaches metadata (source filename, page) used later for
citations in the synthesized answer.
"""
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings

logger = logging.getLogger(__name__)


def load_and_split_papers(papers_dir: str) -> list[Document]:
    """Load every PDF in `papers_dir` and split into overlapping chunks.

    Raises FileNotFoundError early rather than silently returning an
    empty list, since an empty ingestion run is almost always a bug.
    """
    settings = get_settings()
    paper_paths = sorted(Path(papers_dir).glob("*.pdf"))

    if not paper_paths:
        raise FileNotFoundError(
            f"No PDFs found in '{papers_dir}'. Add at least one paper before ingesting."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Document] = []
    for path in paper_paths:
        try:
            pages = PyPDFLoader(str(path)).load()
        except Exception:
            logger.exception("Failed to load %s — skipping", path.name)
            continue

        for page in pages:
            page.metadata["paper_title"] = path.stem
            page.metadata["source"] = path.name

        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
        logger.info("Loaded %s: %d pages -> %d chunks", path.name, len(pages), len(chunks))

    if not all_chunks:
        raise RuntimeError("All PDFs failed to load — check logs above for per-file errors.")

    return all_chunks
