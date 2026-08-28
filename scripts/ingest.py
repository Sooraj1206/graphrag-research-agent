"""
Run once to build both the vector store and the knowledge graph from
PDFs in ./papers.

Usage:
    python -m scripts.ingest
"""
import logging
import sys

from ingestion.graph_builder import build_knowledge_graph
from ingestion.pdf_loader import load_and_split_papers
from ingestion.vector_store import build_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    try:
        chunks = load_and_split_papers("papers")
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info("Building vector store...")
    build_vector_store(chunks)

    logger.info("Building knowledge graph (this calls the LLM once per paper)...")
    build_knowledge_graph(chunks)

    logger.info("Ingestion complete. %d chunks across %d papers indexed.",
                len(chunks), len({c.metadata["paper_title"] for c in chunks}))


if __name__ == "__main__":
    main()
