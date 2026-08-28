from unittest.mock import MagicMock

from langchain_core.documents import Document

from ingestion.graph_builder import _group_chunks_by_paper
from ingestion.pdf_loader import load_and_split_papers


def test_load_and_split_papers_raises_on_missing_dir(tmp_path):
    empty_dir = tmp_path / "no_papers_here"
    empty_dir.mkdir()
    try:
        load_and_split_papers(str(empty_dir))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as e:
        assert "No PDFs found" in str(e)


def test_group_chunks_by_paper_respects_char_cap():
    long_text = "x" * 5000
    chunks = [
        Document(page_content=long_text, metadata={"paper_title": "Paper A"}),
        Document(page_content=long_text, metadata={"paper_title": "Paper A"}),
        Document(page_content="short intro", metadata={"paper_title": "Paper B"}),
    ]
    grouped = _group_chunks_by_paper(chunks)

    assert set(grouped.keys()) == {"Paper A", "Paper B"}
    assert len(grouped["Paper A"]) <= 6000
    assert grouped["Paper B"] == "short intro"
