"""
Builds the paper knowledge graph in Neo4j.

Schema (kept deliberately small so extraction stays reliable):
  (:Paper {title})
  (:Author {name})
  (:Concept {name})
  (:Method {name})
  (:Dataset {name})

  (:Paper)-[:AUTHORED_BY]->(:Author)
  (:Paper)-[:INTRODUCES]->(:Concept)
  (:Paper)-[:USES_METHOD]->(:Method)
  (:Paper)-[:EVALUATES_ON]->(:Dataset)
  (:Paper)-[:CITES]->(:Paper)

Extraction runs per-paper (on the concatenated first N chunks, i.e. the
abstract/intro — where these facts concentrate) rather than per-chunk,
which keeps LLM calls and duplicate-entity noise down.
"""
import logging
from collections import defaultdict

from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings

logger = logging.getLogger(__name__)

MAX_CHARS_FOR_EXTRACTION = 6000  # abstract + intro is enough signal for this schema


class ExtractedRelation(BaseModel):
    source: str = Field(description="Entity name, e.g. paper title or author")
    relation: str = Field(description="One of: AUTHORED_BY, INTRODUCES, USES_METHOD, EVALUATES_ON, CITES")
    target: str = Field(description="Entity name")
    target_type: str = Field(description="One of: Author, Concept, Method, Dataset, Paper")


class PaperExtraction(BaseModel):
    authors: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    cited_paper_titles: list[str] = Field(default_factory=list)


def _group_chunks_by_paper(chunks: list[Document]) -> dict[str, str]:
    text_by_paper: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        title = chunk.metadata["paper_title"]
        joined_len = sum(len(t) for t in text_by_paper[title])
        if joined_len < MAX_CHARS_FOR_EXTRACTION:
            text_by_paper[title].append(chunk.page_content)
    return {title: "\n".join(parts)[:MAX_CHARS_FOR_EXTRACTION] for title, parts in text_by_paper.items()}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _extract_for_paper(llm: ChatOllama, title: str, text: str) -> PaperExtraction:
    structured_llm = llm.with_structured_output(PaperExtraction)
    prompt = (
        f"Extract structured metadata from this research paper excerpt.\n"
        f"Paper title: {title}\n\n"
        f"Text:\n{text}\n\n"
        "Only extract entities explicitly named in the text. Do not invent authors, "
        "methods, or citations. Use short canonical names (e.g. 'Transformer', not "
        "'the Transformer architecture proposed in this work')."
    )
    return structured_llm.invoke(prompt)


def build_knowledge_graph(chunks: list[Document]) -> Neo4jGraph:
    """Extract entities/relations per paper and MERGE them into Neo4j.
    MERGE (not CREATE) makes this safe to re-run without duplicating nodes."""
    settings = get_settings()
    graph = Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    llm = ChatOllama(model=settings.chat_model, base_url=settings.ollama_base_url, temperature=0)

    graph.query("CREATE CONSTRAINT paper_title IF NOT EXISTS FOR (p:Paper) REQUIRE p.title IS UNIQUE")
    graph.query("CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")

    text_by_paper = _group_chunks_by_paper(chunks)

    for title, text in text_by_paper.items():
        graph.query("MERGE (p:Paper {title: $title})", {"title": title})

        try:
            extraction = _extract_for_paper(llm, title, text)
        except Exception:
            logger.exception("Extraction failed for '%s' after retries — skipping graph edges", title)
            continue

        _write_extraction(graph, title, extraction)
        logger.info(
            "Graph updated for '%s': %d authors, %d concepts, %d methods, %d citations",
            title, len(extraction.authors), len(extraction.concepts),
            len(extraction.methods), len(extraction.cited_paper_titles),
        )

    return graph


def _write_extraction(graph: Neo4jGraph, paper_title: str, extraction: PaperExtraction) -> None:
    for author in extraction.authors:
        graph.query(
            """
            MERGE (a:Author {name: $author})
            MERGE (p:Paper {title: $title})
            MERGE (p)-[:AUTHORED_BY]->(a)
            """,
            {"author": author, "title": paper_title},
        )
    for concept in extraction.concepts:
        graph.query(
            """
            MERGE (c:Concept {name: $concept})
            MERGE (p:Paper {title: $title})
            MERGE (p)-[:INTRODUCES]->(c)
            """,
            {"concept": concept, "title": paper_title},
        )
    for method in extraction.methods:
        graph.query(
            """
            MERGE (m:Method {name: $method})
            MERGE (p:Paper {title: $title})
            MERGE (p)-[:USES_METHOD]->(m)
            """,
            {"method": method, "title": paper_title},
        )
    for dataset in extraction.datasets:
        graph.query(
            """
            MERGE (d:Dataset {name: $dataset})
            MERGE (p:Paper {title: $title})
            MERGE (p)-[:EVALUATES_ON]->(d)
            """,
            {"dataset": dataset, "title": paper_title},
        )
    for cited in extraction.cited_paper_titles:
        graph.query(
            """
            MERGE (cited:Paper {title: $cited})
            MERGE (p:Paper {title: $title})
            MERGE (p)-[:CITES]->(cited)
            """,
            {"cited": cited, "title": paper_title},
        )
