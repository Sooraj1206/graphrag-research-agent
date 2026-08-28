"""
Two retrieval tools the agent's nodes call into:

- vector_search: semantic similarity search over paper text chunks (RAG).
- graph_search: translates the question into Cypher via GraphCypherQAChain
  and queries Neo4j for relational facts (authors, methods, citations).

Kept as plain functions (not @tool-decorated) since the graph *nodes*
decide when to call them (deterministic routing) rather than an LLM
picking tools freely — more predictable and cheaper for this use case.
"""
import logging

from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_ollama import ChatOllama

from config import get_settings
from ingestion.vector_store import load_vector_store

logger = logging.getLogger(__name__)

_CYPHER_GENERATION_TEMPLATE = """You are translating a research question into a Cypher query
for a Neo4j graph with this schema:

  (:Paper {{title}})-[:AUTHORED_BY]->(:Author {{name}})
  (:Paper)-[:INTRODUCES]->(:Concept {{name}})
  (:Paper)-[:USES_METHOD]->(:Method {{name}})
  (:Paper)-[:EVALUATES_ON]->(:Dataset {{name}})
  (:Paper)-[:CITES]->(:Paper)

Question: {question}

Return only the Cypher query, no explanation.
"""


def vector_search(question: str) -> list[str]:
    """Return the top-k most relevant chunk texts (with source paper) for the question."""
    settings = get_settings()
    store = load_vector_store()
    docs = store.similarity_search(question, k=settings.retriever_k)
    return [
        f"[{doc.metadata.get('paper_title', 'unknown')}] {doc.page_content}"
        for doc in docs
    ]


def graph_search(question: str) -> list[str]:
    """Run an LLM-generated Cypher query against Neo4j and return formatted results.
    Returns an empty list (not an exception) on query failure — the synthesis
    node treats missing graph context as 'no relational facts found', which
    is a valid and common outcome, not an error state."""
    settings = get_settings()
    graph = Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    llm = ChatOllama(model=settings.chat_model, base_url=settings.ollama_base_url, temperature=0)

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=_CYPHER_GENERATION_TEMPLATE,
        verbose=False,
        allow_dangerous_requests=True,  # scoped to read-mostly research metadata; see README
        return_direct=True,
    )
    try:
        result = chain.invoke({"query": question})
        rows = result.get("result", [])
        return [str(row) for row in rows] if rows else []
    except Exception:
        logger.exception("Graph query failed for question: %s", question)
        return []
