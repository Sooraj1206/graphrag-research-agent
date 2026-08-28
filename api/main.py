"""
Thin HTTP layer over the agent. Run with:
    uvicorn api.main:app --reload
"""
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.graph import agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GraphRAG Research Assistant", version="0.1.0")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    query_type: str
    is_grounded: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    try:
        result = agent.invoke({"question": request.question})
    except Exception:
        logger.exception("Agent invocation failed for question: %s", request.question)
        raise HTTPException(status_code=500, detail="Internal error processing query")

    return QueryResponse(
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        query_type=result.get("query_type", "unknown"),
        is_grounded=result.get("is_grounded", False),
    )
