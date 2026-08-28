"""
Node functions for the agent's StateGraph. Each takes the current
AgentState and returns a dict of fields to merge in — the LangGraph
convention for partial state updates.
"""
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from agent.state import AgentState
from agent.tools import graph_search, vector_search
from config import get_settings

logger = logging.getLogger(__name__)


def _llm() -> ChatOllama:
    settings = get_settings()
    return ChatOllama(model=settings.chat_model, base_url=settings.ollama_base_url, temperature=settings.llm_temperature)


class RouteDecision(BaseModel):
    query_type: str = Field(description="One of: semantic, relational, hybrid")
    reasoning: str


def route_query(state: AgentState) -> dict:
    """Classify the question so we don't run both retrieval paths on
    every query — relational questions ('who wrote X', 'what does X cite')
    don't need vector search, and vice versa."""
    router_llm = _llm().with_structured_output(RouteDecision)
    decision = router_llm.invoke([
        SystemMessage(content=(
            "Classify the research question as:\n"
            "- 'semantic': asks about content/findings/explanations within papers -> needs text search\n"
            "- 'relational': asks about authors, citations, methods used, datasets -> needs graph traversal\n"
            "- 'hybrid': needs both\n"
        )),
        HumanMessage(content=state["question"]),
    ])
    logger.info("Routed '%s' -> %s (%s)", state["question"], decision.query_type, decision.reasoning)
    return {"query_type": decision.query_type, "retry_count": state.get("retry_count", 0)}


def retrieve_vector(state: AgentState) -> dict:
    return {"vector_context": vector_search(state["question"])}


def retrieve_graph(state: AgentState) -> dict:
    return {"graph_context": graph_search(state["question"])}


def synthesize(state: AgentState) -> dict:
    """Combine whatever context is available and produce a grounded,
    cited answer. Explicitly told to say when it doesn't know rather
    than fill gaps — this is what the grade_answer node checks for."""
    vector_ctx = "\n".join(state.get("vector_context") or []) or "None retrieved."
    graph_ctx = "\n".join(state.get("graph_context") or []) or "None retrieved."

    prompt = f"""Answer the question using ONLY the context below. Cite paper titles
inline like [Paper Title]. If the context is insufficient, say so explicitly
instead of guessing.

Text context (from paper content):
{vector_ctx}

Graph context (from paper relationship data):
{graph_ctx}

Question: {state['question']}
"""
    response = _llm().invoke(prompt)
    citations = list({
        line.split("]")[0].strip("[")
        for line in state.get("vector_context", [])
        if line.startswith("[")
    })
    return {"answer": response.content, "citations": citations}


class GroundednessCheck(BaseModel):
    is_grounded: bool = Field(description="True if the answer is supported by the given context, or correctly admits insufficient context")


def grade_answer(state: AgentState) -> dict:
    """Self-critique step: catches hallucinated answers before returning
    them, and triggers one retry with broader retrieval instead."""
    grader = _llm().with_structured_output(GroundednessCheck)
    check = grader.invoke([
        SystemMessage(content="Judge whether the ANSWER is actually supported by the CONTEXT, or honestly says it can't answer."),
        HumanMessage(content=f"CONTEXT:\n{state.get('vector_context')}\n{state.get('graph_context')}\n\nANSWER:\n{state['answer']}"),
    ])
    return {"is_grounded": check.is_grounded, "retry_count": state.get("retry_count", 0) + 1}
