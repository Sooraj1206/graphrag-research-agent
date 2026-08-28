"""
Shared state passed between LangGraph nodes. Kept as a single TypedDict
so every node's signature is (state) -> partial state update, which is
what LangGraph expects.
"""
from typing import Literal, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    query_type: Literal["semantic", "relational", "hybrid"]
    vector_context: list[str]
    graph_context: list[str]
    citations: list[str]
    answer: str
    is_grounded: bool
    retry_count: int
