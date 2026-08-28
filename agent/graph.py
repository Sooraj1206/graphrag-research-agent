"""
Wires the nodes into a StateGraph:

           START
             |
        route_query
        /    |     \\
  retrieve_v  |   retrieve_g       (conditional: semantic / relational / hybrid)
        \\    |     /
          synthesize
             |
        grade_answer
          /       \\
   grounded      not grounded & retries left -> back to route_query
        |               \\
       END          not grounded & out of retries -> END (answer flagged)
"""
from config import get_settings
from langgraph.graph import END, START, StateGraph

from agent.nodes import grade_answer, retrieve_graph, retrieve_vector, route_query, synthesize
from agent.state import AgentState


def _after_route(state: AgentState) -> list[str]:
    routes = {
        "semantic": ["retrieve_vector"],
        "relational": ["retrieve_graph"],
        "hybrid": ["retrieve_vector", "retrieve_graph"],
    }
    return routes.get(state["query_type"], ["retrieve_vector", "retrieve_graph"])


def _after_grade(state: AgentState) -> str:
    settings = get_settings()
    if state.get("is_grounded"):
        return "end"
    if state.get("retry_count", 0) >= settings.max_retries:
        return "end"
    return "retry"


def build_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("route_query", route_query)
    workflow.add_node("retrieve_vector", retrieve_vector)
    workflow.add_node("retrieve_graph", retrieve_graph)
    workflow.add_node("synthesize", synthesize)
    workflow.add_node("grade_answer", grade_answer)

    workflow.add_edge(START, "route_query")
    workflow.add_conditional_edges("route_query", _after_route, ["retrieve_vector", "retrieve_graph"])
    workflow.add_edge("retrieve_vector", "synthesize")
    workflow.add_edge("retrieve_graph", "synthesize")
    workflow.add_edge("synthesize", "grade_answer")
    workflow.add_conditional_edges("grade_answer", _after_grade, {"end": END, "retry": "route_query"})

    return workflow.compile()


# Module-level singleton — compiling the graph is cheap but avoids
# recompiling per-request in the API layer.
agent = build_agent()
