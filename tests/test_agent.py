"""
Unit tests. All external calls (Ollama, Neo4j, Chroma) are mocked so
these run offline/in CI without any services actually running.
"""
from unittest.mock import MagicMock, patch

import pytest

from agent.graph import _after_grade, _after_route
from agent.state import AgentState


class TestRouting:
    def test_semantic_routes_to_vector_only(self):
        state: AgentState = {"query_type": "semantic"}
        assert _after_route(state) == ["retrieve_vector"]

    def test_relational_routes_to_graph_only(self):
        state: AgentState = {"query_type": "relational"}
        assert _after_route(state) == ["retrieve_graph"]

    def test_hybrid_routes_to_both(self):
        state: AgentState = {"query_type": "hybrid"}
        assert _after_route(state) == ["retrieve_vector", "retrieve_graph"]

    def test_unknown_type_defaults_to_both(self):
        state: AgentState = {"query_type": "garbage"}
        assert set(_after_route(state)) == {"retrieve_vector", "retrieve_graph"}


class TestGradingLoop:
    @patch("agent.graph.get_settings")
    def test_grounded_answer_ends(self, mock_settings):
        mock_settings.return_value.max_retries = 2
        state: AgentState = {"is_grounded": True, "retry_count": 1}
        assert _after_grade(state) == "end"

    @patch("agent.graph.get_settings")
    def test_ungrounded_with_retries_left_loops_back(self, mock_settings):
        mock_settings.return_value.max_retries = 2
        state: AgentState = {"is_grounded": False, "retry_count": 0}
        assert _after_grade(state) == "retry"

    @patch("agent.graph.get_settings")
    def test_ungrounded_out_of_retries_ends(self, mock_settings):
        mock_settings.return_value.max_retries = 2
        state: AgentState = {"is_grounded": False, "retry_count": 2}
        assert _after_grade(state) == "end"


class TestSynthesizeNode:
    @patch("agent.nodes._llm")
    def test_synthesize_extracts_citations_from_vector_context(self, mock_llm_factory):
        mock_response = MagicMock(content="Some answer [Attention Is All You Need]")
        mock_llm_factory.return_value.invoke.return_value = mock_response

        from agent.nodes import synthesize

        state: AgentState = {
            "question": "What is self-attention?",
            "vector_context": ["[Attention Is All You Need] Self-attention relates positions..."],
            "graph_context": [],
        }
        result = synthesize(state)
        assert result["answer"] == "Some answer [Attention Is All You Need]"
        assert "Attention Is All You Need" in result["citations"]

    @patch("agent.nodes._llm")
    def test_synthesize_handles_empty_context_without_crashing(self, mock_llm_factory):
        mock_llm_factory.return_value.invoke.return_value = MagicMock(content="I don't have enough context.")

        from agent.nodes import synthesize

        state: AgentState = {"question": "Anything?", "vector_context": [], "graph_context": []}
        result = synthesize(state)
        assert result["citations"] == []


class TestGraphSearchFailureHandling:
    @patch("agent.tools.Neo4jGraph")
    @patch("agent.tools.GraphCypherQAChain")
    def test_graph_search_returns_empty_list_on_failure_not_exception(self, mock_chain_cls, mock_graph_cls):
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("Neo4j connection refused")
        mock_chain_cls.from_llm.return_value = mock_chain

        from agent.tools import graph_search

        result = graph_search("Who wrote this paper?")
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
