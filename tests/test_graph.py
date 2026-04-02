"""Tests for graph topology and section_pipeline_node gather behavior."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.graph import END
from langgraph.types import Command


def test_graph_compiles_without_error() -> None:
    from deep_agents.graph import build_research_graph

    graph = build_research_graph()
    assert graph is not None


def test_section_subgraph_compiles() -> None:
    from deep_agents.graph import build_section_subgraph

    sg = build_section_subgraph()
    assert sg is not None


def test_no_routing_functions_exported() -> None:
    """Routing functions are removed — they should not be importable from graph."""
    import deep_agents.graph as g

    for name in (
        "route_after_clarify",
        "fan_out_sections",
        "route_after_collection",
        "route_after_synthesis",
        "route_after_review",
    ):
        assert not hasattr(g, name), f"{name} should be removed from graph.py"


async def test_section_pipeline_node_merges_results() -> None:
    """section_pipeline_node gathers results from all sections and returns Command."""
    from deep_agents.graph import section_pipeline_node

    section_result = {
        "section_facts": [{"content": "fact1", "section_id": "sec_1"}],
        "section_data_points": [{"id": "dp1"}],
        "section_hypothesis_evidence": [],
        "section_charts": [],
        "section_insights": ["insight1"],
        "section_contradictions": [],
        "section_sources": [{"url": "https://example.com"}],
        "missing_info": ["question1"],
    }

    state = {
        "sections": [
            {
                "id": "sec_1",
                "title": "市场概况",
                "description": "规模",
                "search_queries": ["q1"],
                "priority": 1,
            }
        ],
        "research_goal": "了解市场",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = AsyncMock(return_value=section_result)

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        result = await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})

    assert isinstance(result, Command)
    assert result.goto == "lead_writer"
    assert len(result.update["facts"]) == 1
    assert len(result.update["sources"]) == 1
    assert result.update["insights"][0]["section_id"] == "sec_1"
    assert result.update["open_questions"][0]["section_id"] == "sec_1"
    assert result.update["failed_sections"] == []


async def test_section_pipeline_node_handles_failed_section() -> None:
    """A section that raises an exception is recorded in failed_sections; others succeed."""
    from deep_agents.graph import section_pipeline_node

    good_result = {
        "section_facts": [{"content": "fact from sec_1"}],
        "section_data_points": [],
        "section_hypothesis_evidence": [],
        "section_charts": [],
        "section_insights": [],
        "section_contradictions": [],
        "section_sources": [],
        "missing_info": [],
    }

    call_count = 0

    async def mock_ainvoke(inp, config=None):
        nonlocal call_count
        call_count += 1
        if inp["section_id"] == "sec_2":
            raise RuntimeError("LLM failed")
        return good_result

    state = {
        "sections": [
            {
                "id": "sec_1",
                "title": "市场概况",
                "description": "规模",
                "search_queries": ["q1"],
                "priority": 1,
            },
            {
                "id": "sec_2",
                "title": "竞争格局",
                "description": "品牌",
                "search_queries": ["q2"],
                "priority": 2,
            },
        ],
        "research_goal": "了解市场",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = mock_ainvoke

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        result = await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})

    assert isinstance(result, Command)
    assert "sec_2" in result.update["failed_sections"]
    assert len(result.update["facts"]) == 1


async def test_section_pipeline_routes_to_outline_reviser_when_refuted() -> None:
    """section_pipeline routes to outline_reviser when ≥2 hypotheses are refuted and count < 1."""
    from deep_agents.graph import section_pipeline_node

    refuted_evidence = [
        {"evidence_type": "refutes", "hypothesis_id": "h1"},
        {"evidence_type": "refutes", "hypothesis_id": "h2"},
    ]
    section_result = {
        "section_facts": [],
        "section_data_points": [],
        "section_hypothesis_evidence": refuted_evidence,
        "section_charts": [],
        "section_insights": [],
        "section_contradictions": [],
        "section_sources": [],
        "missing_info": [],
    }

    state = {
        "sections": [
            {"id": "sec_1", "title": "T", "description": "D", "search_queries": ["q"], "priority": 1}
        ],
        "research_goal": "研究",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = AsyncMock(return_value=section_result)

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        result = await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})

    assert result.goto == "outline_reviser"


async def test_section_pipeline_does_not_route_to_outline_reviser_when_count_at_1() -> None:
    """After one outline revision (count=1), re-outline is blocked even with refuted hypotheses."""
    from deep_agents.graph import section_pipeline_node

    refuted_evidence = [
        {"evidence_type": "refutes", "hypothesis_id": "h1"},
        {"evidence_type": "refutes", "hypothesis_id": "h2"},
    ]
    section_result = {
        "section_facts": [],
        "section_data_points": [],
        "section_hypothesis_evidence": refuted_evidence,
        "section_charts": [],
        "section_insights": [],
        "section_contradictions": [],
        "section_sources": [],
        "missing_info": [],
    }

    state = {
        "sections": [
            {"id": "sec_1", "title": "T", "description": "D", "search_queries": ["q"], "priority": 1}
        ],
        "research_goal": "研究",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 1,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = AsyncMock(return_value=section_result)

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        result = await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})

    assert result.goto == "lead_writer"
