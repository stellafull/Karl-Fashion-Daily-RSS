# tests/test_graph.py
import pytest
from unittest.mock import MagicMock


def test_graph_compiles_without_error():
    """The compiled graph must be importable and callable."""
    from deep_agents.graph import build_research_graph
    graph = build_research_graph()
    assert graph is not None


def test_section_subgraph_compiles():
    from deep_agents.graph import build_section_subgraph
    sg = build_section_subgraph()
    assert sg is not None


def test_route_after_clarify_needs_clarification():
    from deep_agents.graph import route_after_clarify
    from langgraph.graph import END

    state = {"need_clarification": True}
    assert route_after_clarify(state) == END


def test_route_after_clarify_no_clarification():
    from deep_agents.graph import route_after_clarify

    state = {"need_clarification": False}
    assert route_after_clarify(state) == "planner"


def test_fan_out_sections_creates_sends():
    from deep_agents.graph import fan_out_sections
    from langgraph.types import Send

    state = {
        "sections": [
            {"id": "sec_1", "title": "市场概况", "description": "规模", "search_queries": ["query1"], "priority": 1},
            {"id": "sec_2", "title": "竞争格局", "description": "品牌", "search_queries": ["query2"], "priority": 2},
        ],
        "research_goal": "了解市场",
        "hypotheses": [],
        "budget": {"max_searches": 5},
        "language": "zh",
    }
    sends = fan_out_sections(state)
    assert len(sends) == 2
    assert all(isinstance(s, Send) for s in sends)
    assert sends[0].node == "section_pipeline"


def test_route_after_review_fail_triggers_reviser():
    from deep_agents.graph import route_after_review

    state = {
        "review_result": {"verdict": "fail", "quality_score": 5},
        "revision_count": 0,
    }
    assert route_after_review(state) == "reviser"


def test_route_after_review_pass_goes_to_final_check():
    from deep_agents.graph import route_after_review

    state = {
        "review_result": {"verdict": "pass", "quality_score": 8},
        "revision_count": 0,
    }
    assert route_after_review(state) == "final_check"


def test_route_after_review_max_revisions_goes_to_final_check():
    from deep_agents.graph import route_after_review

    state = {
        "review_result": {"verdict": "fail", "quality_score": 4},
        "revision_count": 2,
    }
    assert route_after_review(state) == "final_check"


def test_route_after_synthesis_trend_analysis():
    from deep_agents.graph import route_after_synthesis

    state = {"research_type": "trend_analysis"}
    assert route_after_synthesis(state) == "trend_triangulator"


def test_route_after_synthesis_non_trend():
    from deep_agents.graph import route_after_synthesis

    state = {"research_type": "brand_analysis"}
    assert route_after_synthesis(state) == "reviewer"
