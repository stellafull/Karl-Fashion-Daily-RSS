"""Tests for deep_scout_node, analyst_node, and data_wiz_node."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from deep_agents.schemas import AnalystOutput, DataWizOutput


@pytest.fixture
def section_state():
    return {
        "section_id": "sec_1",
        "section_title": "市场概况",
        "section_description": "规模与增速",
        "search_queries": ["luxury market 2025"],
        "research_goal": "了解市场",
        "hypotheses": [],
        "language": "zh",
        "search_results": [{"raw": "市场规模3620亿元"}],
        "section_facts": [],
        "section_insights": [],
        "section_hypothesis_evidence": [],
        "section_contradictions": [],
        "section_entities": [],
        "missing_info": [],
        "section_data_points": [],
        "section_charts": [],
        "section_time_series": [],
        "section_sources": [],
    }


async def test_analyst_returns_facts(section_state, mock_config):
    from deep_agents.agents.analyst import analyst_node

    mock_out = AnalystOutput(
        section_facts=[{"content": "3620亿元", "source_id": "s1", "importance": "high"}]
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_out)
    with patch("deep_agents.agents.analyst.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        result = await analyst_node(section_state, mock_config)
    assert len(result["section_facts"]) == 1
    assert "section_insights" in result
    assert "section_hypothesis_evidence" in result
    assert "section_contradictions" in result
    assert "section_entities" in result
    assert "missing_info" in result


async def test_data_wiz_returns_data_points(section_state, mock_config):
    from deep_agents.agents.data_wiz import data_wiz_node

    mock_out = DataWizOutput(section_data_points=[{"id": "dp1", "value": 3620}])
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_out)
    with patch("deep_agents.agents.data_wiz.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        result = await data_wiz_node(section_state, mock_config)
    assert len(result["section_data_points"]) == 1
    assert "section_charts" in result
    assert "section_time_series" in result


async def test_deep_scout_exits_on_research_complete(section_state, mock_config):
    """Loop exits cleanly when model calls ResearchComplete tool."""
    from deep_agents.agents.deep_scout import deep_scout_node
    from langchain_core.messages import AIMessage

    mock_ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ResearchComplete",
                "args": {"reason": "research done"},
                "id": "tc1",
                "type": "tool_call",
            }
        ],
    )
    mock_bound_model = MagicMock()
    mock_bound_model.ainvoke = AsyncMock(return_value=mock_ai_msg)
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_bound_model

    with patch("deep_agents.agents.deep_scout.get_all_tools", AsyncMock(return_value=[])):
        with patch("deep_agents.agents.deep_scout.init_chat_model", return_value=mock_model):
            result = await deep_scout_node(section_state, mock_config)

    mock_model.bind_tools.assert_called_once_with([])
    mock_bound_model.ainvoke.assert_awaited_once()
    assert "search_results" in result
    assert "section_sources" in result
    assert isinstance(result["search_results"], list)


async def test_deep_scout_returns_empty_on_error(section_state, mock_config):
    """On exception, node returns empty results without raising."""
    from deep_agents.agents.deep_scout import deep_scout_node

    mock_model = MagicMock()
    mock_model.bind_tools.return_value.ainvoke = AsyncMock(
        side_effect=RuntimeError("search failed")
    )

    with patch("deep_agents.agents.deep_scout.get_all_tools", AsyncMock(return_value=[])):
        with patch("deep_agents.agents.deep_scout.init_chat_model", return_value=mock_model):
            result = await deep_scout_node(section_state, mock_config)
    assert result["search_results"] == []
    assert result["section_sources"] == []
