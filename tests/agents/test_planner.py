"""Tests for planner_node and outline_reviser_node."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from deep_agents.schemas import ArchitectPlan, Hypothesis, Section, RevisedOutline


async def test_planner_node_returns_correct_keys(mock_config):
    from deep_agents.agents.planner import planner_node
    mock_plan = ArchitectPlan(
        research_type="trend_analysis",
        hypotheses=[Hypothesis(id="h_1", statement="X", evidence_needed=["data"])],
        sections=[Section(id="sec_1", title="T", description="D", search_queries=["q"], priority=1)],
        budget={"max_parallel": 3, "max_searches": 5, "max_deep_reads": 3},
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_plan)
    state = {"research_goal": "了解趋势", "confirmed_constraints": [], "open_dimensions": [], "language": "zh"}
    with patch("deep_agents.agents.planner.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        result = await planner_node(state, mock_config)
    assert result["research_type"] == "trend_analysis"
    assert len(result["sections"]) == 1
    assert result["outline_revision_count"] == 0


async def test_outline_reviser_increments_count(mock_config):
    from deep_agents.agents.outline_reviser import outline_reviser_node
    mock_revised = RevisedOutline(
        sections=[Section(id="sec_1", title="T", description="D", search_queries=["q"], priority=1)]
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_revised)
    state = {
        "research_goal": "了解趋势",
        "sections": [{"id": "sec_1", "title": "T", "description": "D", "search_queries": ["q"], "priority": 1}],
        "hypothesis_evidence": [],
        "outline_revision_count": 0,
    }
    with patch("deep_agents.agents.outline_reviser.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        result = await outline_reviser_node(state, mock_config)
    assert result["outline_revision_count"] == 1
    assert result["outline_status"] == "revised"
