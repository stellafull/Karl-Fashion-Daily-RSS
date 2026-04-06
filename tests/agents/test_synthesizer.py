"""Tests for synthesizer_node."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command


async def test_synthesizer_routes_to_reviewer_for_non_trend(
    sample_state, mock_config
) -> None:
    sample_state["research_type"] = "market_overview"
    sample_state["section_drafts"] = [
        {
            "section_id": "sec_1",
            "content": "内容",
            "citations": [],
            "charts_used": [],
            "weak_claims": [],
        }
    ]
    sample_state["hypothesis_evidence"] = []
    sample_state["contradictions"] = []
    sample_state["sources"] = []

    mock_response = MagicMock()
    mock_response.content = "# 完整报告\n\n内容..."

    with patch("deep_agents.agents.synthesizer.init_chat_model") as m:
        m.return_value.ainvoke = AsyncMock(return_value=mock_response)
        from deep_agents.agents.synthesizer import synthesizer_node

        result = await synthesizer_node(sample_state, mock_config)

    assert isinstance(result, Command)
    assert result.goto == "reviewer"
    assert "full_report" in result.update
    assert "完整报告" in result.update["full_report"]


async def test_synthesizer_routes_to_trend_triangulator_for_trend(
    sample_state, mock_config
) -> None:
    sample_state["research_type"] = "trend_analysis"
    sample_state["section_drafts"] = []
    sample_state["hypothesis_evidence"] = []
    sample_state["contradictions"] = []
    sample_state["sources"] = []

    mock_response = MagicMock()
    mock_response.content = "# 趋势报告"

    with patch("deep_agents.agents.synthesizer.init_chat_model") as m:
        m.return_value.ainvoke = AsyncMock(return_value=mock_response)
        from deep_agents.agents.synthesizer import synthesizer_node

        result = await synthesizer_node(sample_state, mock_config)

    assert isinstance(result, Command)
    assert result.goto == "trend_triangulator"
