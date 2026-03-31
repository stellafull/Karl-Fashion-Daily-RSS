# tests/agents/test_writing.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from deep_agents.schemas import SectionDraft


async def test_writer_produces_section_drafts(sample_state, mock_config):
    from deep_agents.agents.writer import writer_node

    mock_draft = SectionDraft(
        section_id="sec_1",
        content="## 市场概况\n\n2025年中国奢侈品市场规模达3620亿元...",
        citations=[{"claim": "规模达3620亿元", "source_id": "src_001", "url": "https://example.com"}],
        charts_used=[],
        weak_claims=[],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_draft)

    sample_state["facts"] = [{"content": "市场规模3620亿元", "source_id": "src_001", "section_id": "sec_1"}]

    with patch("deep_agents.agents.writer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await writer_node(sample_state, mock_config)

    assert len(result["section_drafts"]) == 1
    assert result["section_drafts"][0]["section_id"] == "sec_1"
    assert "市场概况" in result["section_drafts"][0]["content"]


async def test_synthesizer_produces_full_report(sample_state, mock_config):
    from deep_agents.agents.synthesizer import synthesizer_node

    mock_response = MagicMock()
    mock_response.content = "# 2025年中国奢侈品市场深度研究\n\n## 执行摘要\n..."

    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=mock_response)

    sample_state["section_drafts"] = [
        {"section_id": "sec_1", "content": "## 市场概况\n\n内容...", "citations": [], "charts_used": [], "weak_claims": []}
    ]

    with patch("deep_agents.agents.synthesizer.init_chat_model") as mock_init:
        mock_init.return_value = mock_model
        result = await synthesizer_node(sample_state, mock_config)

    assert len(result["full_report"]) > 0
    assert "执行摘要" in result["full_report"]


async def test_trend_triangulator_updates_report(sample_state, mock_config):
    from deep_agents.agents.trend_triangulator import trend_triangulator_node

    mock_response = MagicMock()
    mock_response.content = "# 修订后报告\n\n## 趋势验证摘要\n..."

    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=mock_response)

    sample_state["full_report"] = "# 原始报告\n\n静奢风趋势持续走强。"

    with patch("deep_agents.agents.trend_triangulator.init_chat_model") as mock_init:
        mock_init.return_value = mock_model
        result = await trend_triangulator_node(sample_state, mock_config)

    assert len(result["full_report"]) > 0
