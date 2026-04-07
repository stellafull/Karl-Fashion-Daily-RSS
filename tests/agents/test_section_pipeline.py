"""Tests for section_worker, analyst_node, and data_wiz_node."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
        "section_research": "市场规模3620亿元",
        "section_facts": [],
        "section_insights": [],
        "section_hypothesis_evidence": [],
        "section_contradictions": [],
        "missing_info": [],
        "section_data_points": [],
        "section_charts": [],
    }


async def test_analyst_returns_facts(section_state, mock_config):
    from deep_agents.agents.analyst import analyst_node

    mock_out = AnalystOutput(
        section_facts=[{"content": "3620亿元", "importance": "high"}],
        section_hypothesis_evidence=[
            {
                "hypothesis_statement": "奢侈品市场保持增长",
                "evidence_type": "supports",
                "content": "市场规模同比增长",
            }
        ],
        section_contradictions=[{"claim_a": "线上渗透率提升", "claim_b": "线下门店仍主导"}],
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_out)
    with patch("deep_agents.agents.analyst.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        result = await analyst_node(section_state, mock_config)

    assert result["section_facts"] == [{"content": "3620亿元", "importance": "high"}]
    assert result["section_insights"] == []
    assert result["section_hypothesis_evidence"] == [
        {
            "hypothesis_statement": "奢侈品市场保持增长",
            "evidence_type": "supports",
            "content": "市场规模同比增长",
        }
    ]
    assert result["section_contradictions"] == [
        {"claim_a": "线上渗透率提升", "claim_b": "线下门店仍主导"}
    ]
    assert result["missing_info"] == []
    assert m.call_args.kwargs["disable_streaming"] is True


async def test_data_wiz_returns_data_points(section_state, mock_config):
    from deep_agents.agents.data_wiz import data_wiz_node

    mock_out = DataWizOutput(
        section_data_points=[
            {
                "name": "市场规模",
                "value": 3620,
                "unit": "亿元",
                "year": 2025,
                "category": "market_size",
                "confidence": "high",
            }
        ]
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_out)
    with patch("deep_agents.agents.data_wiz.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        result = await data_wiz_node(section_state, mock_config)

    assert result["section_data_points"] == [
        {
            "name": "市场规模",
            "value": 3620,
            "unit": "亿元",
            "year": 2025,
            "category": "market_size",
            "confidence": "high",
        }
    ]
    assert result["section_charts"] == []
    assert m.call_args.kwargs["disable_streaming"] is True


async def test_analyst_prompt_receives_full_section_research_payload(section_state, mock_config):
    from deep_agents.agents.analyst import analyst_node

    full_payload = (
        "[A](https://example.com/a)\n\n"
        + ("x" * 3500)
    )
    section_state["section_research"] = full_payload

    mock_out = AnalystOutput()
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_out)
    with patch("deep_agents.agents.analyst.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        await analyst_node(section_state, mock_config)

    prompt = mock_chain.ainvoke.await_args.args[0][0].content
    assert "压缩后的章节研究素材" in prompt
    assert full_payload in prompt


async def test_data_wiz_prompt_receives_full_section_research_payload(section_state, mock_config):
    from deep_agents.agents.data_wiz import data_wiz_node

    full_payload = (
        "[A](https://example.com/a)\n\n"
        + ("x" * 3500)
    )
    section_state["section_research"] = full_payload

    mock_out = DataWizOutput()
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_out)
    with patch("deep_agents.agents.data_wiz.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        await data_wiz_node(section_state, mock_config)

    prompt = mock_chain.ainvoke.await_args.args[0][0].content
    assert "压缩后的章节研究素材（含数据）" in prompt
    assert full_payload in prompt


async def test_section_worker_seeds_single_artifact_field_and_tags_outputs(mock_config):
    from deep_agents.graph import section_worker

    worker_state = {
        "section_id": "sec_1",
        "section_title": "市场概况",
        "section_description": "规模与增速",
        "search_queries": ["luxury market 2025"],
        "research_goal": "了解市场",
        "hypotheses": [],
        "language": "zh",
    }

    subgraph = MagicMock()
    subgraph.ainvoke = AsyncMock(
        return_value={
            "section_facts": [{"content": "fact", "importance": "high"}],
            "section_data_points": [{"name": "dp", "value": 1}],
            "section_hypothesis_evidence": [
                {
                    "hypothesis_statement": "h1",
                    "evidence_type": "supports",
                    "content": "evidence",
                }
            ],
            "section_charts": [{"title": "chart"}],
            "section_contradictions": [{"claim_a": "a", "claim_b": "b"}],
        }
    )

    with patch("deep_agents.graph._get_section_subgraph", return_value=subgraph):
        result = await section_worker(worker_state, mock_config)

    section_input = subgraph.ainvoke.await_args.args[0]
    assert section_input["section_research"] == ""
    assert "search_results" not in section_input

    assert result == {
        "facts": [{"content": "fact", "importance": "high", "section_id": "sec_1"}],
        "data_points": [{"name": "dp", "value": 1, "section_id": "sec_1"}],
        "hypothesis_evidence": [
            {
                "hypothesis_statement": "h1",
                "evidence_type": "supports",
                "content": "evidence",
                "section_id": "sec_1",
            }
        ],
        "charts": [{"title": "chart", "section_id": "sec_1"}],
        "contradictions": [{"claim_a": "a", "claim_b": "b", "section_id": "sec_1"}],
    }
