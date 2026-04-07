"""Tests for deep_scout_node, analyst_node, and data_wiz_node."""
import json
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
        "missing_info": [],
        "section_data_points": [],
        "section_charts": [],
        "section_sources": [],
    }


async def test_analyst_returns_facts(section_state, mock_config):
    from deep_agents.agents.analyst import analyst_node

    mock_out = AnalystOutput(
        section_facts=[
            {
                "content": "3620亿元",
                "source_url": "https://example.com/market-size",
                "importance": "high",
            }
        ],
        section_hypothesis_evidence=[
            {
                "hypothesis_statement": "奢侈品市场保持增长",
                "evidence_type": "supports",
                "content": "市场规模同比增长",
                "source_url": "https://example.com/hypothesis-evidence",
            }
        ],
        section_contradictions=[
            {
                "claim_a": "线上渗透率提升",
                "claim_b": "线下门店仍主导",
                "source_url_a": "https://example.com/claim-a",
                "source_url_b": "https://example.com/claim-b",
            }
        ],
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_out)
    with patch("deep_agents.agents.analyst.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        result = await analyst_node(section_state, mock_config)
    assert len(result["section_facts"]) == 1
    assert result["section_facts"][0]["source_url"] == "https://example.com/market-size"
    assert "source_id" not in result["section_facts"][0]
    assert "section_insights" in result
    assert "section_hypothesis_evidence" in result
    assert (
        result["section_hypothesis_evidence"][0]["hypothesis_statement"]
        == "奢侈品市场保持增长"
    )
    assert (
        result["section_hypothesis_evidence"][0]["source_url"]
        == "https://example.com/hypothesis-evidence"
    )
    assert "hypothesis_id" not in result["section_hypothesis_evidence"][0]
    assert "source_id" not in result["section_hypothesis_evidence"][0]
    assert "section_contradictions" in result
    assert (
        result["section_contradictions"][0]["source_url_a"]
        == "https://example.com/claim-a"
    )
    assert (
        result["section_contradictions"][0]["source_url_b"]
        == "https://example.com/claim-b"
    )
    assert "source_id_a" not in result["section_contradictions"][0]
    assert "source_id_b" not in result["section_contradictions"][0]
    assert "missing_info" in result
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
                "source_url": "https://example.com/data-point",
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
    assert len(result["section_data_points"]) == 1
    assert (
        result["section_data_points"][0]["source_url"]
        == "https://example.com/data-point"
    )
    assert "id" not in result["section_data_points"][0]
    assert "section_charts" in result
    assert m.call_args.kwargs["disable_streaming"] is True


async def test_analyst_prompt_receives_full_search_results_payload(section_state, mock_config):
    from deep_agents.agents.analyst import analyst_node

    full_payload = (
        '[{"url":"https://example.com/a","title":"A","summary":"'
        + ("x" * 3500)
        + '"}]'
    )
    section_state["search_results"] = [{"raw": full_payload, "sources": []}]

    mock_out = AnalystOutput()
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_out)
    with patch("deep_agents.agents.analyst.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        await analyst_node(section_state, mock_config)

    prompt = mock_chain.ainvoke.await_args.args[0][0].content
    search_results_json = prompt.split("搜索结果：\n", 1)[1].split("\n\n你是时尚行业研究分析师", 1)[0]
    assert json.loads(search_results_json)[0]["raw"] == full_payload


async def test_data_wiz_prompt_receives_full_search_results_payload(section_state, mock_config):
    from deep_agents.agents.data_wiz import data_wiz_node

    full_payload = (
        '[{"url":"https://example.com/a","title":"A","summary":"'
        + ("x" * 3500)
        + '"}]'
    )
    section_state["search_results"] = [{"raw": full_payload, "sources": []}]

    mock_out = DataWizOutput()
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_out)
    with patch("deep_agents.agents.data_wiz.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        await data_wiz_node(section_state, mock_config)

    prompt = mock_chain.ainvoke.await_args.args[0][0].content
    search_results_json = prompt.split("搜索结果（含数据）：\n", 1)[1].split("\n\n你是时尚行业数据分析师", 1)[0]
    assert json.loads(search_results_json)[0]["raw"] == full_payload


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


async def test_deep_scout_raises_on_model_failure(section_state, mock_config):
    """Section collection should fail fast when the model invocation fails."""
    from deep_agents.agents.deep_scout import deep_scout_node

    mock_model = MagicMock()
    mock_model.bind_tools.return_value.ainvoke = AsyncMock(
        side_effect=RuntimeError("search failed")
    )

    with patch("deep_agents.agents.deep_scout.get_all_tools", AsyncMock(return_value=[])):
        with patch("deep_agents.agents.deep_scout.init_chat_model", return_value=mock_model):
            with pytest.raises(RuntimeError, match="search failed"):
                await deep_scout_node(section_state, mock_config)


async def test_deep_scout_raises_when_tool_loading_fails(section_state, mock_config):
    """Section collection should fail fast when tool loading fails."""
    from deep_agents.agents.deep_scout import deep_scout_node

    with patch(
        "deep_agents.agents.deep_scout.get_all_tools",
        AsyncMock(side_effect=RuntimeError("tool registry unavailable")),
    ):
        with pytest.raises(RuntimeError, match="tool registry unavailable"):
            await deep_scout_node(section_state, mock_config)


async def test_section_pipeline_tags_section_scoped_evidence_and_unique_sources() -> None:
    from deep_agents.graph import section_pipeline_node

    async def mock_ainvoke(inp, config=None):
        section_id = inp["section_id"]
        return {
            "section_facts": [
                {
                    "content": f"fact-{section_id}",
                    "source_url": f"https://example.com/{section_id}",
                }
            ],
            "section_data_points": [
                {
                    "name": f"dp-{section_id}",
                    "value": 1,
                    "source_url": f"https://example.com/{section_id}",
                }
            ],
            "section_hypothesis_evidence": [
                {
                    "hypothesis_statement": "h1",
                    "evidence_type": "supports",
                    "content": "evidence",
                    "source_url": f"https://example.com/{section_id}",
                }
            ],
            "section_charts": [{"id": f"chart-{section_id}"}],
            "section_insights": [f"insight-{section_id}"],
            "section_contradictions": [
                {
                    "claim_a": "a",
                    "claim_b": "b",
                    "source_url_a": f"https://example.com/{section_id}",
                    "source_url_b": f"https://example.com/{section_id}",
                }
            ],
            "section_sources": [
                {
                    "url": f"https://example.com/{section_id}",
                    "title": section_id,
                    "summary": "summary",
                }
            ],
            "missing_info": [f"missing-{section_id}"],
        }

    state = {
        "sections": [
            {"id": "sec_1", "title": "A", "description": "A", "search_queries": ["q1"], "priority": 1},
            {"id": "sec_2", "title": "B", "description": "B", "search_queries": ["q2"], "priority": 2},
        ],
        "research_goal": "研究",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = mock_ainvoke

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        result = await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})

    from langgraph.types import Overwrite

    assert isinstance(result.update["facts"], Overwrite)
    assert isinstance(result.update["data_points"], Overwrite)
    assert isinstance(result.update["hypothesis_evidence"], Overwrite)
    assert isinstance(result.update["charts"], Overwrite)
    assert isinstance(result.update["contradictions"], Overwrite)
    assert isinstance(result.update["sources"], Overwrite)

    facts = result.update["facts"].value
    data_points = result.update["data_points"].value
    evidence = result.update["hypothesis_evidence"].value
    charts = result.update["charts"].value
    contradictions = result.update["contradictions"].value
    sources = result.update["sources"].value

    assert all(item["section_id"] in {"sec_1", "sec_2"} for item in facts)
    assert all(item["section_id"] in {"sec_1", "sec_2"} for item in data_points)
    assert all(item["section_id"] in {"sec_1", "sec_2"} for item in evidence)
    assert all(item["section_id"] in {"sec_1", "sec_2"} for item in charts)
    assert all(item["section_id"] in {"sec_1", "sec_2"} for item in contradictions)
    assert all(item["section_id"] in {"sec_1", "sec_2"} for item in sources)

    assert all("source_id" not in item for item in facts)
    assert all("source_id" not in item for item in data_points)
    assert all("source_id" not in item for item in evidence)
    assert all("source_id_a" not in item and "source_id_b" not in item for item in contradictions)
    assert all("source_id" not in source for source in sources)


async def test_section_pipeline_second_pass_uses_overwrite() -> None:
    from deep_agents.graph import section_pipeline_node
    from langgraph.types import Overwrite

    section_result = {
        "section_facts": [{"content": "fresh", "source_url": "https://example.com/fresh"}],
        "section_data_points": [
            {"name": "fresh-dp", "value": 1, "source_url": "https://example.com/fresh"}
        ],
        "section_hypothesis_evidence": [],
        "section_charts": [],
        "section_contradictions": [],
        "section_sources": [{"url": "https://example.com/fresh", "title": "fresh", "summary": ""}],
        "missing_info": [],
    }
    state = {
        "sections": [
            {"id": "sec_1", "title": "A", "description": "A", "search_queries": ["q1"], "priority": 1}
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

    # Overwrite replaces stale data entirely
    assert isinstance(result.update["facts"], Overwrite)
    assert len(result.update["facts"].value) == 1
    assert result.update["facts"].value[0]["content"] == "fresh"


async def test_section_pipeline_rejects_empty_sections() -> None:
    from deep_agents.graph import section_pipeline_node

    state = {
        "sections": [],
        "research_goal": "研究",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    with pytest.raises(ValueError, match="Planner produced no sections"):
        await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})
