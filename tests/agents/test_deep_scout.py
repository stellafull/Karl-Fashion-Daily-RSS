from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage


def _section_state() -> dict:
    return {
        "section_id": "sec_1",
        "section_title": "市场概况",
        "section_description": "规模与增速",
        "search_queries": ["luxury market 2025"],
        "research_goal": "了解市场",
        "hypotheses": [],
        "language": "zh",
        "section_research": "",
        "section_facts": [],
        "section_insights": [],
        "section_hypothesis_evidence": [],
        "section_contradictions": [],
        "missing_info": [],
        "section_data_points": [],
        "section_charts": [],
    }


async def test_deep_scout_exits_on_research_complete_with_empty_artifact(mock_config):
    from deep_agents.agents.deep_scout import deep_scout_node

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
            result = await deep_scout_node(_section_state(), mock_config)

    mock_model.bind_tools.assert_called_once_with([])
    mock_bound_model.ainvoke.assert_awaited_once()
    assert result["section_research"] == ""
    assert "search_results" not in result


async def test_deep_scout_returns_single_compressed_artifact_after_search(mock_config):
    from deep_agents.agents.deep_scout import deep_scout_node

    first_turn = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "tavily_search",
                "args": {"queries": ["luxury market 2025"]},
                "id": "tc1",
                "type": "tool_call",
            }
        ],
    )
    second_turn = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ResearchComplete",
                "args": {"reason": "done"},
                "id": "tc2",
                "type": "tool_call",
            }
        ],
    )

    mock_bound_model = MagicMock()
    mock_bound_model.ainvoke = AsyncMock(side_effect=[first_turn, second_turn])
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_bound_model

    search_tool = MagicMock()
    search_tool.name = "tavily_search"
    search_tool.ainvoke = AsyncMock(
        return_value=[
            {"url": "https://example.com/a", "title": "A", "summary": "summary-a"}
        ]
    )

    with patch("deep_agents.agents.deep_scout.get_all_tools", AsyncMock(return_value=[search_tool])):
        with patch("deep_agents.agents.deep_scout.init_chat_model", return_value=mock_model):
            with patch(
                "deep_agents.agents.deep_scout.compress_search",
                new=AsyncMock(return_value="[A](https://example.com/a)"),
                create=True,
            ):
                result = await deep_scout_node(_section_state(), mock_config)

    assert isinstance(result["section_research"], str)
    assert result["section_research"]
    assert "https://example.com/a" in result["section_research"]
    assert "search_results" not in result


async def test_deep_scout_does_not_persist_reflection_only_turns(mock_config):
    from deep_agents.agents.deep_scout import deep_scout_node

    first_turn = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "think_tool",
                "args": {"reflection": "当前还没有足够证据，需要继续检索。"},
                "id": "tc-think",
                "type": "tool_call",
            }
        ],
    )
    second_turn = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ResearchComplete",
                "args": {"reason": "done"},
                "id": "tc-complete",
                "type": "tool_call",
            }
        ],
    )

    mock_bound_model = MagicMock()
    mock_bound_model.ainvoke = AsyncMock(side_effect=[first_turn, second_turn])
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_bound_model

    think_tool = MagicMock()
    think_tool.name = "think_tool"
    think_tool.ainvoke = AsyncMock(
        return_value="Reflection recorded: 当前还没有足够证据，需要继续检索。"
    )

    with patch("deep_agents.agents.deep_scout.get_all_tools", AsyncMock(return_value=[think_tool])):
        with patch("deep_agents.agents.deep_scout.init_chat_model", return_value=mock_model):
            result = await deep_scout_node(_section_state(), mock_config)

    assert result["section_research"] == ""
    assert "search_results" not in result


async def test_deep_scout_raises_on_model_failure(mock_config):
    from deep_agents.agents.deep_scout import deep_scout_node

    mock_model = MagicMock()
    mock_model.bind_tools.return_value.ainvoke = AsyncMock(
        side_effect=RuntimeError("search failed")
    )

    with patch("deep_agents.agents.deep_scout.get_all_tools", AsyncMock(return_value=[])):
        with patch("deep_agents.agents.deep_scout.init_chat_model", return_value=mock_model):
            with pytest.raises(RuntimeError, match="search failed"):
                await deep_scout_node(_section_state(), mock_config)


async def test_deep_scout_raises_when_tool_loading_fails(mock_config):
    from deep_agents.agents.deep_scout import deep_scout_node

    with patch(
        "deep_agents.agents.deep_scout.get_all_tools",
        AsyncMock(side_effect=RuntimeError("tool registry unavailable")),
    ):
        with pytest.raises(RuntimeError, match="tool registry unavailable"):
            await deep_scout_node(_section_state(), mock_config)
