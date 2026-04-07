from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage


async def test_deep_scout_emits_canonical_sources_only(mock_config):
    from deep_agents.agents.deep_scout import deep_scout_node

    state = {
        "section_id": "sec_1",
        "section_title": "市场概况",
        "section_description": "规模与增速",
        "search_queries": ["luxury market 2025"],
        "research_goal": "了解市场",
        "hypotheses": [],
        "language": "zh",
        "search_results": [],
        "section_facts": [],
        "section_insights": [],
        "section_hypothesis_evidence": [],
        "section_contradictions": [],
        "missing_info": [],
        "section_data_points": [],
        "section_charts": [],
        "section_sources": [],
    }

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
            {"url": "https://example.com/a", "title": "A", "summary": "summary-a"},
            {"url": "https://example.com/b", "title": "B", "summary": "summary-b"},
        ]
    )

    with patch("deep_agents.agents.deep_scout.get_all_tools", AsyncMock(return_value=[search_tool])):
        with patch("deep_agents.agents.deep_scout.init_chat_model", return_value=mock_model):
            result = await deep_scout_node(state, mock_config)

    assert result["search_results"]
    assert "sources" in result["search_results"][0]
    assert len(result["search_results"][0]["sources"]) == 2

    assert len(result["section_sources"]) == 2
    for source in result["section_sources"]:
        assert set(source.keys()) == {"url", "title", "summary"}
        assert isinstance(source["url"], str)
        assert isinstance(source["title"], str)
        assert isinstance(source["summary"], str)
        assert "source_id" not in source
        assert "credibility_score" not in source
        assert "section_id" not in source

    for source in result["search_results"][0]["sources"]:
        assert set(source.keys()) == {"url", "title", "summary"}
        assert "source_id" not in source
        assert "credibility_score" not in source
        assert "section_id" not in source


async def test_deep_scout_skips_non_string_or_blank_urls(mock_config):
    from deep_agents.agents.deep_scout import deep_scout_node

    state = {
        "section_id": "sec_1",
        "section_title": "市场概况",
        "section_description": "规模与增速",
        "search_queries": ["luxury market 2025"],
        "research_goal": "了解市场",
        "hypotheses": [],
        "language": "zh",
        "search_results": [],
        "section_facts": [],
        "section_insights": [],
        "section_hypothesis_evidence": [],
        "section_contradictions": [],
        "missing_info": [],
        "section_data_points": [],
        "section_charts": [],
        "section_sources": [],
    }

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
            {"url": None, "title": "invalid-none"},
            {"url": "", "title": "invalid-empty"},
            {"url": "   ", "title": "invalid-space"},
            {"url": "https://example.com/valid", "title": "valid", "summary": "ok"},
        ]
    )

    with patch("deep_agents.agents.deep_scout.get_all_tools", AsyncMock(return_value=[search_tool])):
        with patch("deep_agents.agents.deep_scout.init_chat_model", return_value=mock_model):
            result = await deep_scout_node(state, mock_config)

    assert len(result["section_sources"]) == 1
    assert result["section_sources"][0]["url"] == "https://example.com/valid"
    assert set(result["section_sources"][0].keys()) == {"url", "title", "summary"}


async def test_deep_scout_executes_search_results_before_honoring_research_complete(
    mock_config,
):
    from deep_agents.agents.deep_scout import deep_scout_node

    state = {
        "section_id": "sec_1",
        "section_title": "市场概况",
        "section_description": "规模与增速",
        "search_queries": ["luxury market 2025"],
        "research_goal": "了解市场",
        "hypotheses": [],
        "language": "zh",
        "search_results": [],
        "section_facts": [],
        "section_insights": [],
        "section_hypothesis_evidence": [],
        "section_contradictions": [],
        "missing_info": [],
        "section_data_points": [],
        "section_charts": [],
        "section_sources": [],
    }

    only_turn = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ResearchComplete",
                "args": {"reason": "done"},
                "id": "tc-complete",
                "type": "tool_call",
            },
            {
                "name": "tavily_search",
                "args": {"queries": ["luxury market 2025"]},
                "id": "tc-search",
                "type": "tool_call",
            },
        ],
    )

    mock_bound_model = MagicMock()
    mock_bound_model.ainvoke = AsyncMock(return_value=only_turn)
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
            result = await deep_scout_node(state, mock_config)

    search_tool.ainvoke.assert_awaited_once()
    assert len(result["section_sources"]) == 1
    assert result["section_sources"][0]["url"] == "https://example.com/a"


async def test_deep_scout_does_not_treat_reflection_as_search_results(mock_config):
    from deep_agents.agents.deep_scout import deep_scout_node

    state = {
        "section_id": "sec_1",
        "section_title": "市场概况",
        "section_description": "规模与增速",
        "search_queries": ["luxury market 2025"],
        "research_goal": "了解市场",
        "hypotheses": [],
        "language": "zh",
        "search_results": [],
        "section_facts": [],
        "section_insights": [],
        "section_hypothesis_evidence": [],
        "section_contradictions": [],
        "missing_info": [],
        "section_data_points": [],
        "section_charts": [],
        "section_sources": [],
    }

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
    think_tool.ainvoke = AsyncMock(return_value="Reflection recorded: 当前还没有足够证据，需要继续检索。")

    with patch("deep_agents.agents.deep_scout.get_all_tools", AsyncMock(return_value=[think_tool])):
        with patch("deep_agents.agents.deep_scout.init_chat_model", return_value=mock_model):
            result = await deep_scout_node(state, mock_config)

    assert result["search_results"] == []
    assert result["section_sources"] == []


async def test_deep_scout_preserves_full_tool_payload_for_followup_reasoning(mock_config):
    from deep_agents.agents.deep_scout import deep_scout_node

    state = {
        "section_id": "sec_1",
        "section_title": "市场概况",
        "section_description": "规模与增速",
        "search_queries": ["luxury market 2025"],
        "research_goal": "了解市场",
        "hypotheses": [],
        "language": "zh",
        "search_results": [],
        "section_facts": [],
        "section_insights": [],
        "section_hypothesis_evidence": [],
        "section_contradictions": [],
        "missing_info": [],
        "section_data_points": [],
        "section_charts": [],
        "section_sources": [],
    }

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
    second_turn = AIMessage(content="", tool_calls=[])

    calls = []

    async def model_ainvoke(messages):
        calls.append(list(messages))
        if len(calls) == 1:
            return first_turn
        return second_turn

    mock_bound_model = MagicMock()
    mock_bound_model.ainvoke = AsyncMock(side_effect=model_ainvoke)
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_bound_model

    raw_result = [
        {
            "url": "https://example.com/a",
            "title": "A",
            "summary": "x" * 3500,
        }
    ]
    search_tool = MagicMock()
    search_tool.name = "tavily_search"
    search_tool.ainvoke = AsyncMock(return_value=raw_result)

    with patch("deep_agents.agents.deep_scout.get_all_tools", AsyncMock(return_value=[search_tool])):
        with patch("deep_agents.agents.deep_scout.init_chat_model", return_value=mock_model):
            await deep_scout_node(state, mock_config)

    assert len(calls) == 2
    assert calls[1][-1].content == (
        '[{"url": "https://example.com/a", "title": "A", "summary": "'
        + ("x" * 3500)
        + '"}]'
    )


async def test_deep_scout_keeps_full_search_results_for_downstream_nodes(mock_config):
    from deep_agents.agents.deep_scout import deep_scout_node

    state = {
        "section_id": "sec_1",
        "section_title": "市场概况",
        "section_description": "规模与增速",
        "search_queries": ["luxury market 2025"],
        "research_goal": "了解市场",
        "hypotheses": [],
        "language": "zh",
        "search_results": [],
        "section_facts": [],
        "section_insights": [],
        "section_hypothesis_evidence": [],
        "section_contradictions": [],
        "missing_info": [],
        "section_data_points": [],
        "section_charts": [],
        "section_sources": [],
    }

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

    raw_result = [
        {
            "url": "https://example.com/a",
            "title": "A",
            "summary": "x" * 3500,
        }
    ]
    search_tool = MagicMock()
    search_tool.name = "tavily_search"
    search_tool.ainvoke = AsyncMock(return_value=raw_result)

    with patch("deep_agents.agents.deep_scout.get_all_tools", AsyncMock(return_value=[search_tool])):
        with patch("deep_agents.agents.deep_scout.init_chat_model", return_value=mock_model):
            result = await deep_scout_node(state, mock_config)

    assert result["search_results"][0]["raw"] == (
        '[{"url": "https://example.com/a", "title": "A", "summary": "'
        + ("x" * 3500)
        + '"}]'
    )
