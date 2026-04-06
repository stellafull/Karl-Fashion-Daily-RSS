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
        "section_entities": [],
        "missing_info": [],
        "section_data_points": [],
        "section_charts": [],
        "section_time_series": [],
        "section_sources": [],
    }

    first_turn = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "web_search",
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
    search_tool.name = "web_search"
    search_tool.ainvoke = AsyncMock(
        return_value={
            "results": [
                {"url": "https://example.com/a", "title": "A", "summary": "summary-a"},
                {"url": "https://example.com/b", "title": "B", "summary": "summary-b"},
            ]
        }
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
        "section_entities": [],
        "missing_info": [],
        "section_data_points": [],
        "section_charts": [],
        "section_time_series": [],
        "section_sources": [],
    }

    first_turn = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "web_search",
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
    search_tool.name = "web_search"
    search_tool.ainvoke = AsyncMock(
        return_value={
            "results": [
                {"url": None, "title": "invalid-none"},
                {"url": "", "title": "invalid-empty"},
                {"url": "   ", "title": "invalid-space"},
                {"url": "https://example.com/valid", "title": "valid"},
            ]
        }
    )

    with patch("deep_agents.agents.deep_scout.get_all_tools", AsyncMock(return_value=[search_tool])):
        with patch("deep_agents.agents.deep_scout.init_chat_model", return_value=mock_model):
            result = await deep_scout_node(state, mock_config)

    assert len(result["section_sources"]) == 1
    assert result["section_sources"][0]["url"] == "https://example.com/valid"
    assert set(result["section_sources"][0].keys()) == {"url", "title", "summary"}


async def test_deep_scout_falls_back_to_content_when_summary_is_non_string(mock_config):
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
        "section_entities": [],
        "missing_info": [],
        "section_data_points": [],
        "section_charts": [],
        "section_time_series": [],
        "section_sources": [],
    }

    first_turn = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "web_search",
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
    search_tool.name = "web_search"
    search_tool.ainvoke = AsyncMock(
        return_value={
            "results": [
                {
                    "url": "https://example.com/valid",
                    "title": "valid",
                    "summary": {"bad": "shape"},
                    "content": "fallback-summary-from-content",
                }
            ]
        }
    )

    with patch("deep_agents.agents.deep_scout.get_all_tools", AsyncMock(return_value=[search_tool])):
        with patch("deep_agents.agents.deep_scout.init_chat_model", return_value=mock_model):
            result = await deep_scout_node(state, mock_config)

    assert len(result["section_sources"]) == 1
    assert result["section_sources"][0]["summary"] == "fallback-summary-from-content"
    assert result["search_results"][0]["sources"][0]["summary"] == "fallback-summary-from-content"


async def test_deep_scout_falls_back_to_content_when_summary_is_blank(mock_config):
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
        "section_entities": [],
        "missing_info": [],
        "section_data_points": [],
        "section_charts": [],
        "section_time_series": [],
        "section_sources": [],
    }

    first_turn = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "web_search",
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
    search_tool.name = "web_search"
    search_tool.ainvoke = AsyncMock(
        return_value={
            "results": [
                {
                    "url": "https://example.com/valid",
                    "title": "valid",
                    "summary": "   ",
                    "content": "fallback-summary-from-content",
                }
            ]
        }
    )

    with patch("deep_agents.agents.deep_scout.get_all_tools", AsyncMock(return_value=[search_tool])):
        with patch("deep_agents.agents.deep_scout.init_chat_model", return_value=mock_model):
            result = await deep_scout_node(state, mock_config)

    assert len(result["section_sources"]) == 1
    assert result["section_sources"][0]["summary"] == "fallback-summary-from-content"
    assert result["search_results"][0]["sources"][0]["summary"] == "fallback-summary-from-content"
