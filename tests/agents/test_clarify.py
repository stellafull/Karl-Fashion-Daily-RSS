"""Tests for clarify_node — the entry point of the research graph."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import END
from langgraph.types import Command

from deep_agents.agents.clarify import clarify_node
from deep_agents.schemas import ResearchBrief


async def test_clarify_no_clarification_returns_command_to_planner(
    mock_config,
) -> None:
    """When need_clarification=False, node returns Command(goto='planner') with research fields."""
    mock_brief = ResearchBrief(
        need_clarification=False,
        research_goal="我想了解2025年春夏女装色彩趋势",
        confirmed_constraints=["女装", "春夏"],
        open_dimensions=["价格段", "地区"],
        language="zh",
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_brief)

    with patch("deep_agents.agents.clarify.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = (
            mock_chain
        )
        result = await clarify_node(
            {
                "messages": [HumanMessage(content="请研究2025年春夏女装色彩趋势")],
                "object_context": None,
            },
            mock_config,
        )

    assert isinstance(result, Command)
    assert result.goto == "planner"
    assert result.update["need_clarification"] is False
    assert result.update["research_goal"] == "我想了解2025年春夏女装色彩趋势"
    assert result.update["confirmed_constraints"] == ["女装", "春夏"]
    assert result.update["language"] == "zh"
    assert m.call_args.kwargs["disable_streaming"] is True


async def test_clarify_needs_clarification_returns_command_to_end(
    mock_config,
) -> None:
    """When need_clarification=True, node returns Command(goto=END)."""
    mock_brief = ResearchBrief(
        need_clarification=True,
        clarification_question="您希望关注哪个价格段的品牌？",
        language="zh",
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_brief)

    with patch("deep_agents.agents.clarify.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = (
            mock_chain
        )
        result = await clarify_node(
            {
                "messages": [HumanMessage(content="帮我研究时尚品牌")],
                "object_context": None,
            },
            mock_config,
        )

    assert isinstance(result, Command)
    assert result.goto == END
    assert result.update["need_clarification"] is True
    assert result.update["clarification_question"] == "您希望关注哪个价格段的品牌？"


async def test_clarify_with_image_context_sends_multimodal_message(
    mock_config,
) -> None:
    """When object_context is set, node sends multimodal HumanMessage with image_url + text."""
    mock_brief = ResearchBrief(
        need_clarification=False,
        research_goal="分析图片中的时尚趋势",
        language="zh",
    )
    captured = {}

    async def capture_ainvoke(messages):
        captured["messages"] = messages
        return mock_brief

    mock_chain = MagicMock()
    mock_chain.ainvoke = capture_ainvoke

    with patch("deep_agents.agents.clarify.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = (
            mock_chain
        )
        result = await clarify_node(
            {
                "messages": [HumanMessage(content="请分析这张图片的时尚趋势")],
                "object_context": "https://example.com/fashion.jpg",
            },
            mock_config,
        )

    assert isinstance(result, Command)
    sent_messages = captured["messages"]
    last_msg = sent_messages[-1]
    assert isinstance(last_msg, HumanMessage)
    assert isinstance(last_msg.content, list)
    content_types = [block.get("type") for block in last_msg.content]
    assert "image_url" in content_types
    assert "text" in content_types


async def test_clarify_dict_messages(mock_config) -> None:
    """Messages can be plain dicts — node handles both dicts and LangChain message objects."""
    mock_brief = ResearchBrief(
        need_clarification=False,
        research_goal="研究奢侈品市场",
        language="zh",
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_brief)

    with patch("deep_agents.agents.clarify.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = (
            mock_chain
        )
        result = await clarify_node(
            {
                "messages": [
                    {"role": "user", "content": "请研究奢侈品市场"},
                    {"role": "assistant", "content": "好的"},
                    {"role": "user", "content": "重点关注中国市场"},
                ],
                "object_context": None,
            },
            mock_config,
        )

    assert isinstance(result, Command)
    assert result.goto == "planner"
    assert "research_goal" in result.update
