"""Tests for clarify_node — the entry point of the research graph."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from deep_agents.agents.clarify import clarify_node
from deep_agents.schemas import ResearchBrief


async def test_clarify_no_clarification(mock_config):
    """When need_clarification=False, node returns research_goal and related fields."""
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
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        result = await clarify_node(
            {
                "messages": [HumanMessage(content="请研究2025年春夏女装色彩趋势")],
                "object_context": None,
            },
            mock_config,
        )

    assert result["need_clarification"] is False
    assert result["research_goal"] == "我想了解2025年春夏女装色彩趋势"
    assert result["confirmed_constraints"] == ["女装", "春夏"]
    assert result["open_dimensions"] == ["价格段", "地区"]
    assert result["language"] == "zh"
    assert result["clarification_question"] == ""


async def test_clarify_needs_clarification(mock_config):
    """When need_clarification=True, node returns that state so graph can route to END."""
    mock_brief = ResearchBrief(
        need_clarification=True,
        clarification_question="您希望关注哪个价格段的品牌？",
        language="zh",
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_brief)

    with patch("deep_agents.agents.clarify.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        result = await clarify_node(
            {
                "messages": [HumanMessage(content="帮我研究时尚品牌")],
                "object_context": None,
            },
            mock_config,
        )

    assert result["need_clarification"] is True
    assert result["clarification_question"] == "您希望关注哪个价格段的品牌？"


async def test_clarify_with_image_context(mock_config):
    """When object_context is set, node sends multimodal HumanMessage with image_url + text."""
    mock_brief = ResearchBrief(
        need_clarification=False,
        research_goal="分析图片中的时尚趋势",
        language="zh",
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_brief)

    captured_invocation_args = {}

    async def capture_ainvoke(messages):
        captured_invocation_args["messages"] = messages
        return mock_brief

    mock_chain.ainvoke = capture_ainvoke

    with patch("deep_agents.agents.clarify.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        result = await clarify_node(
            {
                "messages": [HumanMessage(content="请分析这张图片的时尚趋势")],
                "object_context": "https://example.com/fashion.jpg",
            },
            mock_config,
        )

    assert result["need_clarification"] is False
    assert result["research_goal"] == "分析图片中的时尚趋势"

    # Verify a multimodal message was sent (list content with image_url)
    sent_messages = captured_invocation_args["messages"]
    assert len(sent_messages) >= 1
    last_msg = sent_messages[-1]
    assert isinstance(last_msg, HumanMessage)
    assert isinstance(last_msg.content, list)
    content_types = [block.get("type") for block in last_msg.content]
    assert "image_url" in content_types
    assert "text" in content_types


async def test_clarify_dict_messages(mock_config):
    """Messages can be plain dicts — node handles both dicts and LangChain message objects."""
    mock_brief = ResearchBrief(
        need_clarification=False,
        research_goal="研究奢侈品市场",
        language="zh",
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_brief)

    with patch("deep_agents.agents.clarify.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        result = await clarify_node(
            {
                "messages": [
                    {"role": "user", "content": "请研究奢侈品市场"},
                    {"role": "assistant", "content": "好的，我来帮您研究"},
                    {"role": "user", "content": "重点关注中国市场"},
                ],
                "object_context": None,
            },
            mock_config,
        )

    assert result["need_clarification"] is False
    assert "research_goal" in result
    assert "language" in result
