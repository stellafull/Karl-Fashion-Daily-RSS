# tests/test_api.py
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx


@pytest.fixture
def client():
    from deep_agents.api import app
    from httpx import AsyncClient, ASGITransport
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_research_endpoint_clarification_event(client):
    """When clarify returns need_clarification=True, stream should emit clarification event."""

    async def mock_astream(input_state, config, stream_mode):
        yield ("updates", {"clarify": {
            "need_clarification": True,
            "clarification_question": "请问您关注哪个品类？",
        }})

    mock_graph = MagicMock()
    mock_graph.astream = mock_astream

    with patch("deep_agents.api.get_graph", return_value=mock_graph):
        response = await client.post("/research", json={
            "messages": [{"role": "user", "content": "帮我做个研究"}],
            "object_context": None,
            "thread_id": "test-001",
        })

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    events = [json.loads(line[6:]) for line in response.text.split("\n\n") if line.startswith("data: ")]
    clarification_events = [e for e in events if e.get("type") == "clarification"]
    assert len(clarification_events) == 1
    assert "品类" in clarification_events[0]["question"]


async def test_research_endpoint_section_done_events(client):
    """writer_node emits section_done via custom stream; api should forward without order guarantees."""

    async def mock_astream(input_state, config, stream_mode):
        yield ("custom", {"type": "section_done", "section_id": "sec_2"})
        yield ("custom", {"type": "section_done", "section_id": "sec_1"})
        yield ("updates", {"lead_writer": {"section_drafts": []}})

    mock_graph = MagicMock()
    mock_graph.astream = mock_astream

    with patch("deep_agents.api.get_graph", return_value=mock_graph):
        response = await client.post("/research", json={
            "messages": [{"role": "user", "content": "分析市场"}],
            "object_context": None,
            "thread_id": "test-004",
        })

    events = [json.loads(line[6:]) for line in response.text.split("\n\n") if line.startswith("data: ")]
    section_events = [e for e in events if e.get("type") == "section_done"]
    assert len(section_events) == 2
    assert {event["section_id"] for event in section_events} == {"sec_1", "sec_2"}


async def test_writer_node_emits_section_done_custom_events() -> None:
    """writer_node should emit custom section_done events during real node execution."""
    from deep_agents.agents.writer import writer_node
    from deep_agents.schemas import SectionDraft

    state = {
        "research_goal": "研究",
        "language": "zh",
        "sections": [
            {"id": "sec_1", "title": "章节1", "description": "描述1", "search_queries": ["q1"], "priority": 1},
            {"id": "sec_2", "title": "章节2", "description": "描述2", "search_queries": ["q2"], "priority": 2},
        ],
        "facts": [],
        "data_points": [],
        "charts": [],
        "contradictions": [],
        "hypothesis_evidence": [],
    }
    config = {"configurable": {"thread_id": "test-writer-events"}}

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(
        side_effect=[
            SectionDraft(content="draft-1", charts_used=[], weak_claims=[]),
            SectionDraft(content="draft-2", charts_used=[], weak_claims=[]),
        ]
    )

    stream_events: list[dict] = []

    def stream_writer(payload: dict) -> None:
        stream_events.append(payload)

    with patch("deep_agents.agents.writer.get_stream_writer", return_value=stream_writer, create=True):
        with patch("deep_agents.agents.writer.init_chat_model") as mock_init:
            mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
            await writer_node(state, config)

    assert len(stream_events) == 2
    assert {event.get("type") for event in stream_events} == {"section_done"}
    assert {event.get("section_id") for event in stream_events} == {"sec_1", "sec_2"}


async def test_fetch_tokens_handles_missing_runtime_store_without_attribute_error() -> None:
    """Authenticated MCP token loading should tolerate missing runtime store."""
    from deep_agents.utils import fetch_tokens

    config = {
        "configurable": {"thread_id": "thread-1"},
        "metadata": {"owner": "user-1"},
    }

    with patch("deep_agents.utils.get_store", return_value=None):
        assert await fetch_tokens(config) is None


async def test_research_endpoint_report_event(client):
    """Full pipeline: should stream progress events and end with report event."""

    async def mock_astream(input_state, config, stream_mode):
        yield ("updates", {"planner": {"research_type": "trend_analysis", "hypotheses": [], "sections": []}})
        yield ("updates", {"section_pipeline": {"facts": [], "data_points": []}})
        yield ("custom", {"type": "section_done", "section_id": "sec_1"})
        yield ("updates", {"lead_writer": {"section_drafts": []}})
        yield ("updates", {"final_check": {"full_report": "# 最终报告\n\n内容...", "final_result": {}}})

    mock_graph = MagicMock()
    mock_graph.astream = mock_astream

    with patch("deep_agents.api.get_graph", return_value=mock_graph):
        response = await client.post("/research", json={
            "messages": [{"role": "user", "content": "分析中国奢侈品市场"}],
            "object_context": None,
            "thread_id": "test-002",
        })

    events = [json.loads(line[6:]) for line in response.text.split("\n\n") if line.startswith("data: ")]
    event_types = [e["type"] for e in events]
    assert "progress" in event_types
    assert "section_done" in event_types
    assert "report" in event_types
    report_event = next(e for e in events if e["type"] == "report")
    assert "最终报告" in report_event["content"]


async def test_research_endpoint_error_event(client):
    """Unhandled exception should produce error event."""

    async def mock_astream(input_state, config, stream_mode):
        raise RuntimeError("LLM call failed")
        yield  # make it a generator

    mock_graph = MagicMock()
    mock_graph.astream = mock_astream

    with patch("deep_agents.api.get_graph", return_value=mock_graph):
        response = await client.post("/research", json={
            "messages": [{"role": "user", "content": "测试"}],
            "object_context": None,
            "thread_id": "test-003",
        })

    events = [json.loads(line[6:]) for line in response.text.split("\n\n") if line.startswith("data: ")]
    error_events = [e for e in events if e.get("type") == "error"]
    assert len(error_events) == 1
    assert "LLM call failed" in error_events[0]["message"]
