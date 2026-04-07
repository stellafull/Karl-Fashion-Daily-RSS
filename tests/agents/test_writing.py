# tests/agents/test_writing.py
import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from deep_agents.prompts import analyst_prompt, data_wiz_prompt, writer_prompt
from deep_agents.schemas import SectionDraft


async def test_writer_produces_section_drafts(sample_state, mock_config):
    from deep_agents.agents.writer import writer_node

    mock_draft = SectionDraft(
        content="## 市场概况\n\n2025年中国奢侈品市场规模达3620亿元...",
        charts_used=[],
        weak_claims=[],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_draft)

    sample_state["facts"] = [
        {"content": "市场规模3620亿元", "section_id": "sec_1"}
    ]

    with patch("deep_agents.agents.writer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await writer_node(sample_state, mock_config)

    assert len(result["section_drafts"]) == 1
    assert result["section_drafts"][0]["section_id"] == "sec_1"
    assert "市场概况" in result["section_drafts"][0]["content"]
    assert mock_init.call_args.kwargs["disable_streaming"] is True


async def test_synthesizer_produces_full_report(sample_state, mock_config):
    from deep_agents.agents.synthesizer import synthesizer_node
    from langgraph.types import Command

    mock_response = MagicMock()
    mock_response.content = "# 2025年中国奢侈品市场深度研究\n\n## 执行摘要\n..."

    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=mock_response)

    sample_state["section_drafts"] = [
        {"section_id": "sec_1", "content": "## 市场概况\n\n内容...", "charts_used": [], "weak_claims": []}
    ]

    with patch("deep_agents.agents.synthesizer.init_chat_model") as mock_init:
        mock_init.return_value = mock_model
        result = await synthesizer_node(sample_state, mock_config)

    assert isinstance(result, Command)
    assert len(result.update["full_report"]) > 0
    assert "执行摘要" in result.update["full_report"]


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


async def test_writer_prompt_inputs_are_strictly_section_local(mock_config):
    from deep_agents.agents.writer import writer_node

    state = {
        "research_goal": "研究",
        "language": "zh",
        "sections": [
            {"id": "sec_1", "title": "章节1", "description": "描述1", "search_queries": ["q1"], "priority": 1},
            {"id": "sec_2", "title": "章节2", "description": "描述2", "search_queries": ["q2"], "priority": 2},
        ],
        "facts": [
            {"content": "fact-sec-1", "section_id": "sec_1"},
            {"content": "fact-sec-2", "section_id": "sec_2"},
            {"content": "fact-without-section"},
        ],
        "data_points": [
            {"name": "dp-sec-1", "value": 1, "section_id": "sec_1"},
            {"name": "dp-sec-2", "value": 2, "section_id": "sec_2"},
            {"name": "dp-without-section", "value": 3},
        ],
        "charts": [
            {"id": "chart-sec-1", "section_id": "sec_1"},
            {"id": "chart-sec-2", "section_id": "sec_2"},
            {"id": "chart-without-section"},
        ],
        "contradictions": [
            {
                "claim_a": "contra-a-sec-1",
                "claim_b": "contra-b-sec-1",
                "section_id": "sec_1",
            },
            {
                "claim_a": "contra-a-sec-2",
                "claim_b": "contra-b-sec-2",
                "section_id": "sec_2",
            },
            {
                "claim_a": "contra-a-no-sec",
                "claim_b": "contra-b-no-sec",
            },
        ],
        "hypothesis_evidence": [
            {
                "hypothesis_statement": "h-sec-1",
                "evidence_type": "supports",
                "content": "e-sec-1",
                "section_id": "sec_1",
            },
            {
                "hypothesis_statement": "h-sec-2",
                "evidence_type": "refutes",
                "content": "e-sec-2",
                "section_id": "sec_2",
            },
            {
                "hypothesis_statement": "h-no-sec",
                "evidence_type": "inconclusive",
                "content": "e-no-sec",
            },
        ],
    }

    mock_drafts = [
        SectionDraft(content="c1", charts_used=[], weak_claims=[]),
        SectionDraft(content="c2", charts_used=[], weak_claims=[]),
    ]
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(side_effect=mock_drafts)

    with patch("deep_agents.agents.writer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        await writer_node(state, mock_config)

    prompts = [call.args[0][0].content for call in mock_chain.ainvoke.await_args_list]
    first_prompt = next(prompt for prompt in prompts if "标题：章节1" in prompt)
    second_prompt = next(prompt for prompt in prompts if "标题：章节2" in prompt)

    assert "fact-sec-1" in first_prompt
    assert "fact-sec-2" not in first_prompt
    assert "fact-without-section" not in first_prompt
    assert "dp-sec-1" in first_prompt
    assert "dp-sec-2" not in first_prompt
    assert "dp-without-section" not in first_prompt
    assert "chart-sec-1" in first_prompt
    assert "chart-sec-2" not in first_prompt
    assert "chart-without-section" not in first_prompt
    assert "contra-a-sec-1" in first_prompt
    assert "contra-a-sec-2" not in first_prompt
    assert "contra-a-no-sec" not in first_prompt
    assert "h-sec-1" in first_prompt
    assert "h-sec-2" not in first_prompt
    assert "h-no-sec" not in first_prompt

    assert "fact-sec-2" in second_prompt
    assert "fact-sec-1" not in second_prompt
    assert "fact-without-section" not in second_prompt
    assert "dp-sec-2" in second_prompt
    assert "dp-sec-1" not in second_prompt
    assert "dp-without-section" not in second_prompt
    assert "chart-sec-2" in second_prompt
    assert "chart-sec-1" not in second_prompt
    assert "chart-without-section" not in second_prompt
    assert "contra-a-sec-2" in second_prompt
    assert "contra-a-sec-1" not in second_prompt
    assert "contra-a-no-sec" not in second_prompt
    assert "h-sec-2" in second_prompt
    assert "h-sec-1" not in second_prompt
    assert "h-no-sec" not in second_prompt
    assert '"section_id"' not in first_prompt
    assert '"section_id"' not in second_prompt


async def test_writer_sections_list_excludes_runtime_ids_from_prompt(mock_config):
    from deep_agents.agents.writer import writer_node

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

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(
        side_effect=[
                SectionDraft(content="c1", charts_used=[], weak_claims=[]),
                SectionDraft(content="c2", charts_used=[], weak_claims=[]),
        ]
    )

    with patch("deep_agents.agents.writer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        await writer_node(state, mock_config)

    prompts = [call.args[0][0].content for call in mock_chain.ainvoke.await_args_list]
    first_prompt = next(prompt for prompt in prompts if "标题：章节1" in prompt)

    assert "完整章节大纲：" in first_prompt
    assert '"title": "章节1"' in first_prompt
    assert '"title": "章节2"' in first_prompt
    assert '"id": "sec_1"' not in first_prompt
    assert '"id": "sec_2"' not in first_prompt
    assert '"id"' not in first_prompt


async def test_writer_writes_sections_in_parallel_with_synchronization(mock_config):
    from deep_agents.agents.writer import writer_node

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

    both_started = asyncio.Event()
    release_all = asyncio.Event()
    started_sections: set[str] = set()
    active_calls = 0
    max_active_calls = 0

    mock_chain = MagicMock()

    async def synchronized_ainvoke(messages):
        nonlocal active_calls, max_active_calls
        prompt = messages[0].content
        if "标题：章节1" in prompt:
            section_id = "sec_1"
        elif "标题：章节2" in prompt:
            section_id = "sec_2"
        else:
            raise AssertionError("unexpected section prompt")

        started_sections.add(section_id)
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)
        if started_sections == {"sec_1", "sec_2"}:
            both_started.set()

        await release_all.wait()
        active_calls -= 1
        return SectionDraft(content=f"draft-{section_id}", charts_used=[], weak_claims=[])

    mock_chain.ainvoke = AsyncMock(side_effect=synchronized_ainvoke)

    with patch("deep_agents.agents.writer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        writer_task = asyncio.create_task(writer_node(state, mock_config))
        await asyncio.wait_for(both_started.wait(), timeout=1)
        assert max_active_calls >= 2

        release_all.set()
        result = await asyncio.wait_for(writer_task, timeout=1)

    assert {draft["section_id"] for draft in result["section_drafts"]} == {"sec_1", "sec_2"}


async def test_writer_cancels_siblings_and_emits_no_late_section_done_on_failure(mock_config):
    from deep_agents.agents.writer import writer_node

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

    both_started = asyncio.Event()
    release_second = asyncio.Event()
    second_cancelled = asyncio.Event()
    second_completed = asyncio.Event()
    started_sections: set[str] = set()
    stream_events: list[dict] = []

    mock_chain = MagicMock()

    async def failfast_ainvoke(messages):
        prompt = messages[0].content
        if "标题：章节1" in prompt:
            section_id = "sec_1"
        elif "标题：章节2" in prompt:
            section_id = "sec_2"
        else:
            raise AssertionError("unexpected section prompt")

        started_sections.add(section_id)
        if started_sections == {"sec_1", "sec_2"}:
            both_started.set()

        await both_started.wait()
        if section_id == "sec_1":
            raise RuntimeError("boom")

        try:
            await release_second.wait()
        except asyncio.CancelledError:
            second_cancelled.set()
            raise

        second_completed.set()
        return SectionDraft(content="draft-sec-2", charts_used=[], weak_claims=[])

    mock_chain.ainvoke = AsyncMock(side_effect=failfast_ainvoke)

    with patch("deep_agents.agents.writer.get_stream_writer", return_value=stream_events.append, create=True):
        with patch("deep_agents.agents.writer.init_chat_model") as mock_init:
            mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
            writer_task = asyncio.create_task(writer_node(state, mock_config))
            await asyncio.wait_for(both_started.wait(), timeout=1)
            with pytest.raises(Exception) as exc_info:
                await asyncio.wait_for(writer_task, timeout=1)

    assert "boom" in str(exc_info.value)

    # If sibling tasks were not cancelled, this release would allow a late completion/event.
    release_second.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert second_cancelled.is_set()
    assert not second_completed.is_set()
    assert stream_events == []


async def test_writer_rejects_empty_sections(mock_config):
    from deep_agents.agents.writer import writer_node

    state = {
        "research_goal": "研究",
        "language": "zh",
        "sections": [],
        "facts": [],
        "data_points": [],
        "charts": [],
        "contradictions": [],
        "hypothesis_evidence": [],
    }

    with pytest.raises(ValueError, match="Cannot write report without sections"):
        await writer_node(state, mock_config)


def test_analyst_prompt_uses_url_based_fields() -> None:
    assert "content, importance" in analyst_prompt
    assert "hypothesis_statement, evidence_type, content" in analyst_prompt
    assert "claim_a, claim_b" in analyst_prompt


def test_data_wiz_prompt_has_no_source_url_contract() -> None:
    assert "source_url" not in data_wiz_prompt
    assert "name, value, unit, year, category, confidence" in data_wiz_prompt
    assert "不得输出 id 或 source_id 字段" in data_wiz_prompt


def test_writer_prompt_citation_contract_has_no_model_facing_section_id() -> None:
    assert "章节ID：{section_id}" not in writer_prompt
    assert "包含字段：section_id" not in writer_prompt
    assert "已完成章节摘要" not in writer_prompt
    assert "完整章节大纲：{sections_list}" in writer_prompt
    assert "citations" not in writer_prompt
    assert "source_id" not in writer_prompt
