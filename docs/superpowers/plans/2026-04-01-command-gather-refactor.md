# Command + asyncio.gather Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Send-based fan-out and standalone routing functions with Command-based routing inside nodes and asyncio.gather inside section_pipeline_node, fixing the parallel-write collision and unreachable outline-revision path.

**Architecture:** Each routing node returns `Command(goto=next_node, update={...})` so logic stays co-located with the node that has the context. `section_pipeline_node` fans out via `asyncio.gather` with `return_exceptions=True` so one bad section cannot cancel siblings. `planner` routes directly to `section_pipeline`, making `outline_reviser` a recovery-only node reached only when refuted evidence warrants it.

**Tech Stack:** LangGraph `Command` (langgraph.types), `asyncio.gather`, `langchain_core.messages.SystemMessage`/`ToolMessage`, Python 3.11+

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/deep_agents/state.py` | Modify | Add `failed_sections: list[str]` field |
| `src/deep_agents/agents/clarify.py` | Modify | Return `Command` instead of plain dict |
| `src/deep_agents/agents/synthesizer.py` | Modify | Return `Command` instead of plain dict |
| `src/deep_agents/agents/reviewer.py` | Modify | Return `Command` instead of plain dict |
| `src/deep_agents/agents/deep_scout.py` | Modify | Replace `create_react_agent` with explicit tool loop |
| `src/deep_agents/graph.py` | Modify | Gather-based `section_pipeline_node`, new topology, remove 5 routing functions |
| `tests/agents/conftest.py` | Modify | Add `failed_sections` to `sample_state` |
| `tests/agents/test_clarify.py` | Modify | Assert on `Command.update` and `Command.goto` |
| `tests/agents/test_section_pipeline.py` | Modify | Update deep_scout test for new explicit loop |
| `tests/agents/test_quality.py` | Modify | Assert on `Command.update` for reviewer |
| `tests/test_graph.py` | Modify | Remove routing function tests; add Command + topology tests |

**No new files. `schemas.py` unchanged (`ResearchComplete` already exists with `reason: str`).**

---

## Task 1: Add `failed_sections` to ResearchState

**Files:**
- Modify: `src/deep_agents/state.py:37-69`
- Modify: `tests/agents/conftest.py:10-39`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_state.py`:

```python
def test_failed_sections_field_exists():
    from deep_agents.state import ResearchState
    import typing
    hints = typing.get_type_hints(ResearchState)
    assert "failed_sections" in hints
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/czy/Karl/World_Fashion_Daily && uv run pytest tests/test_state.py::test_failed_sections_field_exists -v
```

Expected: FAIL with `AssertionError`

- [ ] **Step 3: Add the field to ResearchState**

In `src/deep_agents/state.py`, after the `section_drafts` line (line 64), add:

```python
    failed_sections: list[str]
```

The full block around it:
```python
    section_drafts: Annotated[list[dict], override_reducer]

    failed_sections: list[str]

    full_report: str
```

- [ ] **Step 4: Add `failed_sections` to conftest sample_state**

In `tests/agents/conftest.py`, add to the `sample_state` dict (after `"section_drafts": []`):

```python
        "failed_sections": [],
```

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_state.py::test_failed_sections_field_exists -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/deep_agents/state.py tests/agents/conftest.py tests/test_state.py
git commit -m "feat: add failed_sections field to ResearchState"
```

---

## Task 2: clarify_node → return Command

**Files:**
- Modify: `src/deep_agents/agents/clarify.py:42-94`
- Modify: `tests/agents/test_clarify.py`

- [ ] **Step 1: Write failing tests**

Replace all assertions in `tests/agents/test_clarify.py` to check `Command` objects. Rewrite the entire file:

```python
"""Tests for clarify_node — the entry point of the research graph."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import END

from deep_agents.agents.clarify import clarify_node
from deep_agents.schemas import ResearchBrief


async def test_clarify_no_clarification_returns_command_to_planner(mock_config):
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
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        result = await clarify_node(
            {"messages": [HumanMessage(content="请研究2025年春夏女装色彩趋势")], "object_context": None},
            mock_config,
        )

    assert isinstance(result, Command)
    assert result.goto == "planner"
    assert result.update["need_clarification"] is False
    assert result.update["research_goal"] == "我想了解2025年春夏女装色彩趋势"
    assert result.update["confirmed_constraints"] == ["女装", "春夏"]
    assert result.update["language"] == "zh"


async def test_clarify_needs_clarification_returns_command_to_end(mock_config):
    """When need_clarification=True, node returns Command(goto=END)."""
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
            {"messages": [HumanMessage(content="帮我研究时尚品牌")], "object_context": None},
            mock_config,
        )

    assert isinstance(result, Command)
    assert result.goto == END
    assert result.update["need_clarification"] is True
    assert result.update["clarification_question"] == "您希望关注哪个价格段的品牌？"


async def test_clarify_with_image_context_sends_multimodal_message(mock_config):
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
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/agents/test_clarify.py -v
```

Expected: FAIL — `result` is a `dict`, not a `Command`

- [ ] **Step 3: Update clarify_node to return Command**

Replace the entire `src/deep_agents/agents/clarify.py` file:

```python
"""Clarify node — entry point of the research graph.

Reads the conversation messages (and optional image context), calls the model with
structured output to produce a ResearchBrief, then returns a Command routing to
'planner' or END based on whether clarification is needed.
"""
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command

from deep_agents.configuration import Configuration
from deep_agents.prompts import clarify_prompt
from deep_agents.schemas import ResearchBrief
from deep_agents.state import ResearchState
from deep_agents.utils import get_api_key_for_model, get_today_str


def _format_messages(messages: list) -> str:
    """Format a list of LangChain message objects or plain dicts into a readable string."""
    lines = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            class_name = type(msg).__name__.lower()
            if "human" in class_name:
                role = "user"
            elif "ai" in class_name or "assistant" in class_name:
                role = "assistant"
            elif "system" in class_name:
                role = "system"
            else:
                role = class_name
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def clarify_node(
    state: ResearchState, config: RunnableConfig
) -> Command[Literal["planner"]]:
    """Entry-point node. Returns Command(goto=END) if clarification needed, else Command(goto='planner')."""
    configurable = Configuration.from_runnable_config(config)

    model = (
        init_chat_model(
            model=configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            base_url=configurable.openai_compatible_base_url,
        )
        .with_structured_output(ResearchBrief)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    messages = state.get("messages", [])
    object_context = state.get("object_context")
    messages_text = _format_messages(messages)

    image_context_str = ""
    if object_context:
        image_context_str = f"\n用户还提供了一张图片供参考：{object_context}"

    prompt_text = clarify_prompt.format(
        date=get_today_str(),
        messages=messages_text,
        image_context=image_context_str,
    )

    if object_context:
        invoke_input = [
            HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": object_context}},
                    {"type": "text", "text": prompt_text},
                ]
            )
        ]
    else:
        invoke_input = [HumanMessage(content=prompt_text)]

    brief: ResearchBrief = await model.ainvoke(invoke_input)

    update = {
        "need_clarification": brief.need_clarification,
        "clarification_question": brief.clarification_question,
        "research_goal": brief.research_goal,
        "confirmed_constraints": brief.confirmed_constraints,
        "open_dimensions": brief.open_dimensions,
        "language": brief.language,
    }

    if brief.need_clarification:
        return Command(goto=END, update=update)
    return Command(goto="planner", update=update)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/agents/test_clarify.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/deep_agents/agents/clarify.py tests/agents/test_clarify.py
git commit -m "feat: clarify_node returns Command for routing"
```

---

## Task 3: synthesizer_node → return Command

**Files:**
- Modify: `src/deep_agents/agents/synthesizer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_synthesizer.py`:

```python
"""Tests for synthesizer_node."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command


async def test_synthesizer_routes_to_reviewer_for_non_trend(sample_state, mock_config):
    sample_state["research_type"] = "market_overview"
    sample_state["section_drafts"] = [{"section_id": "sec_1", "content": "内容", "citations": [], "charts_used": [], "weak_claims": []}]
    sample_state["hypothesis_evidence"] = []
    sample_state["contradictions"] = []
    sample_state["sources"] = []

    mock_response = MagicMock()
    mock_response.content = "# 完整报告\n\n内容..."

    with patch("deep_agents.agents.synthesizer.init_chat_model") as m:
        m.return_value.ainvoke = AsyncMock(return_value=mock_response)
        from deep_agents.agents.synthesizer import synthesizer_node
        result = await synthesizer_node(sample_state, mock_config)

    assert isinstance(result, Command)
    assert result.goto == "reviewer"
    assert "full_report" in result.update
    assert "完整报告" in result.update["full_report"]


async def test_synthesizer_routes_to_trend_triangulator_for_trend(sample_state, mock_config):
    sample_state["research_type"] = "trend_analysis"
    sample_state["section_drafts"] = []
    sample_state["hypothesis_evidence"] = []
    sample_state["contradictions"] = []
    sample_state["sources"] = []

    mock_response = MagicMock()
    mock_response.content = "# 趋势报告"

    with patch("deep_agents.agents.synthesizer.init_chat_model") as m:
        m.return_value.ainvoke = AsyncMock(return_value=mock_response)
        from deep_agents.agents.synthesizer import synthesizer_node
        result = await synthesizer_node(sample_state, mock_config)

    assert isinstance(result, Command)
    assert result.goto == "trend_triangulator"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/agents/test_synthesizer.py -v
```

Expected: FAIL — result is a `dict`, not `Command`

- [ ] **Step 3: Update synthesizer_node to return Command**

Replace `src/deep_agents/agents/synthesizer.py`:

```python
# src/deep_agents/agents/synthesizer.py
"""Synthesizer: merge all section drafts into a complete full_report."""
import json
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from deep_agents.configuration import Configuration
from deep_agents.prompts import synthesizer_prompt
from deep_agents.state import ResearchState
from deep_agents.utils import get_api_key_for_model


async def synthesizer_node(
    state: ResearchState, config: RunnableConfig
) -> Command[Literal["trend_triangulator", "reviewer"]]:
    """Merge section drafts into full report; route to trend_triangulator or reviewer."""
    configurable = Configuration.from_runnable_config(config)
    model = init_chat_model(
        model=configurable.final_report_model,
        max_tokens=configurable.final_report_model_max_tokens,
        api_key=get_api_key_for_model(configurable.final_report_model, config),
        base_url=configurable.openai_compatible_base_url,
    )

    prompt_text = synthesizer_prompt.format(
        research_goal=state["research_goal"],
        language=state.get("language", "zh"),
        section_drafts=json.dumps(state.get("section_drafts", []), ensure_ascii=False, indent=2),
        hypothesis_evidence=json.dumps(state.get("hypothesis_evidence", [])[:20], ensure_ascii=False),
        contradictions=json.dumps(state.get("contradictions", [])[:10], ensure_ascii=False),
        sources=json.dumps(state.get("sources", [])[:30], ensure_ascii=False),
    )

    response = await model.ainvoke([HumanMessage(content=prompt_text)])

    next_node = "trend_triangulator" if state.get("research_type") == "trend_analysis" else "reviewer"
    return Command(goto=next_node, update={"full_report": response.content})
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/agents/test_synthesizer.py -v
```

Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/deep_agents/agents/synthesizer.py tests/agents/test_synthesizer.py
git commit -m "feat: synthesizer_node returns Command for routing"
```

---

## Task 4: reviewer_node → return Command

**Files:**
- Modify: `src/deep_agents/agents/reviewer.py`
- Modify: `tests/agents/test_quality.py`

- [ ] **Step 1: Write failing tests**

Replace `test_reviewer_pass_on_high_score` in `tests/agents/test_quality.py` with two tests that check the `Command`:

```python
async def test_reviewer_pass_returns_command_to_final_check(sample_state, mock_config):
    from deep_agents.agents.reviewer import reviewer_node
    from langgraph.types import Command

    mock_result = ReviewResult(
        quality_score=8,
        verdict="pass",
        issues=[],
        claim_checks=[],
        missing_aspects=[],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_result)
    sample_state["full_report"] = "# 报告\n\n内容..."
    sample_state["revision_count"] = 0

    with patch("deep_agents.agents.reviewer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await reviewer_node(sample_state, mock_config)

    assert isinstance(result, Command)
    assert result.goto == "final_check"
    assert result.update["review_result"]["verdict"] == "pass"
    assert result.update["review_result"]["quality_score"] == 8


async def test_reviewer_fail_returns_command_to_reviser(sample_state, mock_config):
    from deep_agents.agents.reviewer import reviewer_node
    from langgraph.types import Command

    mock_result = ReviewResult(
        quality_score=4,
        verdict="fail",
        issues=[{"id": "issue_1", "description": "缺少数据支撑"}],
        claim_checks=[],
        missing_aspects=["数据来源"],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_result)
    sample_state["full_report"] = "# 报告"
    sample_state["revision_count"] = 0

    with patch("deep_agents.agents.reviewer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await reviewer_node(sample_state, mock_config)

    assert isinstance(result, Command)
    assert result.goto == "reviser"
    assert result.update["review_result"]["verdict"] == "fail"


async def test_reviewer_max_revisions_goes_to_final_check(sample_state, mock_config):
    from deep_agents.agents.reviewer import reviewer_node
    from langgraph.types import Command

    mock_result = ReviewResult(
        quality_score=4,
        verdict="fail",
        issues=[],
        claim_checks=[],
        missing_aspects=[],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_result)
    sample_state["full_report"] = "# 报告"
    sample_state["revision_count"] = 2  # at max

    with patch("deep_agents.agents.reviewer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await reviewer_node(sample_state, mock_config)

    assert isinstance(result, Command)
    assert result.goto == "final_check"
```

Also remove the old `test_reviewer_pass_on_high_score` function (replace it with the two new tests above).

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/agents/test_quality.py::test_reviewer_pass_returns_command_to_final_check -v
```

Expected: FAIL

- [ ] **Step 3: Update reviewer_node to return Command**

Replace `src/deep_agents/agents/reviewer.py`:

```python
# src/deep_agents/agents/reviewer.py
"""Reviewer: strict quality review → ReviewResult with quality_score and issues."""
import json
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from deep_agents.configuration import Configuration
from deep_agents.prompts import reviewer_prompt
from deep_agents.schemas import ReviewResult
from deep_agents.state import ResearchState
from deep_agents.utils import get_api_key_for_model


async def reviewer_node(
    state: ResearchState, config: RunnableConfig
) -> Command[Literal["reviser", "final_check"]]:
    """Review the full report; route to reviser if quality insufficient, else final_check."""
    configurable = Configuration.from_runnable_config(config)
    model = (
        init_chat_model(
            model=configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            base_url=configurable.openai_compatible_base_url,
        )
        .with_structured_output(ReviewResult)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    prompt_text = reviewer_prompt.format(
        research_goal=state["research_goal"],
        sections=json.dumps(
            [{"id": s["id"], "title": s["title"]} for s in state.get("sections", [])],
            ensure_ascii=False,
        ),
        full_report=state.get("full_report", ""),
        facts=json.dumps(state.get("facts", [])[:30], ensure_ascii=False),
        data_points=json.dumps(state.get("data_points", [])[:20], ensure_ascii=False),
    )

    result: ReviewResult = await model.ainvoke([HumanMessage(content=prompt_text)])
    review_result = result.model_dump()

    if review_result.get("verdict") != "pass" and state.get("revision_count", 0) < 2:
        return Command(goto="reviser", update={"review_result": review_result})
    return Command(goto="final_check", update={"review_result": review_result})
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/agents/test_quality.py -v
```

Expected: all tests PASS (reviser and final_check tests are unchanged and should still pass)

- [ ] **Step 5: Commit**

```bash
git add src/deep_agents/agents/reviewer.py tests/agents/test_quality.py
git commit -m "feat: reviewer_node returns Command for routing"
```

---

## Task 5: deep_scout_node → explicit tool loop

**Files:**
- Modify: `src/deep_agents/agents/deep_scout.py`
- Modify: `tests/agents/test_section_pipeline.py`

- [ ] **Step 1: Write failing tests**

In `tests/agents/test_section_pipeline.py`, replace `test_deep_scout_returns_empty_on_error` and add a success test:

```python
async def test_deep_scout_exits_on_research_complete(section_state, mock_config):
    """Loop exits cleanly when model calls ResearchComplete tool."""
    from deep_agents.agents.deep_scout import deep_scout_node
    from langchain_core.messages import AIMessage

    mock_ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "ResearchComplete", "args": {"reason": "research done"}, "id": "tc1", "type": "tool_call"}],
    )
    mock_bound_model = MagicMock()
    mock_bound_model.ainvoke = AsyncMock(return_value=mock_ai_msg)
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_bound_model

    with patch("deep_agents.agents.deep_scout.get_all_tools", AsyncMock(return_value=[])):
        with patch("deep_agents.agents.deep_scout.init_chat_model", return_value=mock_model):
            result = await deep_scout_node(section_state, mock_config)

    assert "search_results" in result
    assert "section_sources" in result
    assert isinstance(result["search_results"], list)


async def test_deep_scout_returns_empty_on_error(section_state, mock_config):
    """On exception, node returns empty results without raising."""
    from deep_agents.agents.deep_scout import deep_scout_node

    mock_model = MagicMock()
    mock_model.bind_tools.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("search failed"))

    with patch("deep_agents.agents.deep_scout.get_all_tools", AsyncMock(return_value=[])):
        with patch("deep_agents.agents.deep_scout.init_chat_model", return_value=mock_model):
            result = await deep_scout_node(section_state, mock_config)

    assert result["search_results"] == []
    assert result["section_sources"] == []
```

Remove the old `test_deep_scout_returns_empty_on_error` that patches `create_react_agent`.

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/agents/test_section_pipeline.py::test_deep_scout_exits_on_research_complete -v
```

Expected: FAIL (imports `create_react_agent` path not matching)

- [ ] **Step 3: Replace create_react_agent with explicit loop**

Replace `src/deep_agents/agents/deep_scout.py`:

```python
"""DeepScout node — explicit tool-calling loop for section-level evidence collection.

Runs a researcher → researcher_tools loop: the LLM calls search tools and reflects
until it calls ResearchComplete or hits max_react_tool_calls.
"""
import json
import logging
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import deep_scout_prompt
from deep_agents.state import SectionState
from deep_agents.utils import get_all_tools, get_api_key_for_model, get_today_str

logger = logging.getLogger(__name__)

_RESEARCH_COMPLETE_TOOL_NAME = "ResearchComplete"


def _extract_results(tool_result_content: str, sources: list, seen_urls: set) -> dict:
    """Parse a tool result string, extract URL sources, return search_result entry."""
    search_result = {"raw": tool_result_content[:3000]}
    try:
        data = json.loads(tool_result_content)
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                url = item.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({
                        "source_id": f"src_{len(sources):03d}",
                        "url": url,
                        "title": item.get("title", ""),
                        "credibility_score": 0.7,
                    })
    except Exception:
        pass
    return search_result


async def deep_scout_node(state: SectionState, config: RunnableConfig) -> dict:
    """Run an explicit researcher loop to collect evidence for a single research section."""
    configurable = Configuration.from_runnable_config(config)

    try:
        tools = await get_all_tools(config)
    except Exception as exc:
        logger.warning("deep_scout_node: get_all_tools failed: %s", exc)
        tools = []

    # Separate callable tools (BaseTool with .name) from dict-style tools (e.g. OpenAI web_search)
    callable_tools = [t for t in tools if hasattr(t, "name") and callable(getattr(t, "ainvoke", None))]
    tool_map = {t.name: t for t in callable_tools}

    model = init_chat_model(
        model=configurable.research_model,
        max_tokens=configurable.research_model_max_tokens,
        api_key=get_api_key_for_model(configurable.research_model, config),
        base_url=configurable.openai_compatible_base_url,
    )
    bound_model = model.bind_tools(tools)

    system_prompt = deep_scout_prompt.format(
        date=get_today_str(),
        research_goal=state.get("research_goal", ""),
        section_title=state.get("section_title", ""),
        section_description=state.get("section_description", ""),
        search_queries="\n".join(state.get("search_queries", [])),
        hypotheses=json.dumps(state.get("hypotheses", []), ensure_ascii=False),
        max_searches=state.get("budget", {}).get("max_searches", 5),
    )

    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="请开始研究当前章节，收集足够的证据。"),
    ]

    search_results = []
    sources: list[dict] = []
    seen_urls: set[str] = set()

    try:
        for _ in range(configurable.max_react_tool_calls):
            response: AIMessage = await bound_model.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            done = False
            tool_messages = []
            for tc in response.tool_calls:
                if tc["name"] == _RESEARCH_COMPLETE_TOOL_NAME:
                    done = True
                    break

                tool = tool_map.get(tc["name"])
                if tool is None:
                    tool_messages.append(
                        ToolMessage(content=f"Unknown tool: {tc['name']}", tool_call_id=tc["id"], name=tc["name"])
                    )
                    continue

                try:
                    raw_result = await tool.ainvoke(tc, config=config)
                    content = raw_result if isinstance(raw_result, str) else json.dumps(raw_result, ensure_ascii=False)
                    search_results.append(_extract_results(content, sources, seen_urls))
                    tool_messages.append(ToolMessage(content=content[:3000], tool_call_id=tc["id"], name=tc["name"]))
                except Exception as tool_exc:
                    logger.warning("Tool %s failed: %s", tc["name"], tool_exc)
                    tool_messages.append(
                        ToolMessage(content=f"Tool error: {tool_exc}", tool_call_id=tc["id"], name=tc["name"])
                    )

            messages.extend(tool_messages)

            if done:
                break

    except Exception as exc:
        logger.warning("deep_scout_node failed: %s", exc)
        return {"search_results": [], "section_sources": []}

    return {
        "search_results": search_results,
        "section_sources": sources,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/agents/test_section_pipeline.py -v
```

Expected: all 3 tests PASS (`test_analyst_returns_facts`, `test_data_wiz_returns_data_points`, and both deep_scout tests)

- [ ] **Step 5: Commit**

```bash
git add src/deep_agents/agents/deep_scout.py tests/agents/test_section_pipeline.py
git commit -m "feat: deep_scout_node replaces create_react_agent with explicit tool loop"
```

---

## Task 6: graph.py — topology fix + gather-based section_pipeline_node

**Files:**
- Modify: `src/deep_agents/graph.py`
- Modify: `tests/test_graph.py`

- [ ] **Step 1: Write failing tests**

Replace `tests/test_graph.py` entirely:

```python
# tests/test_graph.py
"""Tests for graph topology and section_pipeline_node gather behavior."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langgraph.types import Command
from langgraph.graph import END


def test_graph_compiles_without_error():
    from deep_agents.graph import build_research_graph
    graph = build_research_graph()
    assert graph is not None


def test_section_subgraph_compiles():
    from deep_agents.graph import build_section_subgraph
    sg = build_section_subgraph()
    assert sg is not None


def test_no_routing_functions_exported():
    """Routing functions are removed — they should not be importable from graph."""
    import deep_agents.graph as g
    for name in ("route_after_clarify", "fan_out_sections", "route_after_collection",
                  "route_after_synthesis", "route_after_review"):
        assert not hasattr(g, name), f"{name} should be removed from graph.py"


async def test_section_pipeline_node_merges_results():
    """section_pipeline_node gathers results from all sections and returns Command."""
    from deep_agents.graph import section_pipeline_node

    section_result = {
        "section_facts": [{"content": "fact1", "section_id": "sec_1"}],
        "section_data_points": [{"id": "dp1"}],
        "section_hypothesis_evidence": [],
        "section_charts": [],
        "section_insights": ["insight1"],
        "section_contradictions": [],
        "section_sources": [{"url": "https://example.com"}],
        "missing_info": ["question1"],
    }

    state = {
        "sections": [
            {"id": "sec_1", "title": "市场概况", "description": "规模", "search_queries": ["q1"], "priority": 1}
        ],
        "research_goal": "了解市场",
        "hypotheses": [],
        "budget": {"max_searches": 5},
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = AsyncMock(return_value=section_result)

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        result = await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})

    assert isinstance(result, Command)
    assert result.goto == "lead_writer"
    assert len(result.update["facts"]) == 1
    assert len(result.update["sources"]) == 1
    assert result.update["insights"][0]["section_id"] == "sec_1"
    assert result.update["open_questions"][0]["section_id"] == "sec_1"
    assert result.update["failed_sections"] == []


async def test_section_pipeline_node_handles_failed_section():
    """A section that raises an exception is recorded in failed_sections; others succeed."""
    from deep_agents.graph import section_pipeline_node

    good_result = {
        "section_facts": [{"content": "fact from sec_1"}],
        "section_data_points": [],
        "section_hypothesis_evidence": [],
        "section_charts": [],
        "section_insights": [],
        "section_contradictions": [],
        "section_sources": [],
        "missing_info": [],
    }

    call_count = 0

    async def mock_ainvoke(inp, config=None):
        nonlocal call_count
        call_count += 1
        if inp["section_id"] == "sec_2":
            raise RuntimeError("LLM failed")
        return good_result

    state = {
        "sections": [
            {"id": "sec_1", "title": "市场概况", "description": "规模", "search_queries": ["q1"], "priority": 1},
            {"id": "sec_2", "title": "竞争格局", "description": "品牌", "search_queries": ["q2"], "priority": 2},
        ],
        "research_goal": "了解市场",
        "hypotheses": [],
        "budget": {"max_searches": 5},
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = mock_ainvoke

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        result = await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})

    assert isinstance(result, Command)
    assert "sec_2" in result.update["failed_sections"]
    assert len(result.update["facts"]) == 1  # only sec_1 contributed


async def test_section_pipeline_routes_to_outline_reviser_when_refuted():
    """section_pipeline routes to outline_reviser when ≥2 hypotheses are refuted and count < 1."""
    from deep_agents.graph import section_pipeline_node

    refuted_evidence = [
        {"evidence_type": "refutes", "hypothesis_id": "h1"},
        {"evidence_type": "refutes", "hypothesis_id": "h2"},
    ]
    section_result = {
        "section_facts": [],
        "section_data_points": [],
        "section_hypothesis_evidence": refuted_evidence,
        "section_charts": [],
        "section_insights": [],
        "section_contradictions": [],
        "section_sources": [],
        "missing_info": [],
    }

    state = {
        "sections": [
            {"id": "sec_1", "title": "T", "description": "D", "search_queries": ["q"], "priority": 1}
        ],
        "research_goal": "研究",
        "hypotheses": [],
        "budget": {},
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = AsyncMock(return_value=section_result)

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        result = await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})

    assert result.goto == "outline_reviser"


async def test_section_pipeline_does_not_route_to_outline_reviser_when_count_at_1():
    """After one outline revision (count=1), re-outline is blocked even with refuted hypotheses."""
    from deep_agents.graph import section_pipeline_node

    refuted_evidence = [
        {"evidence_type": "refutes", "hypothesis_id": "h1"},
        {"evidence_type": "refutes", "hypothesis_id": "h2"},
    ]
    section_result = {
        "section_facts": [],
        "section_data_points": [],
        "section_hypothesis_evidence": refuted_evidence,
        "section_charts": [],
        "section_insights": [],
        "section_contradictions": [],
        "section_sources": [],
        "missing_info": [],
    }

    state = {
        "sections": [
            {"id": "sec_1", "title": "T", "description": "D", "search_queries": ["q"], "priority": 1}
        ],
        "research_goal": "研究",
        "hypotheses": [],
        "budget": {},
        "language": "zh",
        "outline_revision_count": 1,  # already revised once
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = AsyncMock(return_value=section_result)

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        result = await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})

    assert result.goto == "lead_writer"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: several FAIL — routing functions still exist, `section_pipeline_node` still uses Send

- [ ] **Step 3: Rewrite graph.py**

Replace `src/deep_agents/graph.py` entirely:

```python
# src/deep_agents/graph.py
"""LangGraph builder: main research graph + section subgraph."""
import asyncio
import logging
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from deep_agents.state import ResearchState, SectionState

# Agent node imports
from deep_agents.agents.clarify import clarify_node
from deep_agents.agents.planner import planner_node
from deep_agents.agents.outline_reviser import outline_reviser_node
from deep_agents.agents.analyst import analyst_node
from deep_agents.agents.data_wiz import data_wiz_node
from deep_agents.agents.deep_scout import deep_scout_node
from deep_agents.agents.writer import writer_node
from deep_agents.agents.synthesizer import synthesizer_node
from deep_agents.agents.trend_triangulator import trend_triangulator_node
from deep_agents.agents.reviewer import reviewer_node
from deep_agents.agents.reviser import reviser_node
from deep_agents.agents.final_check import final_check_node

logger = logging.getLogger(__name__)

# ── Section subgraph (lazy singleton) ───────────────────────────────────────

_section_subgraph = None


def _get_section_subgraph():
    global _section_subgraph
    if _section_subgraph is None:
        _section_subgraph = build_section_subgraph()
    return _section_subgraph


# ── section_pipeline_node (gather-based) ────────────────────────────────────

async def section_pipeline_node(
    state: ResearchState, config: RunnableConfig
) -> Command[Literal["lead_writer", "outline_reviser"]]:
    """Fan out to all section subgraphs concurrently; merge results; route via Command."""
    sg = _get_section_subgraph()

    section_inputs = [
        {
            "section_id": s["id"],
            "section_title": s["title"],
            "section_description": s["description"],
            "search_queries": s["search_queries"],
            "research_goal": state["research_goal"],
            "hypotheses": state.get("hypotheses", []),
            "budget": state.get("budget", {}),
            "language": state.get("language", "zh"),
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
        for s in state.get("sections", [])
    ]

    raw = await asyncio.gather(
        *[sg.ainvoke(inp, config) for inp in section_inputs],
        return_exceptions=True,
    )

    merged: dict = {
        "facts": [],
        "data_points": [],
        "hypothesis_evidence": [],
        "charts": [],
        "insights": [],
        "contradictions": [],
        "sources": [],
        "open_questions": [],
        "failed_sections": [],
    }

    for s, r in zip(state.get("sections", []), raw):
        if isinstance(r, Exception):
            logger.warning("Section %s failed: %s", s["id"], r)
            merged["failed_sections"].append(s["id"])
            continue
        merged["facts"].extend(r.get("section_facts", []))
        merged["data_points"].extend(r.get("section_data_points", []))
        merged["hypothesis_evidence"].extend(r.get("section_hypothesis_evidence", []))
        merged["charts"].extend(r.get("section_charts", []))
        merged["insights"].extend(
            {"section_id": s["id"], "insight": i} for i in r.get("section_insights", [])
        )
        merged["contradictions"].extend(r.get("section_contradictions", []))
        merged["sources"].extend(r.get("section_sources", []))
        merged["open_questions"].extend(
            {"section_id": s["id"], "question": q} for q in r.get("missing_info", [])
        )

    refuted = sum(
        1 for h in merged["hypothesis_evidence"] if h.get("evidence_type") == "refutes"
    )
    next_node = (
        "outline_reviser"
        if refuted >= 2 and state.get("outline_revision_count", 0) < 1
        else "lead_writer"
    )

    return Command(goto=next_node, update=merged)


# ── Graph builders ────────────────────────────────────────────────────────────

def build_section_subgraph():
    graph = StateGraph(SectionState)
    graph.add_node("deep_scout", deep_scout_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("data_wiz", data_wiz_node)
    graph.add_edge(START, "deep_scout")
    graph.add_edge("deep_scout", "analyst")
    graph.add_edge("analyst", "data_wiz")
    graph.add_edge("data_wiz", END)
    return graph.compile()


def build_research_graph():
    graph = StateGraph(ResearchState)

    # Nodes
    graph.add_node("clarify", clarify_node)
    graph.add_node("planner", planner_node)
    graph.add_node("outline_reviser", outline_reviser_node)
    graph.add_node("section_pipeline", section_pipeline_node)
    graph.add_node("lead_writer", writer_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("trend_triangulator", trend_triangulator_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("reviser", reviser_node)
    graph.add_node("final_check", final_check_node)

    # Edges
    # clarify returns Command → no edge declaration needed from clarify
    graph.add_edge(START, "clarify")
    # planner routes directly to section_pipeline (outline_reviser is recovery-only)
    graph.add_edge("planner", "section_pipeline")
    # outline_reviser routes back to section_pipeline for re-research
    graph.add_edge("outline_reviser", "section_pipeline")
    # section_pipeline returns Command → no conditional edge needed
    graph.add_edge("lead_writer", "synthesizer")
    # synthesizer returns Command → no conditional edge needed
    graph.add_edge("trend_triangulator", "reviewer")
    # reviewer returns Command → no conditional edge needed
    graph.add_edge("reviser", "reviewer")
    graph.add_edge("final_check", END)

    return graph.compile(checkpointer=MemorySaver())
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS. If `tests/test_graph.py` tests that import removed functions fail with `ImportError`, that confirms they're gone — those specific old tests should be replaced (you already replaced them in Step 1 above).

- [ ] **Step 5: Commit**

```bash
git add src/deep_agents/graph.py tests/test_graph.py
git commit -m "feat: graph — gather-based section_pipeline, planner→section_pipeline direct edge, remove routing functions"
```

---

## Final Verification

- [ ] **Run full test suite**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: all tests pass, no imports of removed symbols.

- [ ] **Verify graph compiles and topology is correct**

```bash
uv run python -c "
from deep_agents.graph import build_research_graph
g = build_research_graph()
print('Graph nodes:', list(g.nodes))
print('OK')
"
```

Expected: prints node list without error.

- [ ] **Final commit**

```bash
git add -p  # review any remaining changes
git commit -m "chore: command-gather refactor complete"
```
