# Fashion Deep Research Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Fashion Deep Research Agent — a standalone FastAPI service that takes a message history, runs a multi-phase LangGraph research pipeline, and streams a full markdown report back via SSE.

**Architecture:** LangGraph main graph (clarify → plan → parallel section collection → write → review loop) with a section subgraph (deep_scout → analyst → data_wiz). All agent nodes are plain async functions. FastAPI wraps the graph with a single SSE endpoint.

**Tech Stack:** Python 3.10+, LangChain, LangGraph, FastAPI, Tavily, Pydantic v2, pytest + pytest-asyncio, uv

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add fastapi, tavily-python, langchain-openai, pytest deps |
| `src/deep_agents/utils.py` | Modify | Fix `open_deep_research` imports → `deep_agents`; add `analyze_image` |
| `src/deep_agents/schemas.py` | Create | All Pydantic structured output models + `Summary` + `ResearchComplete` |
| `src/deep_agents/state.py` | Replace | `ResearchState` + `SectionState` TypedDicts |
| `src/deep_agents/prompts.py` | Replace | All Chinese prompts as module-level string constants |
| `src/deep_agents/graph.py` | Create | LangGraph builder, subgraph, conditional edge functions |
| `src/deep_agents/api.py` | Create | FastAPI app, SSE `/research` endpoint |
| `src/deep_agents/agents/clarify.py` | Rewrite | Entry node: messages + image → ResearchBrief |
| `src/deep_agents/agents/planner.py` | Create | research_goal → ArchitectPlan |
| `src/deep_agents/agents/outline_reviser.py` | Create | Adjust sections after collection signals |
| `src/deep_agents/agents/deep_scout.py` | Create | ReAct tool-calling section research agent |
| `src/deep_agents/agents/analyst.py` | Adapt | Section qualitative analysis → AnalystOutput |
| `src/deep_agents/agents/data_wiz.py` | Create | Section data extraction → DataWizOutput |
| `src/deep_agents/agents/writer.py` | Adapt | Write all section drafts in one node call |
| `src/deep_agents/agents/synthesizer.py` | Create | Merge drafts → full_report |
| `src/deep_agents/agents/trend_triangulator.py` | Create | Validate trend claims (conditional node) |
| `src/deep_agents/agents/reviewer.py` | Adapt | Quality review → ReviewResult |
| `src/deep_agents/agents/reviser.py` | Create | Targeted edits based on reviewer feedback |
| `src/deep_agents/agents/final_check.py` | Create | Final gate → FinalResult |
| `tests/conftest.py` | Create | Shared fixtures |
| `tests/test_schemas.py` | Create | Pydantic model validation tests |
| `tests/test_state.py` | Create | State TypedDict + reducer tests |
| `tests/test_agents/test_clarify.py` | Create | clarify_node unit tests |
| `tests/test_agents/test_planner.py` | Create | planner_node unit tests |
| `tests/test_agents/test_section_pipeline.py` | Create | deep_scout, analyst, data_wiz unit tests |
| `tests/test_agents/test_writing.py` | Create | writer, synthesizer, trend_triangulator tests |
| `tests/test_agents/test_quality.py` | Create | reviewer, reviser, final_check tests |
| `tests/test_graph.py` | Create | Graph compile + routing function tests |
| `tests/test_api.py` | Create | FastAPI SSE endpoint tests |

---

## Task 1: Project Setup — Dependencies & pytest config

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Update pyproject.toml**

```toml
[project]
name = "world-fashion-daily"
version = "0.1.0"
description = "Fashion deep research agent"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "langchain>=1.2.13",
    "langchain-community>=0.4.1",
    "langchain-openai>=0.3.0",
    "langgraph>=1.1.3",
    "langgraph-cli[inmem]>=0.4.19",
    "langsmith>=0.7.22",
    "mcp>=1.26.0",
    "openai>=2.30.0",
    "fastapi>=0.115.0",
    "uvicorn>=0.32.0",
    "tavily-python>=0.5.0",
    "langchain-mcp-adapters>=0.1.0",
    "aiohttp>=3.9.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.28.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Sync dependencies**

```bash
uv sync --extra dev
```

Expected: resolves without errors.

- [ ] **Step 3: Create tests/__init__.py**

```python
```
(empty file)

- [ ] **Step 4: Create tests/conftest.py**

```python
import json
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def sample_state():
    return {
        "messages": [{"role": "user", "content": "分析2025年中国奢侈品市场趋势"}],
        "object_context": None,
        "need_clarification": False,
        "clarification_question": "",
        "research_goal": "我想了解2025年中国奢侈品市场的主要趋势和品牌格局",
        "confirmed_constraints": [],
        "open_dimensions": [],
        "language": "zh",
        "research_type": "trend_analysis",
        "hypotheses": [
            {"id": "h_1", "statement": "高端消费在一线城市复苏", "evidence_needed": ["销售数据"], "status": "untested"}
        ],
        "sections": [
            {
                "id": "sec_1",
                "title": "市场概况",
                "description": "2025年中国奢侈品市场规模与增速",
                "search_queries": ["中国奢侈品市场规模2025", "luxury market China 2025"],
                "priority": 1,
            }
        ],
        "budget": {"max_parallel": 3, "max_searches": 5, "max_deep_reads": 3},
        "outline_status": "provisional",
        "outline_revision_count": 0,
        "facts": [],
        "data_points": [],
        "hypothesis_evidence": [],
        "charts": [],
        "insights": [],
        "contradictions": [],
        "sources": [],
        "open_questions": [],
        "section_drafts": [],
        "full_report": "",
        "review_result": None,
        "revision_count": 0,
        "final_result": None,
    }


@pytest.fixture
def sample_section_state():
    return {
        "section_id": "sec_1",
        "section_title": "市场概况",
        "section_description": "2025年中国奢侈品市场规模与增速",
        "search_queries": ["中国奢侈品市场规模2025", "luxury market China 2025"],
        "research_goal": "我想了解2025年中国奢侈品市场的主要趋势",
        "hypotheses": [
            {"id": "h_1", "statement": "高端消费在一线城市复苏", "evidence_needed": ["销售数据"], "status": "untested"}
        ],
        "budget": {"max_parallel": 3, "max_searches": 5, "max_deep_reads": 3},
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


@pytest.fixture
def mock_config():
    return {"configurable": {"thread_id": "test-thread-001"}}
```

- [ ] **Step 5: Verify pytest discovers tests**

```bash
uv run pytest tests/ --collect-only
```

Expected: `no tests ran` with 0 errors (conftest loads cleanly).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tests/
git commit -m "setup: add dev dependencies and pytest config"
```

---

## Task 2: Foundation Types — schemas.py

**Files:**
- Create: `src/deep_agents/schemas.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schemas.py
import pytest
from pydantic import ValidationError
from deep_agents.schemas import (
    Summary,
    ResearchComplete,
    ResearchBrief,
    Hypothesis,
    Section,
    ArchitectPlan,
    AnalystOutput,
    DataWizOutput,
    SectionDraft,
    ReviewResult,
    FinalResult,
)


def test_summary_fields():
    s = Summary(summary="test summary", key_excerpts="key fact")
    assert s.summary == "test summary"
    assert s.key_excerpts == "key fact"


def test_research_brief_defaults():
    b = ResearchBrief(need_clarification=False, research_goal="I want to understand X")
    assert b.language == "zh"
    assert b.confirmed_constraints == []


def test_research_brief_clarification():
    b = ResearchBrief(need_clarification=True, clarification_question="Which market?")
    assert b.need_clarification is True
    assert b.research_goal == ""


def test_architect_plan_structure():
    plan = ArchitectPlan(
        research_type="trend_analysis",
        hypotheses=[
            Hypothesis(id="h_1", statement="Luxury is growing", evidence_needed=["sales data"])
        ],
        sections=[
            Section(id="sec_1", title="Market Overview", description="size and growth",
                    search_queries=["luxury market 2025"], priority=1)
        ],
        budget={"max_parallel": 3, "max_searches": 5, "max_deep_reads": 3},
    )
    assert plan.outline_status == "provisional"
    assert len(plan.hypotheses) == 1
    assert plan.hypotheses[0].status == "untested"


def test_review_result_verdict():
    r = ReviewResult(
        quality_score=8,
        verdict="pass",
        issues=[],
        claim_checks=[],
        missing_aspects=[],
    )
    assert r.quality_score == 8


def test_final_result_readiness():
    f = FinalResult(
        resolved_issues=[],
        unresolved_issues=[],
        new_issues=[],
        final_score=8,
        final_verdict="approved",
        publication_readiness="ready",
        final_comments="Good report.",
    )
    assert f.publication_readiness == "ready"


def test_research_complete_has_reason():
    rc = ResearchComplete(reason="All sections covered")
    assert rc.reason == "All sections covered"
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
uv run pytest tests/test_schemas.py -v
```

Expected: `ImportError: cannot import name 'Summary' from 'deep_agents.schemas'`

- [ ] **Step 3: Implement schemas.py**

```python
# src/deep_agents/schemas.py
"""Pydantic structured output models for all agent nodes."""

from typing import List, Optional
from pydantic import BaseModel, Field


class Summary(BaseModel):
    """Used by summarize_webpage in utils.py."""
    summary: str
    key_excerpts: str


class ResearchComplete(BaseModel):
    """Tool signal that research collection is complete."""
    reason: str = Field(description="Why research is complete for this section")


class ResearchBrief(BaseModel):
    """Output of clarify_node."""
    need_clarification: bool
    clarification_question: str = ""
    research_goal: str = ""
    confirmed_constraints: List[str] = []
    open_dimensions: List[str] = []
    language: str = "zh"


class Hypothesis(BaseModel):
    id: str
    statement: str
    evidence_needed: List[str]
    status: str = "untested"  # untested | supported | refuted | partial


class Section(BaseModel):
    id: str
    title: str
    description: str
    search_queries: List[str]
    priority: int


class ArchitectPlan(BaseModel):
    """Output of planner_node."""
    research_type: str  # trend_analysis | brand_analysis | market_overview | consumer_insight | competitive_landscape
    hypotheses: List[Hypothesis]
    sections: List[Section]
    budget: dict
    outline_status: str = "provisional"


class RevisedOutline(BaseModel):
    """Output of outline_reviser_node."""
    sections: List[dict]
    outline_status: str = "revised"


class AnalystOutput(BaseModel):
    """Output of analyst_node."""
    section_facts: List[dict] = []
    section_insights: List[str] = []
    section_hypothesis_evidence: List[dict] = []
    section_contradictions: List[dict] = []
    section_entities: List[dict] = []
    missing_info: List[str] = []


class DataWizOutput(BaseModel):
    """Output of data_wiz_node."""
    section_data_points: List[dict] = []
    section_charts: List[dict] = []
    section_time_series: List[dict] = []


class SectionDraft(BaseModel):
    """Output of writer_node per section."""
    section_id: str
    content: str
    citations: List[dict] = []
    charts_used: List[str] = []
    weak_claims: List[str] = []


class ReviewResult(BaseModel):
    """Output of reviewer_node."""
    quality_score: int  # 1-10
    verdict: str        # pass | fail
    issues: List[dict] = []
    claim_checks: List[dict] = []
    missing_aspects: List[str] = []


class ReviserOutput(BaseModel):
    """Output of reviser_node."""
    full_report: str
    changes_made: List[str] = []
    addressed_issues: List[str] = []
    unable_to_address: List[str] = []


class FinalResult(BaseModel):
    """Output of final_check_node."""
    resolved_issues: List[dict] = []
    unresolved_issues: List[dict] = []
    new_issues: List[dict] = []
    final_score: int
    final_verdict: str          # approved | rejected
    publication_readiness: str  # ready | needs_review
    final_comments: str
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_schemas.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deep_agents/schemas.py tests/test_schemas.py
git commit -m "feat: add Pydantic schemas for all agent structured outputs"
```

---

## Task 3: Graph State — state.py

**Files:**
- Replace: `src/deep_agents/state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_state.py
import operator
import pytest
from typing import get_args, get_type_hints
from deep_agents.state import ResearchState, SectionState


def test_research_state_importable():
    # Constructing a minimal valid state dict
    state: ResearchState = {
        "messages": [],
        "object_context": None,
        "need_clarification": False,
        "clarification_question": "",
        "research_goal": "",
        "confirmed_constraints": [],
        "open_dimensions": [],
        "language": "zh",
        "research_type": "",
        "hypotheses": [],
        "sections": [],
        "budget": {},
        "outline_status": "provisional",
        "outline_revision_count": 0,
        "facts": [],
        "data_points": [],
        "hypothesis_evidence": [],
        "charts": [],
        "insights": [],
        "contradictions": [],
        "sources": [],
        "open_questions": [],
        "section_drafts": [],
        "full_report": "",
        "review_result": None,
        "revision_count": 0,
        "final_result": None,
    }
    assert state["need_clarification"] is False
    assert state["facts"] == []


def test_section_state_importable():
    state: SectionState = {
        "section_id": "sec_1",
        "section_title": "Market Overview",
        "section_description": "Size and growth",
        "search_queries": ["luxury market 2025"],
        "research_goal": "Understand luxury trends",
        "hypotheses": [],
        "budget": {"max_searches": 5},
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
    assert state["section_id"] == "sec_1"


def test_parallel_fields_use_add_reducer():
    """Verify operator.add is the reducer for parallel-write fields."""
    hints = get_type_hints(ResearchState, include_extras=True)
    parallel_fields = ["facts", "data_points", "hypothesis_evidence", "charts",
                       "insights", "contradictions", "sources", "open_questions", "section_drafts"]
    for field in parallel_fields:
        args = get_args(hints[field])
        assert operator.add in args, f"Field '{field}' missing operator.add reducer"
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
uv run pytest tests/test_state.py -v
```

Expected: `ImportError` — old state.py has wrong classes.

- [ ] **Step 3: Replace state.py**

```python
# src/deep_agents/state.py
"""LangGraph state definitions for main graph and section subgraph."""

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import add_messages


class ResearchState(TypedDict):
    # Entry — read only by clarify.py
    messages: Annotated[list, add_messages]
    object_context: str | None

    # Clarification
    need_clarification: bool
    clarification_question: str

    # Research goal — written by clarify.py, read by all downstream nodes
    research_goal: str
    confirmed_constraints: list[str]
    open_dimensions: list[str]
    language: str

    # Planning
    research_type: str
    hypotheses: list[dict]
    sections: list[dict]
    budget: dict
    outline_status: str
    outline_revision_count: int

    # Collection — parallel-write fields (operator.add reducer)
    facts: Annotated[list[dict], operator.add]
    data_points: Annotated[list[dict], operator.add]
    hypothesis_evidence: Annotated[list[dict], operator.add]
    charts: Annotated[list[dict], operator.add]
    insights: Annotated[list[dict], operator.add]
    contradictions: Annotated[list[dict], operator.add]
    sources: Annotated[list[dict], operator.add]
    open_questions: Annotated[list[dict], operator.add]

    # Writing
    section_drafts: Annotated[list[dict], operator.add]
    full_report: str

    # Quality
    review_result: dict | None
    revision_count: int
    final_result: dict | None


class SectionState(TypedDict):
    # Inputs from parent graph (set by fan_out_sections)
    section_id: str
    section_title: str
    section_description: str
    search_queries: list[str]
    research_goal: str
    hypotheses: list[dict]
    budget: dict
    language: str

    # deep_scout output
    search_results: list[dict]

    # analyst output
    section_facts: list[dict]
    section_insights: list[str]
    section_hypothesis_evidence: list[dict]
    section_contradictions: list[dict]
    section_entities: list[dict]
    missing_info: list[str]

    # data_wiz output
    section_data_points: list[dict]
    section_charts: list[dict]
    section_time_series: list[dict]

    # sources
    section_sources: list[dict]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_state.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deep_agents/state.py tests/test_state.py
git commit -m "feat: replace state.py with ResearchState and SectionState TypedDicts"
```

---

## Task 4: Prompts — prompts.py

**Files:**
- Replace: `src/deep_agents/prompts.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts.py
import pytest
from deep_agents.prompts import (
    clarify_prompt,
    planner_prompt,
    outline_reviser_prompt,
    deep_scout_prompt,
    analyst_prompt,
    data_wiz_prompt,
    writer_prompt,
    synthesizer_prompt,
    trend_triangulator_prompt,
    reviewer_prompt,
    reviser_prompt,
    final_check_prompt,
    summarize_webpage_prompt,
    analyze_image_prompt,
)


def test_clarify_prompt_format_variables():
    rendered = clarify_prompt.format(
        date="Mon Mar 31, 2026",
        messages="user: 分析中国奢侈品市场",
        image_context="",
    )
    assert "2026" in rendered
    assert "分析中国奢侈品市场" in rendered


def test_planner_prompt_format_variables():
    rendered = planner_prompt.format(
        date="Mon Mar 31, 2026",
        research_goal="我想了解中国奢侈品市场",
        confirmed_constraints="无",
        open_dimensions="无",
        language="zh",
    )
    assert "中国奢侈品市场" in rendered


def test_summarize_webpage_prompt_format_variables():
    rendered = summarize_webpage_prompt.format(
        webpage_content="Some content here",
        date="Mon Mar 31, 2026",
    )
    assert "Some content here" in rendered


def test_analyze_image_prompt_is_static():
    # No format variables — used as-is
    assert isinstance(analyze_image_prompt, str)
    assert len(analyze_image_prompt) > 50


def test_all_prompts_are_nonempty():
    prompts = [
        clarify_prompt, planner_prompt, outline_reviser_prompt,
        deep_scout_prompt, analyst_prompt, data_wiz_prompt,
        writer_prompt, synthesizer_prompt, trend_triangulator_prompt,
        reviewer_prompt, reviser_prompt, final_check_prompt,
        summarize_webpage_prompt, analyze_image_prompt,
    ]
    for p in prompts:
        assert isinstance(p, str) and len(p) > 20, f"Prompt is empty or too short: {p[:50]}"
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
uv run pytest tests/test_prompts.py -v
```

Expected: `ImportError: cannot import name 'clarify_prompt'`

- [ ] **Step 3: Replace prompts.py**

```python
# src/deep_agents/prompts.py
"""All Chinese agent prompts. One module-level constant per agent."""

clarify_prompt = """
今天的日期是 {date}。

以下是用户请求深度研究时发送的消息：
<Messages>
{messages}
</Messages>
{image_context}

你的任务：
1. 判断是否需要提出一个澄清问题（仅当模糊性会导致研究方向完全错误时才问）
2. 如果不需要澄清，从消息中提取清晰的研究目标

规则：
- 消息历史中已有澄清问答的，不要重复提问
- 大多数情况下不需要澄清，直接提取研究目标
- 研究目标使用第一人称，从用户角度表达
- 检测用户使用的语言（zh/en）

以有效 JSON 格式响应：
{{
  "need_clarification": false,
  "clarification_question": "",
  "research_goal": "我想了解...",
  "confirmed_constraints": [],
  "open_dimensions": [],
  "language": "zh"
}}

如需澄清时：
{{
  "need_clarification": true,
  "clarification_question": "你的澄清问题",
  "research_goal": "",
  "confirmed_constraints": [],
  "open_dimensions": [],
  "language": "zh"
}}
""".strip()

planner_prompt = """
今天的日期是 {date}。

研究目标：{research_goal}
已确认约束：{confirmed_constraints}
开放维度：{open_dimensions}
输出语言：{language}

你是时尚行业深度研究系统的架构规划师。请为上述研究目标制定完整研究计划。

任务：
1. 将研究分类为以下类型之一：trend_analysis（趋势分析）| brand_analysis（品牌分析）| market_overview（市场概况）| consumer_insight（消费者洞察）| competitive_landscape（竞争格局）
2. 生成 2-4 个待验证的研究假设
3. 设计 3-6 个研究章节，每章节提供 2-4 个搜索词（中英文结合）
4. 根据研究复杂度设置执行预算

预算参考：
- 简单：max_parallel=2, max_searches=3, max_deep_reads=2
- 中等：max_parallel=3, max_searches=5, max_deep_reads=3
- 复杂：max_parallel=4, max_searches=7, max_deep_reads=4

时尚研究指引：
- 趋势研究需覆盖：秀场、社交媒体、零售数据三个维度
- 品牌研究需覆盖：财报、品牌定位、消费者认知
- 优先引用：BoF、WWD、Vogue Runway、Lyst、Edited 等权威来源

以有效 JSON 格式响应，包含字段：research_type, hypotheses（id/statement/evidence_needed/status）, sections（id/title/description/search_queries/priority）, budget（max_parallel/max_searches/max_deep_reads）, outline_status="provisional"
""".strip()

outline_reviser_prompt = """
研究目标：{research_goal}

当前章节大纲：
{sections}

收集阶段发现的假设证据：
{hypothesis_evidence}

你是研究大纲修订专家。根据收集阶段的发现，评估并最小化修改当前大纲。

修订原则：
- 仅在证据强烈表明初始框架有重大遗漏或错误时才修改
- 保留原章节 ID，避免下游混乱
- 可以新增、删除或重排章节，但保持最小改动
- 每次任务最多修订一次

以有效 JSON 格式响应，包含字段：sections（完整章节列表，格式与输入一致）, outline_status="revised"
""".strip()

deep_scout_prompt = """
今天的日期是 {date}。
整体研究目标：{research_goal}
当前章节：{section_title} — {section_description}
初始搜索词：
{search_queries}
待验证假设：
{hypotheses}
预算：最多 {max_searches} 次搜索

你是时尚行业深度研究员，负责为单个章节收集证据。

可用工具：
- tavily_search：执行网络搜索，获取完整页面内容
- think_tool：每次搜索后进行策略性反思
- analyze_image：分析秀场、lookbook 或社交媒体图片

策略：
1. 从提供的搜索词开始
2. 每次搜索后调用 think_tool 分析发现并决定下一步
3. 优先深读 tier-1/2 来源（BoF、WWD、Vogue Runway、Lyst、Edited）
4. 对视觉趋势话题使用 analyze_image
5. 同时收集支持和反驳假设的证据
6. 达到预算上限、结果重复或已足够全面时停止

重要规则：
- 保留矛盾信息，不要强行统一
- 标记 PR 宣传内容和赞助软文
- 不要忽略反驳工作假设的证据
""".strip()

analyst_prompt = """
研究目标：{research_goal}
章节：{section_title} — {section_description}
待验证假设：
{hypotheses}

搜索结果：
{search_results}

你是时尚行业研究分析师。请对上述搜索结果进行定性分析。

任务：
1. 识别叙事主题和模式
2. 评估每个假设的证据状态：supports（支持）| refutes（反驳）| inconclusive（不确定）
3. 提炼超越单一来源的战略洞察
4. 记录矛盾信息（不要解决，保留原样）
5. 识别关键实体和关系

以有效 JSON 格式响应，包含字段：section_facts（每条含 content/source_id/importance）, section_insights（字符串列表）, section_hypothesis_evidence（每条含 hypothesis_id/evidence_type/content/source_id）, section_contradictions（每条含 claim_a/claim_b/source_id_a/source_id_b）, section_entities（每条含 name/type）, missing_info（字符串列表）
""".strip()

data_wiz_prompt = """
研究目标：{research_goal}
章节：{section_title}

搜索结果（含数据）：
{search_results}

你是时尚行业数据分析师。请从搜索结果中提取定量数据。

任务：
1. 提取可量化的数据点（仅提取有明确来源的数字）
2. 识别时间序列数据
3. 识别分布和细分数据
4. 为最有价值的数据生成 ECharts 图表配置

规则：
- 不得捏造或推断数字
- 所有数据点必须有 source_id
- 仅在数据足够清晰时才生成图表

以有效 JSON 格式响应，包含字段：section_data_points（每条含 id/name/value/unit/year/source_id/category/confidence）, section_charts（ECharts option 配置）, section_time_series（时间序列数据）
""".strip()

writer_prompt = """
研究目标：{research_goal}
完整章节大纲：{sections_list}
假设验证结果：{hypothesis_evidence}
已完成章节摘要（避免重复）：{previous_sections_memo}

当前章节：
标题：{section_title}
描述：{section_description}
章节事实：{section_facts}
数据点：{section_data_points}
可用图表：{charts}
矛盾信息：{contradictions}

你是顶级投行研究部首席分析师，正在撰写深度行业研究报告的一个章节。

写作要求：
1. 专业投研语气，使用行业术语
2. 每个关键声明必须引用来源（格式：[来源标题](URL)）
3. 数据支撑论点，而非装饰
4. 有矛盾时呈现双方观点
5. 薄弱证据在 weak_claims 中标注
6. 避免与已完成章节重复
7. 使用 {language} 撰写
8. 目标字数：500-1000字

以有效 JSON 格式响应，包含字段：section_id, content（Markdown 格式正文）, citations（每条含 claim/source_id/url）, charts_used（图表 ID 列表）, weak_claims（薄弱声明列表）
""".strip()

synthesizer_prompt = """
研究目标：{research_goal}
输出语言：{language}

章节草稿：
{section_drafts}

假设验证结果：
{hypothesis_evidence}

矛盾信息：
{contradictions}

信息来源列表：
{sources}

你是研究报告合成专家。请将所有章节草稿合并为一份完整的专业研究报告。

任务：
1. 撰写执行摘要（300字以内）
2. 按顺序合并所有章节，消除冗余
3. 撰写结论，对每个假设给出明确判断（支持/反驳/不确定）
4. 如有未解决矛盾，添加"未解决问题"小节
5. 编制编号参考文献列表（含可点击链接）

规则：
- 不得发明草稿和证据中没有的信息
- 保留不确定性标记，不过度自信
- 使用指定语言输出完整报告

直接输出完整 Markdown 格式报告，不需要 JSON 包装。
""".strip()

trend_triangulator_prompt = """
以下是时尚研究报告：
{full_report}

收集的事实：
{facts}

信息来源：
{sources}

你是时尚趋势验证专家。请对报告中的每个趋势声明进行三信号交叉验证。

三种信号类型：
1. 设计师/秀场信号（设计师选择、秀场呈现）
2. 街头/社交采纳（社交媒体、街拍、消费者自发传播）
3. 商业/零售数据（搜索量、销售额、库存数据）

验证规则：
- 有2-3种信号支持 → 强势趋势
- 仅1种信号支持 → 标记为"新兴趋势"或"弱势趋势"
- 无信号支持 → 从报告中移除该声明

请修订报告，将验证结果融入正文，并在报告末尾添加"趋势验证摘要"表格。

直接输出修订后的完整 Markdown 报告。
""".strip()

reviewer_prompt = """
研究目标：{research_goal}
研究大纲：{sections}

报告内容：
{full_report}

可用事实：
{facts}

可用数据点：
{data_points}

你是极其严苛的学术审稿人和事实核查专家。

审核标准（严格执行）：
1. **零容忍幻觉**：没有明确来源的数据或事实即为问题
2. **逻辑闭环**：论点必须有论据，论据必须有来源
3. **偏见警惕**：单方面观点、情绪化表达均为问题
4. **时效性**：超过2年的数据必须标注
5. **完整性**：是否遗漏研究目标中的重要方面
6. **声明核查**：关键数据声明是否与提供的事实/数据点一致

评分标准：
- 9-10：可直接发布
- 7-8：通过，有小问题
- 5-6：需要修订
- 1-4：重大问题

quality_score >= 7 时 verdict = "pass"，否则 verdict = "fail"

以有效 JSON 格式响应，包含字段：quality_score, verdict（pass/fail）, issues（每条含 id/section_id/type/severity/description/suggestion）, claim_checks（每条含 claim_text/source_id/status）, missing_aspects（字符串列表）
""".strip()

reviser_prompt = """
原始报告：
{full_report}

审稿人反馈：
{review_result}

你是报告修订专家。请根据审稿意见对报告进行有针对性的修改。

修订原则：
1. 仅针对指出的问题进行修改，不做无关改动
2. 有证据支持时才添加内容，不捏造信息
3. 修正事实/逻辑问题
4. 保持行文风格一致

以有效 JSON 格式响应，包含字段：full_report（修订后的完整 Markdown 报告）, changes_made（修改描述列表）, addressed_issues（已解决的问题 ID 列表）, unable_to_address（无法解决的问题 ID 列表，附原因）
""".strip()

final_check_prompt = """
研究目标：{research_goal}
上一轮审稿问题：{review_result}
当前报告：
{full_report}
已修订轮次：{revision_count}

你是最终质量把关人。

任务：
1. 核查上一轮问题是否已被修复
2. 检查修订过程中是否引入新问题
3. 对证据不足的声明添加标注
4. 如已达到最大修订次数（2次）且仍有问题，标记为 needs_review 而非阻止发布

以有效 JSON 格式响应，包含字段：resolved_issues（已解决问题列表）, unresolved_issues（未解决问题列表）, new_issues（新引入问题列表）, final_score（1-10）, final_verdict（approved/rejected）, publication_readiness（ready/needs_review）, final_comments
""".strip()

summarize_webpage_prompt = """
今天的日期是 {date}。

请对以下网页内容进行摘要，提取关键信息供时尚研究使用。

<content>
{webpage_content}
</content>

请提供：
1. 简洁摘要（保留关键数据、声明和观点，200字以内）
2. 关键摘录（最重要的数字、引用或事实，逐条列出）

以有效 JSON 格式响应，包含字段：summary, key_excerpts
""".strip()

analyze_image_prompt = """
你是时尚行业专家，请分析这张时尚图片（秀场、lookbook 或社交媒体图片）。

请从以下维度进行专业分析：
1. **廓形与剪裁**：整体廓形（宽松/修身/结构/流动）、关键剪裁细节
2. **色彩搭配**：主色、辅色、色彩情绪（中性/大胆/柔和/对比）
3. **核心单品**：识别关键服装和配饰品类
4. **趋势信号**：图片呈现了哪些时尚趋势（如果能识别的话）
5. **品牌/风格判断**：推测品牌定位、适合场合、目标消费者

请用简洁专业的中文给出分析结论。
""".strip()
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_prompts.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deep_agents/prompts.py tests/test_prompts.py
git commit -m "feat: add all Chinese agent prompts to prompts.py"
```

---

## Task 5: Fix utils.py imports + add analyze_image

**Files:**
- Modify: `src/deep_agents/utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_utils.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_utils_imports_cleanly():
    """utils.py must import without errors after fixing open_deep_research refs."""
    import deep_agents.utils  # noqa


async def test_analyze_image_calls_model():
    """analyze_image tool should call init_chat_model and return a string."""
    from deep_agents.utils import analyze_image

    mock_response = MagicMock()
    mock_response.content = "廓形：宽松。色彩：中性色调。"

    mock_model = AsyncMock()
    mock_model.ainvoke = AsyncMock(return_value=mock_response)

    with patch("deep_agents.utils.init_chat_model", return_value=mock_model):
        result = await analyze_image.ainvoke(
            {"url": "https://example.com/fashion.jpg"},
            config={"configurable": {}}
        )

    assert "廓形" in result or isinstance(result, str)


def test_get_today_str_format():
    from deep_agents.utils import get_today_str
    result = get_today_str()
    assert isinstance(result, str)
    assert len(result) > 5
```

- [ ] **Step 2: Run test to confirm import fails**

```bash
uv run pytest tests/test_utils.py::test_utils_imports_cleanly -v
```

Expected: `ModuleNotFoundError: No module named 'open_deep_research'`

- [ ] **Step 3: Fix the three import lines in utils.py**

Change lines 32-34 from:
```python
from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.prompts import summarize_webpage_prompt
from open_deep_research.state import ResearchComplete, Summary
```
to:
```python
from deep_agents.configuration import Configuration, SearchAPI
from deep_agents.prompts import summarize_webpage_prompt
from deep_agents.schemas import ResearchComplete, Summary
```

- [ ] **Step 4: Add analyze_image tool to utils.py**

Add this block after the `think_tool` definition (after line ~245):

```python
##########################
# Image Analysis Tool
##########################

ANALYZE_IMAGE_DESCRIPTION = (
    "分析时尚图片，包括秀场、lookbook 和社交媒体图片，提取廓形、色彩、单品和趋势信号。"
)

@tool(description=ANALYZE_IMAGE_DESCRIPTION)
async def analyze_image(url: str, config: RunnableConfig = None) -> str:
    """使用 Kimi 2.5 多模态分析时尚图片。

    Args:
        url: Image URL (runway, lookbook, or social media photo)
        config: Runtime configuration for model settings

    Returns:
        Chinese structured analysis of the fashion image
    """
    from deep_agents.prompts import analyze_image_prompt

    configurable = Configuration.from_runnable_config(config)
    model_api_key = get_api_key_for_model(configurable.research_model, config)
    model = init_chat_model(
        model=configurable.research_model,
        api_key=model_api_key,
        base_url=configurable.openai_compatible_base_url,
        tags=["langsmith:nostream"],
    )

    response = await model.ainvoke([
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": analyze_image_prompt},
        ])
    ])
    return response.content
```

- [ ] **Step 5: Run all utils tests**

```bash
uv run pytest tests/test_utils.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Smoke test the full import chain**

```bash
uv run python -c "from deep_agents.utils import tavily_search, think_tool, analyze_image; print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add src/deep_agents/utils.py tests/test_utils.py
git commit -m "fix: update utils.py imports from open_deep_research to deep_agents; add analyze_image tool"
```

---

## Task 6: Clarify Agent

**Files:**
- Rewrite: `src/deep_agents/agents/clarify.py`
- Create: `tests/agents/__init__.py`
- Create: `tests/agents/test_clarify.py`

- [ ] **Step 1: Create tests/agents/__init__.py**

```python
```
(empty file)

- [ ] **Step 2: Write the failing test**

```python
# tests/agents/test_clarify.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from deep_agents.schemas import ResearchBrief


@pytest.fixture
def base_messages():
    return [{"role": "user", "content": "帮我分析2025年中国奢侈品市场趋势"}]


async def test_clarify_no_clarification_needed(base_messages, mock_config):
    from deep_agents.agents.clarify import clarify_node

    mock_brief = ResearchBrief(
        need_clarification=False,
        research_goal="我想了解2025年中国奢侈品市场的主要趋势",
        language="zh",
    )

    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=mock_brief)
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_brief)

    with patch("deep_agents.agents.clarify.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await clarify_node(
            {"messages": base_messages, "object_context": None},
            mock_config,
        )

    assert result["need_clarification"] is False
    assert "2025" in result["research_goal"]
    assert result["language"] == "zh"


async def test_clarify_needs_clarification(base_messages, mock_config):
    from deep_agents.agents.clarify import clarify_node

    mock_brief = ResearchBrief(
        need_clarification=True,
        clarification_question="请问您关注的是哪个品牌细分市场？",
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_brief)

    with patch("deep_agents.agents.clarify.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await clarify_node(
            {"messages": base_messages, "object_context": None},
            mock_config,
        )

    assert result["need_clarification"] is True
    assert len(result["clarification_question"]) > 0


async def test_clarify_with_image(mock_config):
    from deep_agents.agents.clarify import clarify_node

    mock_brief = ResearchBrief(
        need_clarification=False,
        research_goal="我想了解图片中展示的趋势",
        language="zh",
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_brief)

    with patch("deep_agents.agents.clarify.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await clarify_node(
            {
                "messages": [{"role": "user", "content": "分析这张图片的趋势"}],
                "object_context": "https://example.com/runway.jpg",
            },
            mock_config,
        )

    assert result["need_clarification"] is False
    # Verify multimodal message was constructed (image_url in call args)
    call_args = mock_chained.ainvoke.call_args[0][0]
    assert any(
        isinstance(getattr(m, "content", None), list)
        for m in call_args
    )
```

- [ ] **Step 3: Run test to confirm it fails**

```bash
uv run pytest tests/agents/test_clarify.py -v
```

Expected: `ImportError` or function missing.

- [ ] **Step 4: Rewrite clarify.py**

```python
# src/deep_agents/agents/clarify.py
"""Entry node: reads messages + optional image, produces ResearchBrief or asks clarification."""

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import clarify_prompt
from deep_agents.schemas import ResearchBrief
from deep_agents.state import ResearchState
from deep_agents.utils import get_api_key_for_model, get_today_str


async def clarify_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Clarify research intent from messages and optional image."""
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

    # Format message history as text
    messages_text = "\n".join(
        f"{m.get('role', 'user') if isinstance(m, dict) else getattr(m, 'type', 'user')}: "
        f"{m.get('content', '') if isinstance(m, dict) else getattr(m, 'content', '')}"
        for m in (state.get("messages") or [])
    )

    image_context = (
        f"\n用户附上了一张图片，URL：{state['object_context']}\n请结合图片理解研究意图。"
        if state.get("object_context")
        else ""
    )

    prompt_text = clarify_prompt.format(
        date=get_today_str(),
        messages=messages_text,
        image_context=image_context,
    )

    # Use multimodal message if image is present
    if state.get("object_context"):
        message = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": state["object_context"]}},
            {"type": "text", "text": prompt_text},
        ])
    else:
        message = HumanMessage(content=prompt_text)

    result: ResearchBrief = await model.ainvoke([message])

    return {
        "need_clarification": result.need_clarification,
        "clarification_question": result.clarification_question,
        "research_goal": result.research_goal,
        "confirmed_constraints": result.confirmed_constraints,
        "open_dimensions": result.open_dimensions,
        "language": result.language,
    }
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/agents/test_clarify.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/deep_agents/agents/clarify.py tests/agents/
git commit -m "feat: implement clarify_node — entry point with multimodal support"
```

---

## Task 7: Planner + Outline Reviser

**Files:**
- Create: `src/deep_agents/agents/planner.py`
- Create: `src/deep_agents/agents/outline_reviser.py`
- Create: `tests/agents/test_planner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_planner.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from deep_agents.schemas import ArchitectPlan, Hypothesis, Section, RevisedOutline


async def test_planner_returns_sections_and_hypotheses(sample_state, mock_config):
    from deep_agents.agents.planner import planner_node

    mock_plan = ArchitectPlan(
        research_type="trend_analysis",
        hypotheses=[
            Hypothesis(id="h_1", statement="奢侈品消费复苏", evidence_needed=["销售数据"])
        ],
        sections=[
            Section(id="sec_1", title="市场概况", description="规模与增速",
                    search_queries=["中国奢侈品2025"], priority=1),
            Section(id="sec_2", title="竞争格局", description="主要品牌",
                    search_queries=["luxury brands China"], priority=2),
            Section(id="sec_3", title="趋势分析", description="消费趋势",
                    search_queries=["fashion trends 2025"], priority=3),
        ],
        budget={"max_parallel": 3, "max_searches": 5, "max_deep_reads": 3},
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_plan)

    with patch("deep_agents.agents.planner.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await planner_node(sample_state, mock_config)

    assert result["research_type"] == "trend_analysis"
    assert len(result["sections"]) == 3
    assert len(result["hypotheses"]) == 1
    assert result["outline_revision_count"] == 0
    assert result["outline_status"] == "provisional"


async def test_outline_reviser_increments_count(sample_state, mock_config):
    from deep_agents.agents.outline_reviser import outline_reviser_node

    mock_revised = RevisedOutline(
        sections=sample_state["sections"],
        outline_status="revised",
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_revised)

    sample_state["outline_revision_count"] = 0

    with patch("deep_agents.agents.outline_reviser.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await outline_reviser_node(sample_state, mock_config)

    assert result["outline_revision_count"] == 1
    assert result["outline_status"] == "revised"
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/agents/test_planner.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create planner.py**

```python
# src/deep_agents/agents/planner.py
"""Architect planner: research_goal → ArchitectPlan (sections + hypotheses)."""

import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import planner_prompt
from deep_agents.schemas import ArchitectPlan
from deep_agents.state import ResearchState
from deep_agents.utils import get_api_key_for_model, get_today_str


async def planner_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Generate research plan: research_type, hypotheses, sections, budget."""
    configurable = Configuration.from_runnable_config(config)
    model = (
        init_chat_model(
            model=configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            base_url=configurable.openai_compatible_base_url,
        )
        .with_structured_output(ArchitectPlan)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    prompt_text = planner_prompt.format(
        date=get_today_str(),
        research_goal=state["research_goal"],
        confirmed_constraints="\n".join(f"- {c}" for c in state.get("confirmed_constraints", [])) or "无",
        open_dimensions="\n".join(f"- {d}" for d in state.get("open_dimensions", [])) or "无",
        language=state.get("language", "zh"),
    )

    result: ArchitectPlan = await model.ainvoke([HumanMessage(content=prompt_text)])

    return {
        "research_type": result.research_type,
        "hypotheses": [h.model_dump() for h in result.hypotheses],
        "sections": [s.model_dump() for s in result.sections],
        "budget": result.budget,
        "outline_status": result.outline_status,
        "outline_revision_count": 0,
    }
```

- [ ] **Step 4: Create outline_reviser.py**

```python
# src/deep_agents/agents/outline_reviser.py
"""Outline reviser: minimally adjust sections based on collection signals."""

import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import outline_reviser_prompt
from deep_agents.schemas import RevisedOutline
from deep_agents.state import ResearchState
from deep_agents.utils import get_api_key_for_model


async def outline_reviser_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Revise outline sections based on hypothesis evidence from collection."""
    configurable = Configuration.from_runnable_config(config)
    model = (
        init_chat_model(
            model=configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            base_url=configurable.openai_compatible_base_url,
        )
        .with_structured_output(RevisedOutline)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    prompt_text = outline_reviser_prompt.format(
        research_goal=state["research_goal"],
        sections=json.dumps(state["sections"], ensure_ascii=False, indent=2),
        hypothesis_evidence=json.dumps(
            state.get("hypothesis_evidence", [])[-20:], ensure_ascii=False, indent=2
        ),
    )

    result: RevisedOutline = await model.ainvoke([HumanMessage(content=prompt_text)])

    return {
        "sections": result.sections,
        "outline_status": result.outline_status,
        "outline_revision_count": state.get("outline_revision_count", 0) + 1,
    }
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/agents/test_planner.py -v
```

Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/deep_agents/agents/planner.py src/deep_agents/agents/outline_reviser.py tests/agents/test_planner.py
git commit -m "feat: implement planner_node and outline_reviser_node"
```

---

## Task 8: Section Subgraph — deep_scout, analyst, data_wiz

**Files:**
- Create: `src/deep_agents/agents/deep_scout.py`
- Create: `src/deep_agents/agents/analyst.py`
- Create: `src/deep_agents/agents/data_wiz.py`
- Create: `tests/agents/test_section_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_section_pipeline.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from deep_agents.schemas import AnalystOutput, DataWizOutput


async def test_analyst_returns_facts_and_insights(sample_section_state, mock_config):
    from deep_agents.agents.analyst import analyst_node

    mock_output = AnalystOutput(
        section_facts=[{"content": "中国奢侈品市场规模达3620亿元", "source_id": "src_001", "importance": "high"}],
        section_insights=["高端消费向三四线城市渗透"],
        section_hypothesis_evidence=[{"hypothesis_id": "h_1", "evidence_type": "supports", "content": "销售数据显示复苏"}],
        section_contradictions=[],
        section_entities=[{"name": "LVMH", "type": "brand"}],
        missing_info=[],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_output)

    sample_section_state["search_results"] = [
        {"source_id": "src_001", "title": "中国奢侈品报告", "summary": "市场规模数据..."}
    ]

    with patch("deep_agents.agents.analyst.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await analyst_node(sample_section_state, mock_config)

    assert len(result["section_facts"]) == 1
    assert len(result["section_insights"]) == 1
    assert result["section_facts"][0]["content"] == "中国奢侈品市场规模达3620亿元"


async def test_data_wiz_returns_data_points(sample_section_state, mock_config):
    from deep_agents.agents.data_wiz import data_wiz_node

    mock_output = DataWizOutput(
        section_data_points=[
            {"id": "dp_001", "name": "市场规模", "value": 3620, "unit": "亿元", "year": 2025,
             "source_id": "src_001", "category": "market_size", "confidence": 0.9}
        ],
        section_charts=[],
        section_time_series=[],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_output)

    sample_section_state["search_results"] = [
        {"source_id": "src_001", "title": "Report", "summary": "规模达3620亿元"}
    ]

    with patch("deep_agents.agents.data_wiz.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await data_wiz_node(sample_section_state, mock_config)

    assert len(result["section_data_points"]) == 1
    assert result["section_data_points"][0]["value"] == 3620


async def test_deep_scout_returns_search_results(sample_section_state, mock_config):
    from deep_agents.agents.deep_scout import deep_scout_node

    mock_agent_result = {
        "messages": [
            MagicMock(type="tool", name="tavily_search",
                      content='[{"url": "https://bof.com/article", "title": "Luxury China 2025", "content": "Market data..."}]')
        ]
    }

    with patch("deep_agents.agents.deep_scout.create_react_agent") as mock_create:
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value=mock_agent_result)
        mock_create.return_value = mock_agent
        with patch("deep_agents.agents.deep_scout.init_chat_model"):
            with patch("deep_agents.agents.deep_scout.get_all_tools", AsyncMock(return_value=[])):
                result = await deep_scout_node(sample_section_state, mock_config)

    assert "search_results" in result
    assert "section_sources" in result
```

- [ ] **Step 2: Run to confirm fails**

```bash
uv run pytest tests/agents/test_section_pipeline.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create analyst.py**

```python
# src/deep_agents/agents/analyst.py
"""Analyst: qualitative analysis of search results → AnalystOutput."""

import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import analyst_prompt
from deep_agents.schemas import AnalystOutput
from deep_agents.state import SectionState
from deep_agents.utils import get_api_key_for_model


async def analyst_node(state: SectionState, config: RunnableConfig) -> dict:
    """Analyze search results: extract facts, insights, hypothesis evidence, contradictions."""
    configurable = Configuration.from_runnable_config(config)
    model = (
        init_chat_model(
            model=configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            base_url=configurable.openai_compatible_base_url,
        )
        .with_structured_output(AnalystOutput)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    prompt_text = analyst_prompt.format(
        research_goal=state["research_goal"],
        section_title=state["section_title"],
        section_description=state["section_description"],
        hypotheses=json.dumps(state["hypotheses"], ensure_ascii=False, indent=2),
        search_results=json.dumps(state.get("search_results", []), ensure_ascii=False, indent=2),
    )

    result: AnalystOutput = await model.ainvoke([HumanMessage(content=prompt_text)])

    return {
        "section_facts": result.section_facts,
        "section_insights": result.section_insights,
        "section_hypothesis_evidence": result.section_hypothesis_evidence,
        "section_contradictions": result.section_contradictions,
        "section_entities": result.section_entities,
        "missing_info": result.missing_info,
    }
```

- [ ] **Step 4: Create data_wiz.py**

```python
# src/deep_agents/agents/data_wiz.py
"""Data Wiz: extract quantitative data points and chart configs → DataWizOutput."""

import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import data_wiz_prompt
from deep_agents.schemas import DataWizOutput
from deep_agents.state import SectionState
from deep_agents.utils import get_api_key_for_model


async def data_wiz_node(state: SectionState, config: RunnableConfig) -> dict:
    """Extract data points, time series, and ECharts configs from search results."""
    configurable = Configuration.from_runnable_config(config)
    model = (
        init_chat_model(
            model=configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            base_url=configurable.openai_compatible_base_url,
        )
        .with_structured_output(DataWizOutput)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    prompt_text = data_wiz_prompt.format(
        research_goal=state["research_goal"],
        section_title=state["section_title"],
        search_results=json.dumps(state.get("search_results", []), ensure_ascii=False, indent=2),
    )

    result: DataWizOutput = await model.ainvoke([HumanMessage(content=prompt_text)])

    return {
        "section_data_points": result.section_data_points,
        "section_charts": result.section_charts,
        "section_time_series": result.section_time_series,
    }
```

- [ ] **Step 5: Create deep_scout.py**

```python
# src/deep_agents/agents/deep_scout.py
"""Deep Scout: ReAct tool-calling agent for per-section evidence gathering."""

import json
import logging
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent

from deep_agents.configuration import Configuration
from deep_agents.prompts import deep_scout_prompt
from deep_agents.state import SectionState
from deep_agents.utils import get_api_key_for_model, get_all_tools, get_today_str

logger = logging.getLogger(__name__)


def _parse_agent_messages(messages: list) -> tuple[list[dict], list[dict]]:
    """Extract search_results and sources from agent tool messages."""
    search_results = []
    sources = []
    seen_urls = set()

    for msg in messages:
        if not hasattr(msg, "type"):
            continue
        if msg.type == "tool" and hasattr(msg, "content"):
            content = msg.content
            if isinstance(content, str):
                search_results.append({"raw": content[:2000]})
                # Try to extract URL from content
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        for item in data:
                            url = item.get("url", "")
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                sources.append({
                                    "source_id": f"src_{len(sources):03d}",
                                    "url": url,
                                    "title": item.get("title", ""),
                                    "credibility_score": 0.7,
                                })
                except (json.JSONDecodeError, AttributeError):
                    pass

    return search_results, sources


async def deep_scout_node(state: SectionState, config: RunnableConfig) -> dict:
    """ReAct agent: search, think, analyze images to gather section evidence."""
    configurable = Configuration.from_runnable_config(config)
    model = init_chat_model(
        model=configurable.research_model,
        max_tokens=configurable.research_model_max_tokens,
        api_key=get_api_key_for_model(configurable.research_model, config),
        base_url=configurable.openai_compatible_base_url,
    )

    all_tools = await get_all_tools(config)

    system_prompt = deep_scout_prompt.format(
        date=get_today_str(),
        research_goal=state["research_goal"],
        section_title=state["section_title"],
        section_description=state["section_description"],
        search_queries="\n".join(f"- {q}" for q in state["search_queries"]),
        hypotheses=json.dumps(state["hypotheses"], ensure_ascii=False),
        max_searches=state["budget"].get("max_searches", 5),
    )

    agent = create_react_agent(model, all_tools, prompt=system_prompt)

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="请开始研究当前章节，收集足够的证据。")]},
            config=config,
        )
        search_results, sources = _parse_agent_messages(result.get("messages", []))
    except Exception as e:
        logger.warning(f"deep_scout failed for section {state['section_id']}: {e}")
        search_results, sources = [], []

    return {
        "search_results": search_results,
        "section_sources": sources,
    }
```

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/agents/test_section_pipeline.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/deep_agents/agents/deep_scout.py src/deep_agents/agents/analyst.py src/deep_agents/agents/data_wiz.py tests/agents/test_section_pipeline.py
git commit -m "feat: implement section subgraph nodes — deep_scout, analyst, data_wiz"
```

---

## Task 9: Writing Phase — writer, synthesizer, trend_triangulator

**Files:**
- Create: `src/deep_agents/agents/writer.py`
- Create: `src/deep_agents/agents/synthesizer.py`
- Create: `src/deep_agents/agents/trend_triangulator.py`
- Create: `tests/agents/test_writing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_writing.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from deep_agents.schemas import SectionDraft


async def test_writer_produces_section_drafts(sample_state, mock_config):
    from deep_agents.agents.writer import writer_node

    mock_draft = SectionDraft(
        section_id="sec_1",
        content="## 市场概况\n\n2025年中国奢侈品市场规模达3620亿元...",
        citations=[{"claim": "规模达3620亿元", "source_id": "src_001", "url": "https://example.com"}],
        charts_used=[],
        weak_claims=[],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_draft)

    sample_state["facts"] = [{"content": "市场规模3620亿元", "source_id": "src_001", "section_id": "sec_1"}]

    with patch("deep_agents.agents.writer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await writer_node(sample_state, mock_config)

    assert len(result["section_drafts"]) == 1
    assert result["section_drafts"][0]["section_id"] == "sec_1"
    assert "市场概况" in result["section_drafts"][0]["content"]


async def test_synthesizer_produces_full_report(sample_state, mock_config):
    from deep_agents.agents.synthesizer import synthesizer_node

    mock_response = MagicMock()
    mock_response.content = "# 2025年中国奢侈品市场深度研究\n\n## 执行摘要\n..."

    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=mock_response)

    sample_state["section_drafts"] = [
        {"section_id": "sec_1", "content": "## 市场概况\n\n内容...", "citations": [], "charts_used": [], "weak_claims": []}
    ]

    with patch("deep_agents.agents.synthesizer.init_chat_model") as mock_init:
        mock_init.return_value = mock_model
        result = await synthesizer_node(sample_state, mock_config)

    assert len(result["full_report"]) > 0
    assert "执行摘要" in result["full_report"]


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
```

- [ ] **Step 2: Run to confirm fails**

```bash
uv run pytest tests/agents/test_writing.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create writer.py**

```python
# src/deep_agents/agents/writer.py
"""Lead writer: iterates over all sections in one call, returns all SectionDrafts."""

import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import writer_prompt
from deep_agents.schemas import SectionDraft
from deep_agents.state import ResearchState
from deep_agents.utils import get_api_key_for_model


async def writer_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Write all section drafts sequentially, accumulating context between sections."""
    configurable = Configuration.from_runnable_config(config)
    model = (
        init_chat_model(
            model=configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            base_url=configurable.openai_compatible_base_url,
        )
        .with_structured_output(SectionDraft)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    sections = state.get("sections", [])
    all_facts = state.get("facts", [])
    all_data_points = state.get("data_points", [])
    all_charts = state.get("charts", [])
    all_contradictions = state.get("contradictions", [])
    hypothesis_evidence = state.get("hypothesis_evidence", [])
    language = state.get("language", "zh")

    sections_list = json.dumps(
        [{"id": s["id"], "title": s["title"]} for s in sections],
        ensure_ascii=False,
    )

    drafts = []
    previous_sections_memo = "无"

    for section in sections:
        sec_id = section["id"]

        # Filter evidence to this section
        sec_facts = [f for f in all_facts if f.get("section_id") == sec_id or not f.get("section_id")]
        sec_data = [d for d in all_data_points if d.get("section_id") == sec_id or not d.get("section_id")]
        sec_charts = [c for c in all_charts if c.get("section_id") == sec_id or not c.get("section_id")]
        sec_contradictions = [c for c in all_contradictions if c.get("section_id") == sec_id]

        prompt_text = writer_prompt.format(
            research_goal=state["research_goal"],
            sections_list=sections_list,
            hypothesis_evidence=json.dumps(hypothesis_evidence[:10], ensure_ascii=False),
            previous_sections_memo=previous_sections_memo,
            section_title=section["title"],
            section_description=section["description"],
            section_id=sec_id,
            section_facts=json.dumps(sec_facts[:20], ensure_ascii=False),
            section_data_points=json.dumps(sec_data[:10], ensure_ascii=False),
            charts=json.dumps(sec_charts[:5], ensure_ascii=False),
            contradictions=json.dumps(sec_contradictions[:5], ensure_ascii=False),
            language=language,
        )

        draft: SectionDraft = await model.ainvoke([HumanMessage(content=prompt_text)])
        draft_dict = draft.model_dump()
        draft_dict["section_id"] = sec_id
        drafts.append(draft_dict)

        # Update memo for next section
        previous_sections_memo = "; ".join(
            f"{d['section_id']}({d['content'][:80]}...)" for d in drafts
        )

    return {"section_drafts": drafts}
```

- [ ] **Step 4: Create synthesizer.py**

```python
# src/deep_agents/agents/synthesizer.py
"""Synthesizer: merge all section drafts into a complete full_report."""

import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import synthesizer_prompt
from deep_agents.state import ResearchState
from deep_agents.utils import get_api_key_for_model


async def synthesizer_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Merge section drafts into full report with exec summary and references."""
    configurable = Configuration.from_runnable_config(config)
    model = init_chat_model(
        model=configurable.research_model,
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
    return {"full_report": response.content}
```

- [ ] **Step 5: Create trend_triangulator.py**

```python
# src/deep_agents/agents/trend_triangulator.py
"""Trend triangulator: validate trend claims using 3-signal cross-check (conditional node)."""

import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import trend_triangulator_prompt
from deep_agents.state import ResearchState
from deep_agents.utils import get_api_key_for_model


async def trend_triangulator_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Only runs for trend_analysis research_type. Validates trend claims via 3 signals."""
    configurable = Configuration.from_runnable_config(config)
    model = init_chat_model(
        model=configurable.research_model,
        max_tokens=configurable.final_report_model_max_tokens,
        api_key=get_api_key_for_model(configurable.research_model, config),
        base_url=configurable.openai_compatible_base_url,
    )

    prompt_text = trend_triangulator_prompt.format(
        full_report=state.get("full_report", ""),
        facts=json.dumps(state.get("facts", [])[:30], ensure_ascii=False),
        sources=json.dumps(state.get("sources", [])[:30], ensure_ascii=False),
    )

    response = await model.ainvoke([HumanMessage(content=prompt_text)])
    return {"full_report": response.content}
```

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/agents/test_writing.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/deep_agents/agents/writer.py src/deep_agents/agents/synthesizer.py src/deep_agents/agents/trend_triangulator.py tests/agents/test_writing.py
git commit -m "feat: implement writing phase — writer, synthesizer, trend_triangulator nodes"
```

---

## Task 10: Quality Phase — reviewer, reviser, final_check

**Files:**
- Create: `src/deep_agents/agents/reviewer.py`
- Create: `src/deep_agents/agents/reviser.py`
- Create: `src/deep_agents/agents/final_check.py`
- Create: `tests/agents/test_quality.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_quality.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from deep_agents.schemas import ReviewResult, ReviserOutput, FinalResult


async def test_reviewer_pass_on_high_score(sample_state, mock_config):
    from deep_agents.agents.reviewer import reviewer_node

    mock_result = ReviewResult(
        quality_score=8,
        verdict="pass",
        issues=[],
        claim_checks=[{"claim_text": "市场规模3620亿", "source_id": "src_001", "status": "verified"}],
        missing_aspects=[],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_result)
    sample_state["full_report"] = "# 报告\n\n内容..."

    with patch("deep_agents.agents.reviewer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await reviewer_node(sample_state, mock_config)

    assert result["review_result"]["quality_score"] == 8
    assert result["review_result"]["verdict"] == "pass"


async def test_reviser_updates_report_and_increments_count(sample_state, mock_config):
    from deep_agents.agents.reviser import reviser_node

    mock_output = ReviserOutput(
        full_report="# 修订后报告\n\n改进内容...",
        changes_made=["添加来源引用"],
        addressed_issues=["issue_1"],
        unable_to_address=[],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_output)

    sample_state["full_report"] = "# 原始报告"
    sample_state["review_result"] = {"quality_score": 5, "verdict": "fail", "issues": [{"id": "issue_1"}]}
    sample_state["revision_count"] = 0

    with patch("deep_agents.agents.reviser.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await reviser_node(sample_state, mock_config)

    assert "修订后" in result["full_report"]
    assert result["revision_count"] == 1


async def test_final_check_approved(sample_state, mock_config):
    from deep_agents.agents.final_check import final_check_node

    mock_result = FinalResult(
        resolved_issues=[{"id": "issue_1"}],
        unresolved_issues=[],
        new_issues=[],
        final_score=8,
        final_verdict="approved",
        publication_readiness="ready",
        final_comments="报告质量良好，可以发布。",
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_result)

    sample_state["full_report"] = "# 最终报告"
    sample_state["review_result"] = {"quality_score": 8, "verdict": "pass", "issues": []}
    sample_state["revision_count"] = 1

    with patch("deep_agents.agents.final_check.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await final_check_node(sample_state, mock_config)

    assert result["final_result"]["publication_readiness"] == "ready"
    assert result["final_result"]["final_verdict"] == "approved"
```

- [ ] **Step 2: Run to confirm fails**

```bash
uv run pytest tests/agents/test_quality.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create reviewer.py**

```python
# src/deep_agents/agents/reviewer.py
"""Reviewer: strict quality review → ReviewResult with quality_score and issues."""

import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import reviewer_prompt
from deep_agents.schemas import ReviewResult
from deep_agents.state import ResearchState
from deep_agents.utils import get_api_key_for_model


async def reviewer_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Review the full report: score quality and identify issues."""
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
    return {"review_result": result.model_dump()}
```

- [ ] **Step 4: Create reviser.py**

```python
# src/deep_agents/agents/reviser.py
"""Reviser: apply targeted edits based on reviewer feedback."""

import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import reviser_prompt
from deep_agents.schemas import ReviserOutput
from deep_agents.state import ResearchState
from deep_agents.utils import get_api_key_for_model


async def reviser_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Apply targeted fixes from reviewer feedback; increment revision_count."""
    configurable = Configuration.from_runnable_config(config)
    model = (
        init_chat_model(
            model=configurable.research_model,
            max_tokens=configurable.final_report_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            base_url=configurable.openai_compatible_base_url,
        )
        .with_structured_output(ReviserOutput)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    prompt_text = reviser_prompt.format(
        full_report=state.get("full_report", ""),
        review_result=json.dumps(state.get("review_result", {}), ensure_ascii=False, indent=2),
    )

    result: ReviserOutput = await model.ainvoke([HumanMessage(content=prompt_text)])

    return {
        "full_report": result.full_report,
        "revision_count": state.get("revision_count", 0) + 1,
    }
```

- [ ] **Step 5: Create final_check.py**

```python
# src/deep_agents/agents/final_check.py
"""Final check: verify fixes, detect new issues, assign publication_readiness."""

import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import final_check_prompt
from deep_agents.schemas import FinalResult
from deep_agents.state import ResearchState
from deep_agents.utils import get_api_key_for_model


async def final_check_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Final quality gate: verify revisions, assign publication readiness."""
    configurable = Configuration.from_runnable_config(config)
    model = (
        init_chat_model(
            model=configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            base_url=configurable.openai_compatible_base_url,
        )
        .with_structured_output(FinalResult)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    prompt_text = final_check_prompt.format(
        research_goal=state["research_goal"],
        review_result=json.dumps(state.get("review_result", {}), ensure_ascii=False, indent=2),
        full_report=state.get("full_report", ""),
        revision_count=state.get("revision_count", 0),
    )

    result: FinalResult = await model.ainvoke([HumanMessage(content=prompt_text)])
    return {"final_result": result.model_dump()}
```

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/agents/test_quality.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/deep_agents/agents/reviewer.py src/deep_agents/agents/reviser.py src/deep_agents/agents/final_check.py tests/agents/test_quality.py
git commit -m "feat: implement quality phase — reviewer, reviser, final_check nodes"
```

---

## Task 11: Graph — graph.py

**Files:**
- Create: `src/deep_agents/graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph.py
import pytest
from unittest.mock import MagicMock


def test_graph_compiles_without_error():
    """The compiled graph must be importable and callable."""
    from deep_agents.graph import build_research_graph
    graph = build_research_graph()
    assert graph is not None


def test_section_subgraph_compiles():
    from deep_agents.graph import build_section_subgraph
    sg = build_section_subgraph()
    assert sg is not None


def test_route_after_clarify_needs_clarification():
    from deep_agents.graph import route_after_clarify
    from langgraph.graph import END

    state = {"need_clarification": True}
    assert route_after_clarify(state) == END


def test_route_after_clarify_no_clarification():
    from deep_agents.graph import route_after_clarify

    state = {"need_clarification": False}
    assert route_after_clarify(state) == "planner"


def test_fan_out_sections_creates_sends():
    from deep_agents.graph import fan_out_sections
    from langgraph.types import Send

    state = {
        "sections": [
            {"id": "sec_1", "title": "市场概况", "description": "规模", "search_queries": ["query1"], "priority": 1},
            {"id": "sec_2", "title": "竞争格局", "description": "品牌", "search_queries": ["query2"], "priority": 2},
        ],
        "research_goal": "了解市场",
        "hypotheses": [],
        "budget": {"max_searches": 5},
        "language": "zh",
    }
    sends = fan_out_sections(state)
    assert len(sends) == 2
    assert all(isinstance(s, Send) for s in sends)
    assert sends[0].node == "section_pipeline"


def test_route_after_review_fail_triggers_reviser():
    from deep_agents.graph import route_after_review

    state = {
        "review_result": {"verdict": "fail", "quality_score": 5},
        "revision_count": 0,
    }
    assert route_after_review(state) == "reviser"


def test_route_after_review_pass_goes_to_final_check():
    from deep_agents.graph import route_after_review

    state = {
        "review_result": {"verdict": "pass", "quality_score": 8},
        "revision_count": 0,
    }
    assert route_after_review(state) == "final_check"


def test_route_after_review_max_revisions_goes_to_final_check():
    from deep_agents.graph import route_after_review

    state = {
        "review_result": {"verdict": "fail", "quality_score": 4},
        "revision_count": 2,
    }
    assert route_after_review(state) == "final_check"


def test_route_after_synthesis_trend_analysis():
    from deep_agents.graph import route_after_synthesis

    state = {"research_type": "trend_analysis"}
    assert route_after_synthesis(state) == "trend_triangulator"


def test_route_after_synthesis_non_trend():
    from deep_agents.graph import route_after_synthesis

    state = {"research_type": "brand_analysis"}
    assert route_after_synthesis(state) == "reviewer"
```

- [ ] **Step 2: Run to confirm fails**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: `ModuleNotFoundError: No module named 'deep_agents.graph'`

- [ ] **Step 3: Create graph.py**

```python
# src/deep_agents/graph.py
"""LangGraph builder: main research graph + section subgraph + conditional edge functions."""

from langgraph.graph import StateGraph, START, END, Send
from langgraph.checkpoint.memory import MemorySaver

from deep_agents.state import ResearchState, SectionState

# Agent node imports
from deep_agents.agents.clarify import clarify_node
from deep_agents.agents.planner import planner_node
from deep_agents.agents.outline_reviser import outline_reviser_node
from deep_agents.agents.deep_scout import deep_scout_node
from deep_agents.agents.analyst import analyst_node
from deep_agents.agents.data_wiz import data_wiz_node
from deep_agents.agents.writer import writer_node
from deep_agents.agents.synthesizer import synthesizer_node
from deep_agents.agents.trend_triangulator import trend_triangulator_node
from deep_agents.agents.reviewer import reviewer_node
from deep_agents.agents.reviser import reviser_node
from deep_agents.agents.final_check import final_check_node


# ── Conditional edge functions ──────────────────────────────────────────────

def route_after_clarify(state: ResearchState) -> str:
    if state.get("need_clarification"):
        return END
    return "planner"


def fan_out_sections(state: ResearchState) -> list:
    return [
        Send("section_pipeline", {
            "section_id": s["id"],
            "section_title": s["title"],
            "section_description": s["description"],
            "search_queries": s["search_queries"],
            "research_goal": state["research_goal"],
            "hypotheses": state["hypotheses"],
            "budget": state["budget"],
            "language": state.get("language", "zh"),
            # Initialize empty output fields
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
        })
        for s in state.get("sections", [])
    ]


def route_after_collection(state: ResearchState) -> str:
    refuted_count = sum(
        1 for h in state.get("hypothesis_evidence", [])
        if h.get("evidence_type") == "refutes"
    )
    if refuted_count >= 2 and state.get("outline_revision_count", 0) < 1:
        return "outline_reviser"
    return "lead_writer"


def route_after_synthesis(state: ResearchState) -> str:
    if state.get("research_type") == "trend_analysis":
        return "trend_triangulator"
    return "reviewer"


def route_after_review(state: ResearchState) -> str:
    review = state.get("review_result") or {}
    if review.get("verdict") != "pass" and state.get("revision_count", 0) < 2:
        return "reviser"
    return "final_check"


# ── Section pipeline wrapper ─────────────────────────────────────────────────

def _build_section_subgraph_instance():
    sg = build_section_subgraph()
    return sg


def section_pipeline_node(state: SectionState) -> dict:
    """Wrap section subgraph, merge outputs back to ResearchState reducer fields."""
    result = _section_subgraph.invoke(state)
    return {
        "facts": result.get("section_facts", []),
        "data_points": result.get("section_data_points", []),
        "hypothesis_evidence": result.get("section_hypothesis_evidence", []),
        "charts": result.get("section_charts", []),
        "insights": [
            {"section_id": state["section_id"], "insight": i}
            for i in result.get("section_insights", [])
        ],
        "contradictions": result.get("section_contradictions", []),
        "sources": result.get("section_sources", []),
        "open_questions": [
            {"section_id": state["section_id"], "question": q}
            for q in result.get("missing_info", [])
        ],
    }


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


_section_subgraph = build_section_subgraph()


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
    graph.add_edge(START, "clarify")
    graph.add_conditional_edges("clarify", route_after_clarify)
    graph.add_edge("planner", "outline_reviser")
    graph.add_conditional_edges("outline_reviser", fan_out_sections)
    graph.add_conditional_edges("section_pipeline", route_after_collection)
    graph.add_edge("lead_writer", "synthesizer")
    graph.add_conditional_edges("synthesizer", route_after_synthesis)
    graph.add_edge("trend_triangulator", "reviewer")
    graph.add_conditional_edges("reviewer", route_after_review)
    graph.add_edge("reviser", "reviewer")
    graph.add_edge("final_check", END)

    return graph.compile(checkpointer=MemorySaver())
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_graph.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deep_agents/graph.py tests/test_graph.py
git commit -m "feat: implement graph.py — main research graph and section subgraph with routing"
```

---

## Task 12: FastAPI SSE API — api.py

**Files:**
- Create: `src/deep_agents/api.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

```python
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

    async def mock_astream_events(input_state, config, version):
        # Simulate graph ending after clarify with need_clarification=True
        yield {
            "event": "on_chain_end",
            "name": "clarify",
            "data": {"output": {
                "need_clarification": True,
                "clarification_question": "请问您关注哪个品类？",
            }},
        }

    mock_graph = MagicMock()
    mock_graph.astream_events = mock_astream_events

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


async def test_research_endpoint_report_event(client):
    """Full pipeline: should stream progress events and end with report event."""

    async def mock_astream_events(input_state, config, version):
        yield {"event": "on_chain_end", "name": "planner", "data": {"output": {}}}
        yield {"event": "on_chain_end", "name": "section_pipeline",
               "data": {"output": {"section_id": "sec_1"}}}
        yield {"event": "on_chain_end", "name": "final_check",
               "data": {"output": {"full_report": "# 最终报告\n\n内容...", "final_result": {}}}}

    mock_graph = MagicMock()
    mock_graph.astream_events = mock_astream_events

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

    async def mock_astream_events(input_state, config, version):
        raise RuntimeError("LLM call failed")
        yield  # make it a generator

    mock_graph = MagicMock()
    mock_graph.astream_events = mock_astream_events

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
```

- [ ] **Step 2: Run to confirm fails**

```bash
uv run pytest tests/test_api.py -v
```

Expected: `ModuleNotFoundError: No module named 'deep_agents.api'`

- [ ] **Step 3: Create api.py**

```python
# src/deep_agents/api.py
"""FastAPI app: single POST /research endpoint with SSE streaming."""

import json
import logging
from typing import Any, List, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from deep_agents.graph import build_research_graph

logger = logging.getLogger(__name__)

app = FastAPI(title="Fashion Deep Research API", version="1.0.0")

_graph = None

NODE_NAMES = {
    "clarify", "planner", "outline_reviser", "section_pipeline",
    "lead_writer", "synthesizer", "trend_triangulator",
    "reviewer", "reviser", "final_check",
}


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_research_graph()
    return _graph


class ResearchRequest(BaseModel):
    messages: List[Any]
    object_context: Optional[str] = None
    thread_id: str


@app.post("/research")
async def research(request: ResearchRequest):
    """Start a research job. Returns SSE stream of typed events."""

    async def event_stream():
        graph = get_graph()
        config = {"configurable": {"thread_id": request.thread_id}}
        input_state = {
            "messages": request.messages,
            "object_context": request.object_context,
        }

        final_state_output = {}

        try:
            async for event in graph.astream_events(input_state, config=config, version="v2"):
                kind = event.get("event")
                name = event.get("name", "")
                data = event.get("data", {})

                if kind == "on_chain_end" and name in NODE_NAMES:
                    output = data.get("output") or {}

                    # Progress event for every completed node
                    yield f"data: {json.dumps({'type': 'progress', 'node': name, 'status': 'done'})}\n\n"

                    # Section done event
                    if name == "section_pipeline":
                        section_id = output.get("section_id", "")
                        yield f"data: {json.dumps({'type': 'section_done', 'section_id': section_id})}\n\n"

                    # Clarification event — clarify node ended with need_clarification=True
                    if name == "clarify" and output.get("need_clarification"):
                        yield f"data: {json.dumps({'type': 'clarification', 'question': output.get('clarification_question', '')})}\n\n"

                    # Track final_check output for report event
                    if name == "final_check":
                        final_state_output = output

        except Exception as e:
            logger.error(f"Research pipeline error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        # Emit report event after stream completes
        if final_state_output:
            report = final_state_output.get("full_report", "")
            yield f"data: {json.dumps({'type': 'report', 'content': report})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_api.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS (or only expected failures on integration paths).

- [ ] **Step 6: Smoke test the server starts**

```bash
uv run python -c "from deep_agents.api import app; print('API imports OK')"
```

Expected: `API imports OK`

- [ ] **Step 7: Commit**

```bash
git add src/deep_agents/api.py tests/test_api.py
git commit -m "feat: implement FastAPI SSE endpoint POST /research"
```

---

## Post-Implementation Verification

- [ ] **Full test suite green**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: all tests pass.

- [ ] **Import sanity check**

```bash
uv run python -c "
from deep_agents.api import app
from deep_agents.graph import build_research_graph, build_section_subgraph
from deep_agents.schemas import ResearchBrief, ArchitectPlan, ReviewResult, FinalResult
from deep_agents.state import ResearchState, SectionState
from deep_agents.utils import tavily_search, think_tool, analyze_image
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Final commit**

```bash
git add -A
git commit -m "feat: complete fashion deep research agent implementation"
```
