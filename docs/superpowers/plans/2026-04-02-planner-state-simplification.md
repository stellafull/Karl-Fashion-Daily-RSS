# Planner State Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove planner-owned execution budgeting and simplify planner LLM output to content-only fields, with `planner_node` performing deterministic normalization into the existing runtime shape.

**Architecture:** The planner LLM now outputs only `research_type`, a list of hypothesis statements, and a list of sections (title/description/search_queries). `planner_node` normalizes this into the existing runtime dict shape (generating IDs, status, priority, outline_status, outline_revision_count). `budget` is removed from all state and fan-out payloads; `deep_scout` execution bounds come entirely from configuration.

**Tech Stack:** Python 3.12, Pydantic v2, LangGraph, pytest + pytest-asyncio (`asyncio_mode = "auto"`)

---

## File Map

| File | Change |
|------|--------|
| `src/deep_agents/schemas.py` | Add `PlannerHypothesis`, `PlannerSection`, `SimplifiedPlan`; remove `Hypothesis`, `ArchitectPlan` |
| `src/deep_agents/agents/planner.py` | Use `SimplifiedPlan`; normalize inline; remove budget from return |
| `src/deep_agents/prompts.py` | Simplify `planner_prompt`; remove `{max_searches}` from `deep_scout_prompt` |
| `src/deep_agents/state.py` | Remove `budget: dict` from `ResearchState` and `SectionState` |
| `src/deep_agents/graph.py` | Remove `"budget"` from `section_pipeline_node` fan-out inputs |
| `src/deep_agents/agents/deep_scout.py` | Remove `max_searches=...` from `deep_scout_prompt.format()` |
| `tests/test_schemas.py` | Replace `ArchitectPlan`/`Hypothesis` tests with `SimplifiedPlan` tests |
| `tests/agents/test_planner.py` | Delete coercion test; update planner test to use `SimplifiedPlan`; assert normalized shape |
| `tests/test_state.py` | Move `budget` from `required_keys` to `removed_keys` |
| `tests/test_graph.py` | Remove `budget` from all state fixtures |
| `tests/agents/conftest.py` | Remove `budget` from `sample_state` fixture |
| `tests/agents/test_section_pipeline.py` | Remove `budget` from `section_state` fixture |

---

### Task 1: Update `test_schemas.py` — replace ArchitectPlan/Hypothesis tests

**Files:**
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Replace the import line and remove/replace old tests**

Open `tests/test_schemas.py`. Replace the import block at the top:

```python
from deep_agents.schemas import (
    AnalystOutput,
    DataWizOutput,
    FinalResult,
    PlannerHypothesis,
    PlannerSection,
    ResearchBrief,
    ResearchComplete,
    ReviewResult,
    ReviserOutput,
    RevisedOutline,
    Section,
    SectionDraft,
    SimplifiedPlan,
    Summary,
)
```

Then delete the following three test functions entirely:
- `test_hypothesis_defaults`
- `test_architect_plan_defaults`
- `test_hypothesis_status_invalid_values_raise`
- `test_architect_plan_outline_status_invalid_value_raises`

And add these tests in their place (after `test_research_brief_defaults`):

```python
def test_planner_hypothesis_schema() -> None:
    model = PlannerHypothesis(statement="X is rising")
    assert model.statement == "X is rising"


def test_planner_section_schema() -> None:
    model = PlannerSection(
        title="Market Snapshot",
        description="Topline market movement",
        search_queries=["fashion market 2026", "luxury sales report"],
    )
    assert model.title == "Market Snapshot"
    assert len(model.search_queries) == 2


def test_simplified_plan_schema() -> None:
    model = SimplifiedPlan(
        research_type="trend_analysis",
        hypotheses=[PlannerHypothesis(statement="A is rising")],
        sections=[
            PlannerSection(
                title="T1",
                description="D1",
                search_queries=["q1"],
            )
        ],
    )
    assert model.research_type == "trend_analysis"
    assert len(model.hypotheses) == 1
    assert len(model.sections) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/czy/Karl/World_Fashion_Daily && uv run pytest tests/test_schemas.py -v
```

Expected: `ImportError` — `PlannerHypothesis`, `PlannerSection`, `SimplifiedPlan` do not exist yet.

- [ ] **Step 3: Commit test changes**

```bash
git add tests/test_schemas.py
git commit -m "test: replace ArchitectPlan/Hypothesis schema tests with SimplifiedPlan"
```

---

### Task 2: Update `schemas.py` — add simplified schemas, remove old ones

**Files:**
- Modify: `src/deep_agents/schemas.py`

- [ ] **Step 1: Replace `Hypothesis` and `ArchitectPlan` with new simplified classes**

In `src/deep_agents/schemas.py`, delete the `Hypothesis` class (lines 24–28) and `ArchitectPlan` class (lines 39–44), and add the following three classes in their place (between `ResearchBrief` and `Section`):

```python
class PlannerHypothesis(BaseModel):
    statement: str


class PlannerSection(BaseModel):
    title: str
    description: str
    search_queries: list[str]


class SimplifiedPlan(BaseModel):
    research_type: str
    hypotheses: list[PlannerHypothesis]
    sections: list[PlannerSection]
```

`Section` and `RevisedOutline` stay unchanged — they are still used by the outline reviser.

The `typing.Literal` import may no longer be needed for `ArchitectPlan`, but it's still used by `RevisedOutline` (`Literal["revised"]`) and others, so keep it.

- [ ] **Step 2: Run schema tests to verify they pass**

```bash
cd /home/czy/Karl/World_Fashion_Daily && uv run pytest tests/test_schemas.py -v
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/deep_agents/schemas.py
git commit -m "feat: add SimplifiedPlan/PlannerHypothesis/PlannerSection schemas, remove ArchitectPlan/Hypothesis"
```

---

### Task 3: Update `test_planner.py` — new expectations before implementation

**Files:**
- Modify: `tests/agents/test_planner.py`

- [ ] **Step 1: Replace entire file contents**

Replace `tests/agents/test_planner.py` with:

```python
"""Tests for planner_node and outline_reviser_node."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from deep_agents.schemas import PlannerHypothesis, PlannerSection, RevisedOutline, Section, SimplifiedPlan


async def test_planner_node_returns_normalized_shape(mock_config):
    from deep_agents.agents.planner import planner_node

    mock_plan = SimplifiedPlan(
        research_type="trend_analysis",
        hypotheses=[PlannerHypothesis(statement="X is rising")],
        sections=[
            PlannerSection(title="T", description="D", search_queries=["q"])
        ],
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_plan)
    state = {
        "research_goal": "了解趋势",
        "confirmed_constraints": [],
        "open_dimensions": [],
        "language": "zh",
    }
    with patch("deep_agents.agents.planner.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        result = await planner_node(state, mock_config)

    assert result["research_type"] == "trend_analysis"
    assert result["outline_status"] == "provisional"
    assert result["outline_revision_count"] == 0
    assert "budget" not in result

    # Normalized hypothesis shape
    assert len(result["hypotheses"]) == 1
    h = result["hypotheses"][0]
    assert h["id"] == "h_1"
    assert h["statement"] == "X is rising"
    assert h["status"] == "untested"
    assert h["evidence_needed"] == []

    # Normalized section shape
    assert len(result["sections"]) == 1
    s = result["sections"][0]
    assert s["id"] == "sec_1"
    assert s["title"] == "T"
    assert s["priority"] == 1


async def test_planner_node_ids_increment_correctly(mock_config):
    from deep_agents.agents.planner import planner_node

    mock_plan = SimplifiedPlan(
        research_type="market_overview",
        hypotheses=[
            PlannerHypothesis(statement="H1"),
            PlannerHypothesis(statement="H2"),
        ],
        sections=[
            PlannerSection(title="S1", description="D1", search_queries=["q1"]),
            PlannerSection(title="S2", description="D2", search_queries=["q2"]),
            PlannerSection(title="S3", description="D3", search_queries=["q3"]),
        ],
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_plan)
    state = {
        "research_goal": "市场",
        "confirmed_constraints": [],
        "open_dimensions": [],
        "language": "zh",
    }
    with patch("deep_agents.agents.planner.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        result = await planner_node(state, mock_config)

    assert [h["id"] for h in result["hypotheses"]] == ["h_1", "h_2"]
    assert [s["id"] for s in result["sections"]] == ["sec_1", "sec_2", "sec_3"]
    assert [s["priority"] for s in result["sections"]] == [1, 2, 3]


async def test_outline_reviser_increments_count(mock_config):
    from deep_agents.agents.outline_reviser import outline_reviser_node

    mock_revised = RevisedOutline(
        sections=[Section(id="sec_1", title="T", description="D", search_queries=["q"], priority=1)]
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_revised)
    state = {
        "research_goal": "了解趋势",
        "sections": [{"id": "sec_1", "title": "T", "description": "D", "search_queries": ["q"], "priority": 1}],
        "hypothesis_evidence": [],
        "outline_revision_count": 0,
    }
    with patch("deep_agents.agents.outline_reviser.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        result = await outline_reviser_node(state, mock_config)

    assert result["outline_revision_count"] == 1
    assert result["outline_status"] == "revised"
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/czy/Karl/World_Fashion_Daily && uv run pytest tests/agents/test_planner.py -v
```

Expected: `test_planner_node_returns_normalized_shape` and `test_planner_node_ids_increment_correctly` fail because `planner_node` still uses `ArchitectPlan` (now removed) and returns `budget`.

- [ ] **Step 3: Commit test changes**

```bash
git add tests/agents/test_planner.py
git commit -m "test: update planner tests to assert normalized shape, delete coercion test"
```

---

### Task 4: Update `planner.py` — use SimplifiedPlan, normalize inline

**Files:**
- Modify: `src/deep_agents/agents/planner.py`

- [ ] **Step 1: Replace planner.py contents**

```python
"""Planner node — generates the initial research plan.

Reads research_goal, confirmed_constraints, open_dimensions, and language from state,
calls the LLM with structured output to produce a SimplifiedPlan, then normalizes the
result into the runtime shape expected by downstream nodes.
"""

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import planner_prompt
from deep_agents.schemas import SimplifiedPlan
from deep_agents.state import ResearchState
from deep_agents.utils import get_api_key_for_model, get_today_str


async def planner_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Generate the initial research plan from the research goal and constraints."""
    configurable = Configuration.from_runnable_config(config)

    model = (
        init_chat_model(
            model=configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            base_url=configurable.openai_compatible_base_url,
        )
        .with_structured_output(SimplifiedPlan)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    research_goal = state.get("research_goal", "")
    confirmed_constraints = state.get("confirmed_constraints", [])
    open_dimensions = state.get("open_dimensions", [])
    language = state.get("language", "zh")

    prompt_text = planner_prompt.format(
        date=get_today_str(),
        research_goal=research_goal,
        confirmed_constraints=confirmed_constraints,
        open_dimensions=open_dimensions,
        language=language,
    )

    plan: SimplifiedPlan = await model.ainvoke([HumanMessage(content=prompt_text)])

    hypotheses = [
        {
            "id": f"h_{i + 1}",
            "statement": h.statement,
            "evidence_needed": [],
            "status": "untested",
        }
        for i, h in enumerate(plan.hypotheses)
    ]

    sections = [
        {
            "id": f"sec_{i + 1}",
            "title": s.title,
            "description": s.description,
            "search_queries": s.search_queries,
            "priority": i + 1,
        }
        for i, s in enumerate(plan.sections)
    ]

    return {
        "research_type": plan.research_type,
        "hypotheses": hypotheses,
        "sections": sections,
        "outline_status": "provisional",
        "outline_revision_count": 0,
    }
```

- [ ] **Step 2: Run planner tests**

```bash
cd /home/czy/Karl/World_Fashion_Daily && uv run pytest tests/agents/test_planner.py -v
```

Expected: All three tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/deep_agents/agents/planner.py
git commit -m "feat: use SimplifiedPlan in planner_node, normalize IDs/status/priority inline"
```

---

### Task 5: Update `prompts.py` — simplify planner_prompt, remove max_searches from deep_scout_prompt

**Files:**
- Modify: `src/deep_agents/prompts.py`

- [ ] **Step 1: Replace `planner_prompt`**

In `prompts.py`, replace the entire `planner_prompt` string (lines 25–62) with:

```python
planner_prompt = """
今天的日期是 {date}。

研究目标：{research_goal}
已确认约束：{confirmed_constraints}
开放维度：{open_dimensions}
输出语言：{language}

你是时尚行业深度研究系统的架构规划师。请为上述研究目标制定完整研究计划。

任务：
1. 将研究分类为以下类型之一：
   trend_analysis（趋势分析）| brand_analysis（品牌分析）| market_overview（市场概况）| consumer_insight（消费者洞察）| competitive_landscape（竞争格局）
2. 生成 2-4 个待验证的研究假设，每个假设只需提供一句话陈述
3. 设计 3-6 个研究章节，每章节提供 2-4 个搜索词（中英文结合）

时尚研究指引：
- 趋势研究需覆盖：秀场、社交媒体、零售数据三个维度
- 品牌研究需覆盖：财报、品牌定位、消费者认知
- 优先引用：BoF、WWD、Vogue Runway、Lyst、Edited 等权威来源

输出约束：
- 只输出一个有效 JSON 对象，不要输出 Markdown、解释文字或代码块
- research_type 必须是以下之一：trend_analysis、brand_analysis、market_overview、consumer_insight、competitive_landscape
- hypotheses 是字符串数组，每条只包含假设陈述文字，不含 ID、状态、优先级或其他字段
- sections 每条包含：title、description、search_queries（字符串数组），不含 ID、priority 或其他字段

以有效 JSON 格式响应，包含字段：research_type, hypotheses（字符串数组）, sections（每条含 title/description/search_queries）
""".strip()
```

- [ ] **Step 2: Update `deep_scout_prompt` — remove max_searches line**

In `prompts.py`, replace the `deep_scout_prompt` string. Find the line `预算：最多 {max_searches} 次搜索` and the trailing `{max_searches}` parameter and remove them. The new `deep_scout_prompt` should be:

```python
deep_scout_prompt = """
今天的日期是 {date}。
整体研究目标：{research_goal}
当前章节：{section_title} — {section_description}
初始搜索词：
{search_queries}
待验证假设：
{hypotheses}

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
6. 结果重复或已足够全面时停止

重要规则：
- 保留矛盾信息，不要强行统一
- 标记 PR 宣传内容和赞助软文
- 不要忽略反驳工作假设的证据
""".strip()
```

- [ ] **Step 3: Run the section pipeline tests to catch any format errors**

```bash
cd /home/czy/Karl/World_Fashion_Daily && uv run pytest tests/agents/test_section_pipeline.py -v
```

Expected: Tests pass (deep_scout_node is mocked, so prompt format doesn't execute).

- [ ] **Step 4: Commit**

```bash
git add src/deep_agents/prompts.py
git commit -m "feat: simplify planner_prompt to content-only output, remove max_searches from deep_scout_prompt"
```

---

### Task 6: Update `test_state.py` — budget in removed_keys

**Files:**
- Modify: `tests/test_state.py`

- [ ] **Step 1: Update `test_section_state_matches_prd_shape`**

In `tests/test_state.py`, find `test_section_state_matches_prd_shape` and change `required_keys` and `removed_keys`:

```python
def test_section_state_matches_prd_shape() -> None:
    hints = get_type_hints(SectionState, include_extras=True)
    required_keys = {
        "section_id",
        "section_title",
        "section_description",
        "search_queries",
        "research_goal",
        "hypotheses",
        "language",
        "search_results",
        "section_facts",
        "section_insights",
        "section_hypothesis_evidence",
        "section_contradictions",
        "section_entities",
        "missing_info",
        "section_data_points",
        "section_charts",
        "section_time_series",
        "section_sources",
    }
    removed_keys = {
        "budget",
        "section_priority",
        "section_queries",
        "scout_output",
        "analyst_output",
        "data_wiz_output",
    }

    assert required_keys.issubset(set(hints.keys()))
    assert removed_keys.isdisjoint(set(hints.keys()))
```

Also add a test that `budget` is not in `ResearchState`:

```python
def test_research_state_has_no_budget_field() -> None:
    hints = get_type_hints(ResearchState, include_extras=True)
    assert "budget" not in hints
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/czy/Karl/World_Fashion_Daily && uv run pytest tests/test_state.py -v
```

Expected: `test_section_state_matches_prd_shape` fails (budget still in `SectionState`), `test_research_state_has_no_budget_field` fails (budget still in `ResearchState`).

- [ ] **Step 3: Commit test changes**

```bash
git add tests/test_state.py
git commit -m "test: assert budget removed from ResearchState and SectionState"
```

---

### Task 7: Update `state.py` — remove budget from both TypedDicts

**Files:**
- Modify: `src/deep_agents/state.py`

- [ ] **Step 1: Remove `budget: dict` from `ResearchState`**

In `src/deep_agents/state.py`, delete this line from `ResearchState`:

```python
    budget: dict
```

- [ ] **Step 2: Remove `budget: dict` from `SectionState`**

In `src/deep_agents/state.py`, delete this line from `SectionState`:

```python
    budget: dict
```

- [ ] **Step 3: Run state tests**

```bash
cd /home/czy/Karl/World_Fashion_Daily && uv run pytest tests/test_state.py -v
```

Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/deep_agents/state.py
git commit -m "feat: remove budget field from ResearchState and SectionState"
```

---

### Task 8: Update `graph.py` and `deep_scout.py` — remove budget from fan-out and prompt

**Files:**
- Modify: `src/deep_agents/graph.py`
- Modify: `src/deep_agents/agents/deep_scout.py`

- [ ] **Step 1: Remove budget from `section_pipeline_node` fan-out**

In `src/deep_agents/graph.py`, find the `section_inputs` list comprehension inside `section_pipeline_node`. Delete the line:

```python
            "budget": state.get("budget", {}),
```

- [ ] **Step 2: Remove max_searches from `deep_scout_node` prompt format**

In `src/deep_agents/agents/deep_scout.py`, find the `deep_scout_prompt.format(...)` call (around line 72). Remove the `max_searches=...` keyword argument:

```python
    system_prompt = deep_scout_prompt.format(
        date=get_today_str(),
        research_goal=state.get("research_goal", ""),
        section_title=state.get("section_title", ""),
        section_description=state.get("section_description", ""),
        search_queries="\n".join(state.get("search_queries", [])),
        hypotheses=json.dumps(state.get("hypotheses", []), ensure_ascii=False),
    )
```

- [ ] **Step 3: Run graph and section pipeline tests**

```bash
cd /home/czy/Karl/World_Fashion_Daily && uv run pytest tests/test_graph.py tests/agents/test_section_pipeline.py -v
```

Expected: Tests may still fail because test fixtures still pass `budget` in state. That's fine — the runtime code is correct now. Fixture cleanup is next.

- [ ] **Step 4: Commit**

```bash
git add src/deep_agents/graph.py src/deep_agents/agents/deep_scout.py
git commit -m "feat: remove budget from section_pipeline fan-out and deep_scout prompt"
```

---

### Task 9: Update test fixtures — remove budget from all test state dicts

**Files:**
- Modify: `tests/agents/conftest.py`
- Modify: `tests/agents/test_section_pipeline.py`
- Modify: `tests/test_graph.py`

- [ ] **Step 1: Update `tests/agents/conftest.py`**

Remove the `budget` key from the `sample_state` fixture. Change:

```python
        "hypotheses": [],
        "budget": {"max_parallel": 2, "max_searches": 5, "max_deep_reads": 3},
        "facts": [],
```

To:

```python
        "hypotheses": [],
        "facts": [],
```

- [ ] **Step 2: Update `tests/agents/test_section_pipeline.py`**

In the `section_state` fixture, remove:

```python
        "budget": {"max_searches": 5},
```

- [ ] **Step 3: Update `tests/test_graph.py`**

There are four state dicts that contain `"budget"`. Remove the budget line from each:

In `test_section_pipeline_node_merges_results` (around line 65):
```python
        # remove: "budget": {"max_searches": 5},
```

In `test_section_pipeline_node_handles_failed_section` (around line 129):
```python
        # remove: "budget": {"max_searches": 5},
```

In `test_section_pipeline_routes_to_outline_reviser_when_refuted` (around line 171):
```python
        # remove: "budget": {},
```

In `test_section_pipeline_does_not_route_to_outline_reviser_when_count_at_1` (around line 211):
```python
        # remove: "budget": {},
```

- [ ] **Step 4: Run full test suite**

```bash
cd /home/czy/Karl/World_Fashion_Daily && uv run pytest -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/agents/conftest.py tests/agents/test_section_pipeline.py tests/test_graph.py
git commit -m "test: remove budget from all test fixtures"
```

---

## Self-Review Against Spec

| Spec requirement | Covered by |
|-----------------|------------|
| Planner outputs only `research_type`, hypothesis statements, section title/description/search_queries | Task 2 (schemas), Task 4 (planner.py), Task 5 (prompt) |
| Prompt must explicitly stop requesting `evidence_needed`, `status`, `priority`, `outline_status`, `budget`, IDs | Task 5 |
| `planner_node` synthesizes IDs `h_1…`, `sec_1…`, status `"untested"`, `evidence_needed=[]`, priority `1..n`, `outline_status="provisional"`, `outline_revision_count=0` | Task 4; verified by Task 3 tests |
| No coercion of legacy values — fail fast if they appear | Task 3 (coercion test deleted); new `planner.py` has no coercion code |
| Remove `budget` from `ResearchState` | Task 7 |
| Remove `budget` from `SectionState` | Task 7 |
| Remove `budget` from graph fan-out payloads | Task 8 |
| `section_pipeline_node` stops copying `state["budget"]` | Task 8 |
| Do not add `max_searches` to `Configuration` | Not adding it (no task needed) |
| Remove `max_searches` hint from `deep_scout_prompt` | Task 5 |
| `deep_scout` no longer reads `budget` from state | Task 8 |
| `research_type` remains prompt-constrained (not schema-level Literal) | Task 2 — `SimplifiedPlan.research_type` is `str`, not `Literal` |
| Remove/update all tests referencing `budget` or `ArchitectPlan`/`Hypothesis` | Tasks 1, 3, 6, 9 |
| Verification: raw LLM output has no `id`/`status`/`priority`/`budget`/`outline_status`/`evidence_needed` | Prompt updated in Task 5; schema enforced by Task 2 |
| Verification: state after planner_node has normalized shape | Asserted in Task 3 tests |
