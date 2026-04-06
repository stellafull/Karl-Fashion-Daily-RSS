# URL-Based Citations And Parallel Writer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `source_id`-based provenance flow with simple URL-based citations and convert section writing from sequential to parallel while keeping the overall LangGraph topology intact.

**Architecture:** The collection subgraph will continue to produce section-local evidence, but all citation linkage will be URL-based instead of synthetic source IDs. `deep_scout` will emit canonical sources as `url/title/summary`; `analyst` and `data_wiz` will emit strong typed records keyed by `source_url`. `writer` will fan out in parallel using only section-local inputs plus `sections_list`, and `synthesizer` will remain the sole whole-report unification layer.

**Tech Stack:** Python 3.12, Pydantic v2, LangGraph, LangChain, pytest + pytest-asyncio, uv

---

## File Map

| File | Change |
|------|--------|
| `src/deep_agents/schemas.py` | Replace loose dict-based evidence outputs with strong URL-based models; remove `DataPoint.id`; remove model-facing `section_id` from `SectionDraft`; switch hypothesis evidence to `hypothesis_statement` |
| `src/deep_agents/prompts.py` | Remove `source_id` wording, remove fixed media white-list, remove writer `section_id`, remove previous-section-content dependency, update writer to parallel-friendly inputs |
| `src/deep_agents/agents/deep_scout.py` | Remove synthetic source metadata (`source_id`, `credibility_score`); emit only canonical `url/title/summary` sources |
| `src/deep_agents/agents/analyst.py` | Consume strong typed `AnalystOutput`; enforce URL-based facts/evidence/contradictions |
| `src/deep_agents/agents/data_wiz.py` | Consume strong typed `DataWizOutput`; enforce URL-based data points |
| `src/deep_agents/graph.py` | Remove `source_id` remap logic; replace with optional URL existence validation; keep section tagging only |
| `src/deep_agents/agents/writer.py` | Replace sequential loop context coupling with parallel section drafting; remove `previous_sections_memo`; append runtime `section_id` after model validation |
| `src/deep_agents/agents/reviewer.py` | Update prompt payload assumptions if claim checks remain URL-based |
| `src/deep_agents/state.py` | Remove `failed_sections` if no longer used after fail-fast policy |
| `tests/test_schemas.py` | Add strong typed URL-based schema tests |
| `tests/agents/test_section_pipeline.py` | Replace `source_id`-based expectations with `source_url`-based expectations |
| `tests/agents/test_deep_scout.py` | Assert canonical source shape only |
| `tests/agents/test_writing.py` | Assert parallel writer behavior and prompt isolation |
| `tests/test_graph.py` | Remove `source_id`-based merge assertions and replace with URL-based merge validation assertions |
| `tests/agents/test_quality.py` | Update any URL-based citation / claim-check assumptions |

---

### Task 1: Replace Dict-Based Evidence Schemas With Strong URL-Based Models

**Files:**
- Modify: `src/deep_agents/schemas.py`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Write failing schema tests for the new URL-based evidence contract**

Add tests in `tests/test_schemas.py` for:

```python
from pydantic import ValidationError

from deep_agents.schemas import (
    Source,
    SectionFact,
    HypothesisEvidence,
    Contradiction,
    DataPoint,
    SectionDraft,
)


def test_source_schema_requires_url_title_summary() -> None:
    model = Source(
        url="https://example.com/a",
        title="Example",
        summary="Summary text",
    )
    assert model.url == "https://example.com/a"


def test_section_fact_requires_source_url() -> None:
    model = SectionFact(
        content="销量增长",
        source_url="https://example.com/a",
        importance="high",
    )
    assert model.source_url == "https://example.com/a"


def test_hypothesis_evidence_uses_statement_not_id() -> None:
    model = HypothesisEvidence(
        hypothesis_statement="消费者偏好转向功能性服饰",
        evidence_type="supports",
        content="多个来源提到功能性需求增强",
        source_url="https://example.com/b",
    )
    assert model.hypothesis_statement.startswith("消费者偏好")


def test_data_point_has_no_runtime_id_field() -> None:
    model = DataPoint(
        name="market_size",
        value=3457,
        source_url="https://example.com/c",
    )
    dumped = model.model_dump()
    assert "id" not in dumped


def test_section_draft_has_no_model_facing_section_id() -> None:
    model = SectionDraft(
        content="## 市场概况",
        citations=[{"claim": "增长", "url": "https://example.com/a", "title": "A"}],
        charts_used=[],
        weak_claims=[],
    )
    dumped = model.model_dump()
    assert "section_id" not in dumped


def test_old_source_id_payload_is_rejected() -> None:
    try:
        SectionFact(
            content="增长",
            source_id="src_001",
            importance="high",
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("Expected ValidationError for old source_id payload")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/czy/Karl/World_Fashion_Daily && /home/czy/Karl/World_Fashion_Daily/.venv/bin/pytest tests/test_schemas.py -q
```

Expected:
- Import errors or validation failures because `Source`, `SectionFact`, `HypothesisEvidence`, `Contradiction`, and the new `SectionDraft` contract do not exist yet.

- [ ] **Step 3: Implement the minimal schema changes**

Update `src/deep_agents/schemas.py` to add:

```python
class Source(BaseModel):
    url: str
    title: str
    summary: str


class SectionFact(BaseModel):
    content: str
    source_url: str
    importance: str


class HypothesisEvidence(BaseModel):
    hypothesis_statement: str
    evidence_type: str
    content: str
    source_url: str


class Contradiction(BaseModel):
    claim_a: str
    claim_b: str
    source_url_a: str
    source_url_b: str


class DataPoint(BaseModel):
    name: str
    value: str | int | float
    unit: str | None = None
    year: int | None = None
    source_url: str
    category: str | None = None
    confidence: str | None = None
```

And replace:

```python
class AnalystOutput(BaseModel):
    section_facts: list[dict] = Field(default_factory=list)
    section_insights: list[str] = Field(default_factory=list)
    section_hypothesis_evidence: list[dict] = Field(default_factory=list)
    section_contradictions: list[dict] = Field(default_factory=list)
    section_entities: list[dict] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)


class DataWizOutput(BaseModel):
    section_data_points: list[dict] = Field(default_factory=list)
    section_charts: list[dict] = Field(default_factory=list)
    section_time_series: list[dict] = Field(default_factory=list)


class SectionDraft(BaseModel):
    section_id: str
    content: str
    citations: list[dict] = Field(default_factory=list)
    charts_used: list[str] = Field(default_factory=list)
    weak_claims: list[str] = Field(default_factory=list)
```

with:

```python
class AnalystOutput(BaseModel):
    section_facts: list[SectionFact] = Field(default_factory=list)
    section_insights: list[str] = Field(default_factory=list)
    section_hypothesis_evidence: list[HypothesisEvidence] = Field(default_factory=list)
    section_contradictions: list[Contradiction] = Field(default_factory=list)
    section_entities: list[dict] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)


class DataWizOutput(BaseModel):
    section_data_points: list[DataPoint] = Field(default_factory=list)
    section_charts: list[dict] = Field(default_factory=list)
    section_time_series: list[dict] = Field(default_factory=list)


class Citation(BaseModel):
    claim: str
    url: str
    title: str


class SectionDraft(BaseModel):
    content: str
    citations: list[Citation] = Field(default_factory=list)
    charts_used: list[str] = Field(default_factory=list)
    weak_claims: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /home/czy/Karl/World_Fashion_Daily && /home/czy/Karl/World_Fashion_Daily/.venv/bin/pytest tests/test_schemas.py -q
```

Expected:
- The new schema tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/deep_agents/schemas.py tests/test_schemas.py
git commit -m "refactor: replace source_id schemas with url-based evidence models"
```

---

### Task 2: Simplify Search Sources To Canonical `url/title/summary`

**Files:**
- Modify: `src/deep_agents/agents/deep_scout.py`
- Modify: `src/deep_agents/utils.py`
- Modify: `tests/agents/test_deep_scout.py`

- [ ] **Step 1: Write failing deep_scout tests for canonical source shape**

Add tests in `tests/agents/test_deep_scout.py`:

```python
async def test_deep_scout_emits_canonical_sources_without_source_id(section_state, mock_config):
    from deep_agents.agents.deep_scout import deep_scout_node
    from langchain_core.messages import AIMessage

    search_tool = MagicMock()
    search_tool.name = "web_search"
    search_tool.ainvoke = AsyncMock(
        return_value={
            "results": [
                {
                    "url": "https://example.com/article",
                    "title": "Example Article",
                    "summary": "A concise summary",
                }
            ]
        }
    )

    mock_ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "web_search", "args": {"queries": ["q"]}, "id": "tc1", "type": "tool_call"},
            {"name": "ResearchComplete", "args": {"reason": "done"}, "id": "tc2", "type": "tool_call"},
        ],
    )
    mock_bound = MagicMock()
    mock_bound.ainvoke = AsyncMock(side_effect=[mock_ai])
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_bound

    with patch("deep_agents.agents.deep_scout.get_all_tools", AsyncMock(return_value=[search_tool])):
        with patch("deep_agents.agents.deep_scout.init_chat_model", return_value=mock_model):
            result = await deep_scout_node(section_state, mock_config)

    source = result["section_sources"][0]
    assert set(source.keys()) == {"url", "title", "summary"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/czy/Karl/World_Fashion_Daily && /home/czy/Karl/World_Fashion_Daily/.venv/bin/pytest tests/agents/test_deep_scout.py -q
```

Expected:
- Failure because `deep_scout` still emits `source_id`, `section_id`, or `credibility_score`.

- [ ] **Step 3: Implement the minimal source simplification**

In `src/deep_agents/agents/deep_scout.py`, replace the current source record:

```python
source_record = {
    "source_id": ...,
    "url": ...,
    "title": ...,
    "summary": ...,
    "credibility_score": 0.7,
    "section_id": section_id,
}
```

with:

```python
source_record = {
    "url": url,
    "title": item.get("title", ""),
    "summary": item.get("summary") or item.get("content", ""),
}
```

`search_result["sources"]` should hold the same canonical shape.

Do not add URL normalization rules. Use the Tavily-returned URL directly after existing string/empty checks.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /home/czy/Karl/World_Fashion_Daily && /home/czy/Karl/World_Fashion_Daily/.venv/bin/pytest tests/agents/test_deep_scout.py -q
```

Expected:
- DeepScout tests pass with canonical source shape only.

- [ ] **Step 5: Commit**

```bash
git add src/deep_agents/agents/deep_scout.py tests/agents/test_deep_scout.py
git commit -m "refactor: simplify deep scout sources to url title summary"
```

---

### Task 3: Replace `source_id` Contracts In Analyst And DataWiz With URL-Based Contracts

**Files:**
- Modify: `src/deep_agents/prompts.py`
- Modify: `src/deep_agents/agents/analyst.py`
- Modify: `src/deep_agents/agents/data_wiz.py`
- Modify: `tests/agents/test_section_pipeline.py`

- [ ] **Step 1: Write failing tests for URL-based evidence/data outputs**

Update tests in `tests/agents/test_section_pipeline.py` so analyst/data_wiz expect:

```python
mock_out = AnalystOutput(
    section_facts=[{"content": "3620亿元", "source_url": "https://example.com/s1", "importance": "high"}],
    section_hypothesis_evidence=[
        {
            "hypothesis_statement": "市场规模持续增长",
            "evidence_type": "supports",
            "content": "多个来源支持",
            "source_url": "https://example.com/s1",
        }
    ],
)
```

and:

```python
mock_out = DataWizOutput(
    section_data_points=[{"name": "market_size", "value": 3620, "source_url": "https://example.com/s1"}]
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/czy/Karl/World_Fashion_Daily && /home/czy/Karl/World_Fashion_Daily/.venv/bin/pytest tests/agents/test_section_pipeline.py -q
```

Expected:
- Failures because prompts/schemas/runtime still assume `source_id`.

- [ ] **Step 3: Update prompts and runtime to URL-based outputs**

In `src/deep_agents/prompts.py`:

- Replace:

```text
section_facts（每条含 content/source_id/importance）
section_hypothesis_evidence（每条含 hypothesis_id/evidence_type/content/source_id）
section_contradictions（每条含 claim_a/claim_b/source_id_a/source_id_b）
```

with:

```text
section_facts（每条含 content/source_url/importance）
section_hypothesis_evidence（每条含 hypothesis_statement/evidence_type/content/source_url）
section_contradictions（每条含 claim_a/claim_b/source_url_a/source_url_b）
```

- Replace:

```text
所有数据点必须有 source_id
section_data_points（每条含 id/name/value/unit/year/source_id/category/confidence）
```

with:

```text
所有数据点必须有 source_url
section_data_points（每条含 name/value/unit/year/source_url/category/confidence）
```

Do not change overall node responsibilities.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /home/czy/Karl/World_Fashion_Daily && /home/czy/Karl/World_Fashion_Daily/.venv/bin/pytest tests/agents/test_section_pipeline.py -q
```

Expected:
- Analyst/DataWiz tests pass with URL-based fields.

- [ ] **Step 5: Commit**

```bash
git add src/deep_agents/prompts.py src/deep_agents/agents/analyst.py src/deep_agents/agents/data_wiz.py tests/agents/test_section_pipeline.py
git commit -m "refactor: switch analyst and datawiz outputs to source_url"
```

---

### Task 4: Simplify Section Merge To URL-Based Validation Only

**Files:**
- Modify: `src/deep_agents/graph.py`
- Modify: `tests/test_graph.py`
- Modify: `tests/agents/test_section_pipeline.py`

- [ ] **Step 1: Write failing graph tests for URL-based merge**

Replace `source_id`-based graph tests with URL-based assertions. Add tests like:

```python
async def test_section_pipeline_validates_source_url_against_section_sources() -> None:
    from deep_agents.graph import section_pipeline_node

    section_result = {
        "section_facts": [{"content": "fact1", "source_url": "https://example.com/a"}],
        "section_data_points": [],
        "section_hypothesis_evidence": [],
        "section_charts": [],
        "section_insights": [],
        "section_contradictions": [],
        "section_sources": [{"url": "https://example.com/a", "title": "A", "summary": "S"}],
        "missing_info": [],
    }
    ...
    assert result.update["facts"]["value"][0]["section_id"] == "sec_1"
```

And failure case:

```python
with pytest.raises(ValueError, match="Unresolved source_url"):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/czy/Karl/World_Fashion_Daily && /home/czy/Karl/World_Fashion_Daily/.venv/bin/pytest tests/test_graph.py tests/agents/test_section_pipeline.py -q
```

Expected:
- Failures because graph still expects/remaps `source_id`.

- [ ] **Step 3: Implement minimal URL-based merge logic**

In `src/deep_agents/graph.py`:

- Delete:
  - `_global_source_id`
  - `seen_source_ids`
  - `source_id_map`
  - all `source_id`/`source_id_a`/`source_id_b` remap logic

- Replace with URL-based validation:

```python
section_source_urls = {
    src["url"]
    for src in section_sources
    if isinstance(src, dict) and isinstance(src.get("url"), str) and src.get("url")
}
```

And in merge mapping:

```python
source_url = mapped.get("source_url")
if require_source_url and (not isinstance(source_url, str) or not source_url.strip()):
    raise ValueError(f"Missing required source_url in section '{section_id}'")
if source_url is not None and source_url not in section_source_urls:
    raise ValueError(f"Unresolved source_url '{source_url}' in section '{section_id}'")
```

Equivalent checks for `source_url_a` / `source_url_b`.

Section tagging stays. Override-envelope merge behavior stays.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /home/czy/Karl/World_Fashion_Daily && /home/czy/Karl/World_Fashion_Daily/.venv/bin/pytest tests/test_graph.py tests/agents/test_section_pipeline.py -q
```

Expected:
- Graph and section pipeline tests pass with URL-based merge validation.

- [ ] **Step 5: Commit**

```bash
git add src/deep_agents/graph.py tests/test_graph.py tests/agents/test_section_pipeline.py
git commit -m "refactor: replace source id merge logic with source url validation"
```

---

### Task 5: Parallelize Writer And Remove Model-Facing Runtime Fields

**Files:**
- Modify: `src/deep_agents/prompts.py`
- Modify: `src/deep_agents/agents/writer.py`
- Modify: `tests/agents/test_writing.py`

- [ ] **Step 1: Write failing tests for parallel writer contract**

Add or replace tests in `tests/agents/test_writing.py` so they verify:

1. Prompt payload does not contain `section_id`
2. Prompt payload does not contain previous section body content
3. Each section only receives section-local facts/data/charts/contradictions/hypothesis evidence
4. `writer_node` still returns drafts with runtime-appended `section_id`

Example test shape:

```python
async def test_writer_does_not_put_runtime_section_id_in_prompt(mock_config):
    ...
    captured_prompts = []

    async def fake_ainvoke(messages):
        captured_prompts.append(messages[0].content)
        return SectionDraft(content="body", citations=[], charts_used=[], weak_claims=[])

    ...
    assert "章节ID：" not in captured_prompts[0]
```

And:

```python
async def test_writer_runs_sections_concurrently(mock_config):
    ...
    # Use events or controlled sleeps to prove the second section starts before the first finishes.
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/czy/Karl/World_Fashion_Daily && /home/czy/Karl/World_Fashion_Daily/.venv/bin/pytest tests/agents/test_writing.py -q
```

Expected:
- Failures because writer is still sequential or still puts runtime scaffolding into the prompt.

- [ ] **Step 3: Implement minimal parallel writer**

In `src/deep_agents/prompts.py`:

- Remove `章节ID：{section_id}`
- Remove any requirement for model to return `section_id`
- Keep `sections_list`
- Keep current section-local evidence only
- Keep anti-dup guidance at the whole-report map level, not prior content level

In `src/deep_agents/agents/writer.py`:

- Delete `previous_sections_memo`
- Delete sequential cross-section context dependence
- Build one async write task per section
- Pass only section-local inputs plus `sections_list`
- After each `SectionDraft` returns, append runtime `section_id` into the result dict
- Emit `section_done` custom stream events per finished section as before

Use `asyncio.gather(...)` for section draft fan-out.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /home/czy/Karl/World_Fashion_Daily && /home/czy/Karl/World_Fashion_Daily/.venv/bin/pytest tests/agents/test_writing.py tests/test_api.py -q
```

Expected:
- Writer tests pass
- API still forwards `section_done` events

- [ ] **Step 5: Commit**

```bash
git add src/deep_agents/prompts.py src/deep_agents/agents/writer.py tests/agents/test_writing.py tests/test_api.py
git commit -m "refactor: parallelize writer with url-based citation contract"
```

---

### Task 6: Update Reviewer / Final Citation Consumers And Remove Dead State

**Files:**
- Modify: `src/deep_agents/prompts.py`
- Modify: `src/deep_agents/agents/reviewer.py`
- Modify: `src/deep_agents/state.py`
- Modify: `tests/agents/test_quality.py`
- Modify: `tests/test_state.py`

- [ ] **Step 1: Write failing tests for URL-based review assumptions**

Add/update tests so reviewer-side payloads and state assumptions no longer depend on `source_id` or `failed_sections`.

Add a state test that `failed_sections` is absent if you remove it from `ResearchState`.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/czy/Karl/World_Fashion_Daily && /home/czy/Karl/World_Fashion_Daily/.venv/bin/pytest tests/agents/test_quality.py tests/test_state.py -q
```

Expected:
- Failures because old prompt/state expectations remain.

- [ ] **Step 3: Implement minimal cleanup**

In `src/deep_agents/prompts.py`:

- Replace reviewer wording that mentions `source_id` with URL-based wording

In `src/deep_agents/state.py`:

- Remove `failed_sections` if it is no longer used under fail-fast graph policy

Keep runtime `hypotheses[*]["id"]` and `sections[*]["id"]` internal for graph logic unless later tasks remove them too.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /home/czy/Karl/World_Fashion_Daily && /home/czy/Karl/World_Fashion_Daily/.venv/bin/pytest tests/agents/test_quality.py tests/test_state.py -q
```

Expected:
- Reviewer and state tests pass with URL-based wording and cleaned state.

- [ ] **Step 5: Commit**

```bash
git add src/deep_agents/prompts.py src/deep_agents/agents/reviewer.py src/deep_agents/state.py tests/agents/test_quality.py tests/test_state.py
git commit -m "refactor: remove source id review assumptions and dead state"
```

---

### Task 7: Full Verification

**Files:**
- Verify only: `src/deep_agents/**/*.py`
- Verify only: `tests/**/*.py`

- [ ] **Step 1: Run the full test suite**

Run:

```bash
cd /home/czy/Karl/World_Fashion_Daily && /home/czy/Karl/World_Fashion_Daily/.venv/bin/pytest -q
```

Expected:
- Full test suite passes.

- [ ] **Step 2: Run compile check**

Run:

```bash
cd /home/czy/Karl/World_Fashion_Daily && /home/czy/Karl/World_Fashion_Daily/.venv/bin/python -m compileall src
```

Expected:
- Source tree compiles cleanly.

- [ ] **Step 3: Run configuration sanity check**

Run:

```bash
cd /home/czy/Karl/World_Fashion_Daily && /home/czy/Karl/World_Fashion_Daily/.venv/bin/python -c "from deep_agents.configuration import Configuration; print(Configuration())"
```

Expected:
- Configuration object prints successfully.

- [ ] **Step 4: Commit final verification-only changes if needed**

If verification required no extra changes, skip commit.
If any tiny follow-up fix was needed, commit it:

```bash
git add <touched-files>
git commit -m "chore: finalize url-based citations and parallel writer cleanup"
```
