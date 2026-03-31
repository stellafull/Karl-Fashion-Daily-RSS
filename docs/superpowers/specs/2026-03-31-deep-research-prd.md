# Fashion Deep Research Agent — PRD

**Version:** 1.0  
**Date:** 2026-03-31  
**Status:** Approved — ready for implementation  
**Supersedes:** `docs/spec/deep_research.md` (architecture reference only)

---

## 1. Overview

A fashion deep research system built on **LangChain + LangGraph** that:

- Accepts a compact message history + optional image from a larger host system
- Clarifies research intent (one question max, human-in-the-loop via host system)
- Plans hypotheses and sections
- Collects evidence in parallel by section
- Writes and reviews a coherent research report
- Streams progress back to the host system via SSE (FastAPI)
- Returns a final `full_report` string the host system appends to its message history

### Design constraints

- This is a **standalone FastAPI service** — the host system calls it over HTTP
- All internal agent prompts are in **Chinese**
- Raw conversation messages are only handled in `clarify.py` — all downstream nodes receive `research_goal: str` only
- `configuration.py` and `utils.py` are kept as-is (LangChain foundation)
- No `BaseAgent` class — every agent is a plain `async def node(state, config) -> dict`
- Primary model: **Kimi 2.5** (`openai:kimi/kimi-k2.5`) for reasoning/writing
- Cost-optimized model: **Qwen** (`openai:qwen3.5-flash`) for summarization/extraction

---

## 2. File Layout

```
src/deep_agents/
├── configuration.py        # ✅ keep as-is
├── utils.py                # ✅ keep as-is + add analyze_image tool
├── schemas.py              # NEW: all Pydantic structured output models
├── state.py                # REPLACE: ResearchState + SectionState
├── prompts.py              # REPLACE: all Chinese prompts, one per agent
├── graph.py                # NEW: LangGraph builder + conditional edges
├── api.py                  # NEW: FastAPI app, SSE streaming endpoint
└── agents/
    ├── clarify.py          # entry node: messages + image → research_goal or question
    ├── planner.py          # architect_planner: research_goal → sections + hypotheses
    ├── outline_reviser.py  # outline_reviser: adjust sections after collection signals
    ├── deep_scout.py       # per-section tool-calling research agent
    ├── analyst.py          # per-section qualitative analysis
    ├── data_wiz.py         # per-section data extraction + chart configs
    ├── writer.py           # lead_writer: one section draft at a time
    ├── synthesizer.py      # merge all drafts into full report
    ├── trend_triangulator.py  # conditional: validate trend claims (trend_analysis only)
    ├── reviewer.py         # strict quality review + scoring
    ├── reviser.py          # targeted edits based on reviewer feedback
    └── final_check.py      # final gate: verify fixes, assign publication readiness
```

**Source mapping from existing copied code:**

| New file | Adapted from |
|---|---|
| `planner.py` | `architect.py` → `_initial_planning` |
| `outline_reviser.py` | `architect.py` → `_check_revision` |
| `deep_scout.py` | `scout.py` |
| `analyst.py` | `analyst.py` |
| `data_wiz.py` | `wizard.py` + `data_analyst.py` |
| `writer.py` | `writer.py` |
| `synthesizer.py` | `writer.py` (synthesis section) |
| `reviewer.py` | `critic.py` |
| `reviser.py` | `critic.py` |
| `final_check.py` | `critic.py` |
| `clarify.py` | existing `clarify.py` (incomplete, rewrite) |
| `trend_triangulator.py` | NEW |

**Files dropped from existing code:**
- `base.py` — `BaseAgent`, `call_llm`, `parse_json_response`, `AgentRegistry` all removed
- The `asyncio.Queue` SSE mechanism in state — replaced by LangGraph `.astream_events()`
- Phase enum (`ResearchPhase`) — replaced by LangGraph routing

---

## 3. State Schemas (`state.py`)

### ResearchState — main graph

```python
from typing import Annotated
import operator
from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import add_messages
from typing_extensions import TypedDict

class ResearchState(TypedDict):
    # Entry (input from host system, read only in clarify.py)
    messages: Annotated[list, add_messages]
    object_context: str | None          # image URL or None

    # Clarification
    need_clarification: bool
    clarification_question: str

    # Research goal (written by clarify.py, read by all downstream nodes)
    research_goal: str
    confirmed_constraints: list[str]
    open_dimensions: list[str]
    language: str                        # detected from messages

    # Planning (written by planner.py)
    research_type: str                   # "trend_analysis" | "brand_analysis" | "market_overview" | ...
    hypotheses: list[dict]
    sections: list[dict]
    budget: dict
    outline_status: str                  # "provisional" | "revised"
    outline_revision_count: int

    # Collection — parallel-write fields (operator.add reducers)
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
```

### SectionState — section subgraph

```python
class SectionState(TypedDict):
    # Inputs from parent (set by fan_out_sections)
    section_id: str
    section_title: str
    section_description: str
    search_queries: list[str]
    research_goal: str
    hypotheses: list[dict]
    budget: dict
    language: str

    # DeepScout output
    search_results: list[dict]

    # Analyst output
    section_facts: list[dict]
    section_insights: list[str]
    section_hypothesis_evidence: list[dict]
    section_contradictions: list[dict]
    section_entities: list[dict]
    missing_info: list[str]

    # DataWiz output
    section_data_points: list[dict]
    section_charts: list[dict]
    section_time_series: list[dict]

    # Sources
    section_sources: list[dict]
```

---

## 4. Data Schemas (`schemas.py`)

All are Pydantic `BaseModel` — used with `.with_structured_output()`.

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class ResearchBrief(BaseModel):
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
    status: str = "untested"   # untested | supported | refuted | partial

class Section(BaseModel):
    id: str
    title: str
    description: str
    search_queries: List[str]
    priority: int

class ArchitectPlan(BaseModel):
    research_type: str
    hypotheses: List[Hypothesis]
    sections: List[Section]
    budget: dict
    outline_status: str = "provisional"

class AnalystOutput(BaseModel):
    section_facts: List[dict]
    section_insights: List[str]
    section_hypothesis_evidence: List[dict]
    section_contradictions: List[dict]
    section_entities: List[dict]
    missing_info: List[str]

class DataWizOutput(BaseModel):
    section_data_points: List[dict]
    section_charts: List[dict]
    section_time_series: List[dict]

class SectionDraft(BaseModel):
    section_id: str
    content: str
    citations: List[dict]
    charts_used: List[str]
    weak_claims: List[str]

class ReviewResult(BaseModel):
    quality_score: int              # 1–10
    verdict: str                    # "pass" | "fail"
    issues: List[dict]
    claim_checks: List[dict]
    missing_aspects: List[str]

class FinalResult(BaseModel):
    resolved_issues: List[dict]
    unresolved_issues: List[dict]
    new_issues: List[dict]
    final_score: int
    final_verdict: str              # "approved" | "rejected"
    publication_readiness: str      # "ready" | "needs_review"
    final_comments: str
```

---

## 5. Agent Nodes

Every node follows this signature:

```python
async def <name>_node(state: ResearchState, config: RunnableConfig) -> dict:
    ...
    return {"field": value, ...}
```

Returns only the fields it writes — LangGraph merges into state.

### 5.1 clarify.py

**Reads:** `messages`, `object_context`  
**Writes:** `need_clarification`, `clarification_question`, `research_goal`, `confirmed_constraints`, `open_dimensions`, `language`

Two-step, single LLM call with structured output (`ResearchBrief`):
1. Does the conversation need clarification? If yes → set `need_clarification=True`, write question, return.
2. If no → extract clean `research_goal` from messages (and image if present). All downstream nodes only see `research_goal`.

Image handling: if `object_context` is set, pass image URL to Kimi 2.5 multimodal alongside messages.

### 5.2 planner.py

**Reads:** `research_goal`, `confirmed_constraints`, `open_dimensions`, `language`  
**Writes:** `research_type`, `hypotheses`, `sections`, `budget`, `outline_status`, `outline_revision_count`

Structured output: `ArchitectPlan`. Retries up to `max_structured_output_retries` on failure.

Fashion-specific guidance in prompt:
- Classify `research_type`: `trend_analysis` | `brand_analysis` | `market_overview` | `consumer_insight` | `competitive_landscape`
- Generate 2–4 provisional hypotheses
- Design 3–6 sections with 2–4 search queries each (include Chinese + English queries)
- Budget: simple → `max_parallel=2`, medium → `max_parallel=3`, complex → `max_parallel=4`

### 5.3 outline_reviser.py

**Reads:** `research_goal`, `sections`, `hypothesis_evidence`, `outline_status`, `outline_revision_count`  
**Writes:** `sections`, `outline_status`, `outline_revision_count`

Only triggers when `should_revise_outline()` returns `True` (2+ refuted hypotheses, revision_count < 1). Minimal changes — add/remove/reorder sections, preserve section IDs.

### 5.4 deep_scout.py

**Runs inside section subgraph.**  
**Reads:** full `SectionState`  
**Writes:** `search_results`, `section_sources`

Tool-calling ReAct agent. Tools available:
- `tavily_search` — search + full content fetch + Qwen summarization
- `think_tool` — reflection between searches
- `analyze_image` — Kimi 2.5 multimodal for runway/lookbook/social images

Strategy: run provided queries → `think_tool` after each → deep-read tier-1/2 sources → use `analyze_image` for visual trend topics → stop when budget hit or results are repetitive.

### 5.5 analyst.py

**Runs inside section subgraph.**  
**Reads:** `search_results`, `hypotheses`, `section_title`, `section_description`, `research_goal`  
**Writes:** `section_facts`, `section_insights`, `section_hypothesis_evidence`, `section_contradictions`, `section_entities`, `missing_info`

Structured output: `AnalystOutput`. Preserves contradictions — does not resolve them.

### 5.6 data_wiz.py

**Runs inside section subgraph.**  
**Reads:** `search_results`, `section_title`, `research_goal`  
**Writes:** `section_data_points`, `section_charts`, `section_time_series`

Structured output: `DataWizOutput`. Only extracts clearly sourced numbers. Generates ECharts configs for the most useful data views.

### 5.7 writer.py

**Reads:** `sections`, `facts`, `data_points`, `charts`, `contradictions`, `hypothesis_evidence`, `research_goal`, `language`  
**Writes:** `section_drafts` (all drafts in one call)

`lead_writer` is a single node that iterates over all sections internally in one LLM-per-section loop. For each section it builds a prompt with the current section's evidence packet + a short memo of previously written sections (to avoid repetition). Returns the full `section_drafts` list at the end.

Structured output per section: `SectionDraft`. 500–1000 words per section. Cites every key claim. Flags thin evidence in `weak_claims`. Writes in detected `language`.

### 5.8 synthesizer.py

**Reads:** `section_drafts`, `hypothesis_evidence`, `contradictions`, `sources`, `research_goal`  
**Writes:** `full_report`

Produces: executive summary + merged sections + conclusions with hypothesis verdicts + unresolved questions + numbered reference list with clickable links.

### 5.9 trend_triangulator.py

**Reads:** `full_report`, `facts`, `sources`  
**Writes:** `full_report` (revised in place)

Only runs when `research_type == "trend_analysis"`. Validates each trend claim against 3 signal types:
1. Designer/runway signal
2. Street/social adoption
3. Commercial/retail data

Claims with only one signal type are marked weak or emerging.

### 5.10 reviewer.py

**Reads:** `full_report`, `facts`, `data_points`, `sections`, `research_goal`, `revision_count`  
**Writes:** `review_result`

Structured output: `ReviewResult`. Scoring:
- 9–10: publish-ready
- 7–8: pass with minor issues
- 5–6: needs revision
- 1–4: major problems

`quality_score >= 7` → verdict `"pass"`.

### 5.11 reviser.py

**Reads:** `full_report`, `review_result`  
**Writes:** `full_report`, `revision_count` (incremented)

Targeted edits only. Adds support where evidence exists. Does not invent information.

### 5.12 final_check.py

**Reads:** `full_report`, `review_result`, `revision_count`, `research_goal`  
**Writes:** `final_result`

Structured output: `FinalResult`. Verifies previous issues resolved, checks for new issues introduced during revision. If `revision_count >= 2` and still failing, marks `publication_readiness: "needs_review"` rather than blocking.

---

## 6. Tool Layer (`utils.py` additions)

Two tools already exist: `tavily_search`, `think_tool`.

One tool to add:

### analyze_image

```python
@tool(description="分析时尚图片，包括秀场、lookbook和社交媒体图片")
async def analyze_image(url: str, config: RunnableConfig = None) -> str:
    """使用Kimi 2.5多模态分析时尚图片。
    
    Returns structured Chinese description covering:
    - 廓形与剪裁 (silhouette and cut)
    - 色彩搭配 (color palette)
    - 核心单品 (key pieces)
    - 趋势信号 (trend signals)
    - 品牌/场合判断 (brand/occasion inference)
    """
    configurable = Configuration.from_runnable_config(config)
    model = init_chat_model(
        model=configurable.research_model,
        api_key=get_api_key_for_model(configurable.research_model, config),
        base_url=configurable.openai_compatible_base_url,
    )
    prompt = analyze_image_prompt  # from prompts.py
    response = await model.ainvoke([
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": prompt}
        ])
    ])
    return response.content
```

---

## 7. Graph Builder (`graph.py`)

```python
from langgraph.graph import StateGraph, START, END, Send
from langgraph.checkpoint.memory import MemorySaver

def build_research_graph():
    graph = StateGraph(ResearchState)

    # Nodes
    graph.add_node("clarify", clarify_node)
    graph.add_node("planner", planner_node)
    graph.add_node("outline_reviser", outline_reviser_node)
    graph.add_node("section_pipeline", section_pipeline_node)  # wraps section subgraph
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
    graph.add_conditional_edges("reviewer", route_after_review)
    graph.add_edge("reviser", "reviewer")
    graph.add_edge("trend_triangulator", "reviewer")
    graph.add_edge("final_check", END)

    return graph.compile(checkpointer=MemorySaver())

# Section subgraph
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

section_subgraph = build_section_subgraph()
```

### Conditional edge functions

```python
def route_after_clarify(state: ResearchState) -> str:
    if state["need_clarification"]:
        return END
    return "planner"

def fan_out_sections(state: ResearchState) -> list[Send]:
    return [
        Send("section_pipeline", {
            "section_id": s["id"],
            "section_title": s["title"],
            "section_description": s["description"],
            "search_queries": s["search_queries"],
            "research_goal": state["research_goal"],
            "hypotheses": state["hypotheses"],
            "budget": state["budget"],
            "language": state["language"],
        })
        for s in state["sections"]
    ]

def route_after_collection(state: ResearchState) -> str:
    refuted = sum(
        1 for h in state["hypothesis_evidence"]
        if h.get("evidence_type") == "refutes"
    )
    if refuted >= 2 and state.get("outline_revision_count", 0) < 1:
        return "outline_reviser"
    return "lead_writer"

def route_after_synthesis(state: ResearchState) -> str:
    if state["research_type"] == "trend_analysis":
        return "trend_triangulator"
    return "reviewer"

def route_after_review(state: ResearchState) -> str:
    review = state["review_result"]
    if review["verdict"] != "pass" and state.get("revision_count", 0) < 2:
        return "reviser"
    return "final_check"
```

### Section pipeline wrapper

```python
def section_pipeline_node(state: SectionState) -> dict:
    result = section_subgraph.invoke(state)
    return {
        "facts": result["section_facts"],
        "data_points": result["section_data_points"],
        "hypothesis_evidence": result["section_hypothesis_evidence"],
        "charts": result["section_charts"],
        "insights": [{"section_id": state["section_id"], "insight": i}
                     for i in result["section_insights"]],
        "contradictions": result["section_contradictions"],
        "sources": result["section_sources"],
        "open_questions": [{"section_id": state["section_id"], "question": q}
                           for q in result["missing_info"]],
    }
```

---

## 8. FastAPI API (`api.py`)

### Endpoint

```
POST /research
Content-Type: application/json
Accept: text/event-stream
```

### Request body

```json
{
  "messages": [...],
  "object_context": "https://image-url.com/photo.jpg",
  "thread_id": "uuid-string"
}
```

`messages` is the compact summary + recent turns from the host system.  
`object_context` is optional — pass `null` if no image.  
`thread_id` is used as the LangGraph checkpointer key.

### SSE event stream

Each event is `data: <json>\n\n` format.

| Event type | When fired | Payload |
|---|---|---|
| `clarification` | clarify needs more info | `{"question": "..."}` |
| `progress` | each node completes | `{"node": "planner", "status": "done"}` |
| `section_done` | each section_pipeline completes | `{"section_id": "sec_1", "title": "..."}` |
| `report` | final_check completes | `{"content": "...full markdown report..."}` |
| `error` | any unhandled exception | `{"message": "..."}` |

### Implementation sketch

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.types import StreamMode
import json

app = FastAPI()
graph = build_research_graph()

@app.post("/research")
async def research(request: ResearchRequest):
    async def event_stream():
        try:
            config = {"configurable": {"thread_id": request.thread_id}}
            input_state = {
                "messages": request.messages,
                "object_context": request.object_context,
            }
            async for event in graph.astream_events(input_state, config=config, version="v2"):
                kind = event["event"]
                if kind == "on_chain_end" and event["name"] in NODE_NAMES:
                    yield f"data: {json.dumps({'type': 'progress', 'node': event['name'], 'status': 'done'})}\n\n"
                    if event["name"] == "section_pipeline":
                        section_id = event["data"]["output"].get("section_id")
                        yield f"data: {json.dumps({'type': 'section_done', 'section_id': section_id})}\n\n"
                elif kind == "on_chain_end" and event["name"] == "final_check":
                    report = event["data"]["output"].get("full_report", "")
                    yield f"data: {json.dumps({'type': 'report', 'content': report})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

Clarification event: when `clarify_node` sets `need_clarification=True`, the graph ends. The final state is checked before streaming ends — if `need_clarification` is True, emit a `clarification` event with the question.

---

## 9. Prompts (`prompts.py`)

All prompts are Chinese strings. Each is a module-level constant named after its agent.

| Constant | Agent | Key instructions |
|---|---|---|
| `clarify_prompt` | clarify.py | 判断是否需要澄清；若不需要则提取研究目标；处理图片输入 |
| `planner_prompt` | planner.py | 分类研究类型；生成2-4个假设；设计3-6个章节及搜索词 |
| `outline_reviser_prompt` | outline_reviser.py | 最小化修改；新增/删除/重排章节；保留章节ID |
| `deep_scout_prompt` | deep_scout.py | ReAct循环；每次搜索后think；优先tier-1/2来源；图片分析 |
| `analyst_prompt` | analyst.py | 提取主题和模式；评估假设证据；保留矛盾不解决 |
| `data_wiz_prompt` | data_wiz.py | 仅提取有来源的数字；生成ECharts配置 |
| `writer_prompt` | writer.py | 专业投研语气；每个关键声明引用来源；500-1000字/节 |
| `synthesizer_prompt` | synthesizer.py | 写执行摘要；合并章节；假设结论；参考文献列表 |
| `trend_triangulator_prompt` | trend_triangulator.py | 三信号验证（秀场/社交/零售）；单信号标记为弱势趋势 |
| `reviewer_prompt` | reviewer.py | 零容忍幻觉；逻辑闭合；偏见警报；完整性检查 |
| `reviser_prompt` | reviser.py | 仅针对性修改；不发明信息；保持风格一致 |
| `final_check_prompt` | final_check.py | 验证修复；检查新问题；标记证据不足的声明 |
| `analyze_image_prompt` | utils.py (analyze_image tool) | 廓形/色彩/单品/趋势信号/品牌判断 |

---

## 10. Error Handling

| Phase | Failure | Behavior |
|---|---|---|
| `clarify` | LLM call fails | raise → FastAPI streams `error` event |
| `planner` | structured output fails | retry up to `max_structured_output_retries`, then raise |
| `section_pipeline` | one section fails | log + skip, continue with remaining sections |
| `deep_scout` | search returns empty | write `search_results: []`, analyst handles gracefully |
| `reviewer` loop | still failing after 2 revisions | `final_check` runs anyway, sets `publication_readiness: "needs_review"` |
| any node | token limit exceeded | use `is_token_limit_exceeded` + `remove_up_to_last_ai_message` from `utils.py` |
| FastAPI | unhandled exception | streams single `error` event, closes stream |

---

## 11. Fashion Source Credibility Taxonomy

Used by `deep_scout` to prioritize sources and set `credibility_score` on `SearchResult`.

| Tier | Type | Score | Examples |
|---|---|---|---|
| 1 | `industry_authority` | 0.85–1.00 | BoF, Vogue Runway, WWD, WGSN |
| 2 | `retail_data` | 0.75–0.90 | Lyst, Edited, Trendalytics, resale platforms |
| 3 | `fashion_press` | 0.65–0.80 | Vogue, Harper's Bazaar, Elle editorial |
| 4 | `brand_primary` | 0.60–0.80 | press releases, earnings calls |
| 5 | `social_signal` | 0.40–0.65 | Instagram, TikTok, street style |
| 6 | `affiliate_sponsored` | 0.15–0.35 | affiliate blogs, sponsored lists |

---

## 12. Cost Model

| Operation | Model | Calls per job |
|---|---|---|
| Web page summarization | Qwen | 10–30 |
| Credibility tagging + data extraction | Qwen | 10–30 |
| Clarify + research goal extraction | Kimi 2.5 | 1–2 |
| Planning + hypothesis generation | Kimi 2.5 | 1 |
| Search strategy (DeepScout) | Kimi 2.5 | 3–6 |
| Qualitative analysis (Analyst) | Kimi 2.5 | 3–6 |
| Image analysis | Kimi 2.5 multimodal | 0–5 |
| Section writing | Kimi 2.5 | 3–6 |
| Synthesis + review loop | Kimi 2.5 | 2–4 |

Expected split: **60–70% Qwen** (search preprocessing), **30–40% Kimi 2.5** (all reasoning/writing).

---

## 13. Implementation Order

1. `schemas.py` + `state.py` — define all types first
2. `prompts.py` — extract all prompts from existing agent classes, translate to Chinese where needed
3. `utils.py` — add `analyze_image` tool
4. `agents/clarify.py` — entry point, test with mock messages
5. `agents/planner.py` + `agents/outline_reviser.py`
6. Section subgraph: `deep_scout.py` → `analyst.py` → `data_wiz.py`
7. Writing phase: `writer.py` → `synthesizer.py` → `trend_triangulator.py`
8. Quality phase: `reviewer.py` → `reviser.py` → `final_check.py`
9. `graph.py` — wire everything together, test end-to-end
10. `api.py` — FastAPI + SSE, test with host system

---

## 14. Integration Contract with Host System

**Input (host system → this service):**
- `messages`: compact conversation history (host system manages compression)
- `object_context`: image URL or `null`
- `thread_id`: for checkpointer correlation

**Output (this service → host system):**
- SSE stream of typed events (see Section 8)
- `report` event payload is the final `full_report` markdown string
- Host system appends `full_report` to its own message history
- This service does **not** manage external sessions, memory, or message history

**Human-in-the-loop clarification flow:**
1. This service streams `clarification` event with question text
2. Host system shows question to user, receives answer
3. Host system appends Q&A to messages, re-invokes `POST /research` with updated messages + same `thread_id`
4. `clarify.py` reads updated messages, proceeds to planning
