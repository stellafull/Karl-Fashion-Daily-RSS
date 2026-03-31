# Fashion Deep Research Agent — Final Build Spec

**v5 — Frozen enough to build**

This document merges the domain design from the schema/prompt spec with the concrete LangGraph implementation plan.

**Decision:** this design is sufficient to start implementation. Do not spend more cycles redesigning unless implementation reveals a real problem.

---

## Model Strategy

- **Primary:** Kimi 2.5 — planning, reasoning, writing, review, multimodal image analysis
- **Cost-optimized:** Qwen — summarization, extraction, tagging, compression, dedup support inside tools

### Allocation principle

Use **Kimi 2.5** for:

- ambiguous intent resolution
- hypothesis generation
- search strategy decisions
- qualitative synthesis
- section writing
- review and revision
- multimodal runway / lookbook / social image analysis

Use **Qwen** for:

- web page summarization
- structured fact extraction
- data point extraction
- credibility tagging
- content compression
- repetitive high-volume preprocessing

---

## 1. System Goal

Build a fashion deep research system using **LangChain + LangGraph** that can:

- turn messy user conversation into a clean research brief
- plan provisional hypotheses and sections
- collect evidence in parallel by section
- preserve contradictions instead of flattening them
- write a coherent research report
- review and revise the output with bounded loops
- keep cost under control through a Kimi/Qwen split

The system should optimize for:

- **research quality**
- **clear evidence linkage**
- **cost discipline**
- **implementation simplicity**
- **iterability during production rollout**

---

## 2. High-Level Architecture

The pipeline has 4 phases:

1. **Scoping**
2. **Collection**
3. **Writing**
4. **Quality**

### Phase summary

| Phase      | Stages                                               | Model    | Why                                                |
| ---------- | ---------------------------------------------------- | -------- | -------------------------------------------------- |
| Scoping    | Context Resolver, Architect Planner, Outline Reviser | Kimi 2.5 | Deep reasoning and research framing                |
| Collection | DeepScout, Analyst, Data/CodeWiz                     | Mixed    | Kimi decides and analyzes, Qwen preprocesses pages |
| Writing    | Lead Writer, Synthesizer, Trend Triangulator         | Kimi 2.5 | Cross-source reasoning and narrative coherence     |
| Quality    | Reviewer, Reviser, Final Check                       | Kimi 2.5 | Critical evaluation and claim verification         |

---

## 3. Why LangGraph

LangGraph is the right orchestration layer because it gives us:

1. **Shared state** across nodes
2. **Conditional routing** for review loops and optional stages
3. **Fan-out via `Send()`** for per-section parallel collection
4. **Subgraphs** for reusable section-level workflows
5. **Checkpointing / persistence** for long-running jobs and human interruption

This means we do **not** need a custom orchestrator or a fake "lead researcher" control node.

---

## 4. Graph Topology

```text
START
  -> context_resolver
  -> architect_planner
  -> outline_reviser
  -> Send() section_pipeline per section
       -> deep_scout
       -> analyst
       -> data_wiz
  -> reducer merge into main state
  -> lead_writer
  -> synthesizer
  -> trend_triangulator (only if research_type == trend_analysis)
  -> reviewer
       -> reviser -> reviewer   (if score < 7, max 2 loops)
       -> final_check           (if pass or loop exhausted)
END
```

### Practical rules

- **Collection is parallel** by section.
- **Writing is sequential** by section to maintain coherence.
- **Review loop is bounded** to max 2 iterations.
- **Outline revision after collection is optional and rare**.
- **Full re-plan after writing is manual / exceptional**, not automatic.

---

## 5. Main Shared State

Use one shared `ResearchState` for the main graph.

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import add_messages

class ResearchState(TypedDict):
    # Conversation / entry
    messages: Annotated[list, add_messages]
    trigger_message: str
    object_context: str | None

    # Scoping
    need_clarification: bool
    clarification_question: str
    research_goal: str
    confirmed_constraints: list[str]
    open_dimensions: list[str]
    known_context: list[str]
    excluded_paths: list[str]
    deliverable_type: str
    language: str

    # Planning
    research_type: str
    hypotheses: list[dict]
    sections: list[dict]
    budget: dict
    outline_status: str
    outline_revision_count: int

    # Collection (parallel write fields use reducers)
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

### Guardrail

Do not keep too many prompt-shaped objects in long-lived state. Persist the **raw evidence and control fields**, and format prompts on demand.

---

## 6. Section Subgraph State

Each parallel section pipeline runs on its own `SectionState`.

```python
class SectionState(TypedDict):
    # Inputs from parent graph
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

    # Data output
    section_data_points: list[dict]
    section_charts: list[dict]
    section_time_series: list[dict]

    # Sources
    section_sources: list[dict]
```

---

## 7. Core Schemas

### 7.1 Research Brief

Output of `context_resolver`. Input to `architect_planner`.

```json
{
  "research_goal": "I want to understand...",
  "attached_object": null,
  "confirmed_constraints": [],
  "open_dimensions": [],
  "known_context": [],
  "excluded_paths": [],
  "deliverable_type": "deep_report",
  "language": "en"
}
```

### 7.2 Architect Plan

```json
{
  "research_type": "trend_analysis",
  "hypotheses": [],
  "sections": [],
  "budget": {
    "max_parallel": 3,
    "max_searches": 5,
    "max_deep_reads": 3
  },
  "outline_status": "provisional"
}
```

### 7.3 Hypothesis

```json
{
  "id": "h_1",
  "statement": "Quiet luxury is declining in consumer preference",
  "evidence_needed": [
    "Search volume trend",
    "Runway presence comparison",
    "Retail / resale support"
  ],
  "status": "untested"
}
```

### 7.4 Section

```json
{
  "id": "sec_1",
  "title": "Market Overview",
  "description": "Current state of the market with size, growth, and segmentation",
  "search_queries": [
    "global luxury market size 2025 2026",
    "luxury goods growth rate 2025",
    "personal luxury goods market outlook"
  ],
  "priority": 1
}
```

### 7.5 SearchResult (tool output)

This is the clean result returned by the Qwen-powered search tool.

```json
{
  "source_id": "src_001",
  "url": "https://example.com/report",
  "title": "The State of Fashion 2026",
  "source_type": "industry_authority",
  "credibility_score": 0.92,
  "summary": "Clean markdown summary...",
  "key_facts": [
    {
      "content": "Global personal luxury goods market reached €362B in 2025",
      "importance": "high",
      "data_points": [
        {
          "name": "Global luxury market size",
          "value": 362,
          "unit": "billion EUR",
          "year": 2025
        }
      ]
    }
  ],
  "data_points": [
    {
      "id": "dp_001",
      "name": "Global luxury market size",
      "value": 362,
      "unit": "billion EUR",
      "year": 2025,
      "category": "market_size",
      "confidence": 0.9
    }
  ],
  "publish_date": "2026-01-15",
  "language": "en"
}
```

### 7.6 Fact

```json
{
  "content": "Hermes reported 23% revenue growth in Q3 2025",
  "source_id": "src_003",
  "section_id": "sec_2",
  "importance": "high",
  "related_hypothesis": "h_1"
}
```

### 7.7 DataPoint

```json
{
  "id": "dp_001",
  "name": "Hermes Q3 revenue growth",
  "value": 23,
  "unit": "%",
  "year": 2025,
  "source_id": "src_003",
  "category": "growth_rate",
  "confidence": 0.95
}
```

### 7.8 Contradiction

```json
{
  "claim_a": "Quiet luxury search interest declined 30% YoY",
  "source_id_a": "src_005",
  "claim_b": "Quiet luxury remains the dominant aesthetic trend",
  "source_id_b": "src_008",
  "resolution": null
}
```

### 7.9 SectionDraft

```json
{
  "section_id": "sec_1",
  "content": "## Market Overview\n\nThe global personal luxury goods market...",
  "citations": [
    {
      "claim": "market reached €362B",
      "source_id": "src_001",
      "url": "https://example.com/report"
    }
  ],
  "charts_used": ["chart_001"],
  "weak_claims": ["The Gen Z spending claim is supported by only one source"]
}
```

### 7.10 ReviewResult

```json
{
  "quality_score": 7,
  "verdict": "pass",
  "issues": [
    {
      "id": "issue_1",
      "section_id": "sec_2",
      "type": "missing_source",
      "severity": "major",
      "description": "Revenue growth claim lacks source attribution",
      "suggestion": "Add citation to the earnings report"
    }
  ],
  "claim_checks": [
    {
      "claim_text": "market reached €362B",
      "source_id": "src_001",
      "status": "verified"
    }
  ],
  "missing_aspects": ["No coverage of sustainability regulation impact"]
}
```

---

## 8. Fashion Source Credibility Taxonomy

| Tier | Source type           | Score range | Examples                                            |
| ---- | --------------------- | ----------- | --------------------------------------------------- |
| 1    | `industry_authority`  | 0.85 - 1.00 | BoF, Vogue Runway, WWD, WGSN                        |
| 2    | `retail_data`         | 0.75 - 0.90 | Lyst, Edited, Trendalytics, resale platforms        |
| 3    | `fashion_press`       | 0.65 - 0.80 | Vogue, Harper's Bazaar, Elle editorial              |
| 4    | `brand_primary`       | 0.60 - 0.80 | press releases, earnings calls, official statements |
| 5    | `social_signal`       | 0.40 - 0.65 | Instagram, TikTok, street style                     |
| 6    | `affiliate_sponsored` | 0.15 - 0.35 | affiliate blogs, sponsored lists                    |

---

## 9. Main Graph Builder

```python
from langgraph.graph import StateGraph, START, END, Send
from langgraph.checkpoint.memory import MemorySaver


def build_research_graph():
    graph = StateGraph(ResearchState)

    # Phase 1
    graph.add_node("context_resolver", context_resolver_node)
    graph.add_node("architect_planner", architect_planner_node)
    graph.add_node("outline_reviser", outline_reviser_node)

    # Phase 2
    graph.add_node("section_pipeline", section_pipeline_node)

    # Phase 3
    graph.add_node("lead_writer", lead_writer_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("trend_triangulator", trend_triangulator_node)

    # Phase 4
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("reviser", reviser_node)
    graph.add_node("final_check", final_check_node)

    graph.add_edge(START, "context_resolver")
    graph.add_conditional_edges("context_resolver", route_after_clarify)
    graph.add_edge("architect_planner", "outline_reviser")
    graph.add_conditional_edges("outline_reviser", fan_out_sections)
    graph.add_conditional_edges("section_pipeline", route_after_collection)
    graph.add_edge("lead_writer", "synthesizer")
    graph.add_conditional_edges("synthesizer", route_after_synthesis)
    graph.add_conditional_edges("reviewer", route_after_review)
    graph.add_edge("reviser", "reviewer")
    graph.add_edge("final_check", END)

    return graph.compile(checkpointer=MemorySaver())
```

### Note

In production, replace `MemorySaver()` with a persistent checkpointer.

---

## 10. Conditional Edge Functions

```python
def route_after_clarify(state: ResearchState) -> str:
    if state["need_clarification"]:
        return END
    return "architect_planner"


def fan_out_sections(state: ResearchState) -> list[Send]:
    sends = []
    for section in state["sections"]:
        sends.append(
            Send("section_pipeline", {
                "section_id": section["id"],
                "section_title": section["title"],
                "section_description": section["description"],
                "search_queries": section["search_queries"],
                "research_goal": state["research_goal"],
                "hypotheses": state["hypotheses"],
                "budget": state["budget"],
                "language": state["language"],
            })
        )
    return sends


def route_after_collection(state: ResearchState) -> str:
    if should_revise_outline(state) == "outline_reviser":
        return "outline_reviser"
    return "lead_writer"


def route_after_synthesis(state: ResearchState) -> str:
    if state["research_type"] == "trend_analysis":
        return "trend_triangulator"
    return "reviewer"


def route_after_review(state: ResearchState) -> str:
    review = state["review_result"]
    if review["verdict"] != "pass" and state["revision_count"] < 2:
        return "reviser"
    return "final_check"
```

---

## 11. Section Subgraph

```python
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

### Section pipeline wrapper

```python
def section_pipeline_node(state: SectionState) -> dict:
    result = section_subgraph.invoke(state)
    return {
        "facts": result["section_facts"],
        "data_points": result["section_data_points"],
        "hypothesis_evidence": result["section_hypothesis_evidence"],
        "charts": result["section_charts"],
        "insights": [
            {"section_id": state["section_id"], "insight": i}
            for i in result["section_insights"]
        ],
        "contradictions": result["section_contradictions"],
        "sources": result["section_sources"],
        "open_questions": [
            {"section_id": state["section_id"], "question": q}
            for q in result["missing_info"]
        ],
    }
```

---

## 12. Node Prompts

### 12.1 Context Resolver

**Model:** Kimi 2.5

```text
You are a Context Resolver for a fashion research system. Your job is to translate a conversation into a clean research specification.

Today's date is {date}.

## Conversation context
Running summary: {conversation_summary}
Recent turns: {recent_turns}
Current object: {object_context}
Trigger message: {trigger_message}

## Your task
1. Identify the user's active research intent
2. Discard stale conversation branches
3. Preserve settled constraints
4. Identify any critical ambiguity that MUST be resolved before research can begin
5. If no critical ambiguity, emit a research specification

## Rules
- Most of the time, DO NOT ask a clarifying question.
- Only ask if ambiguity would lead to a fundamentally wrong research direction.
- If a question was already asked and answered in history, never ask again.
- For unspecified dimensions, mark them as open_dimensions.
- Phrase research_goal in first person from the user's perspective.
- Detect the user's language.

## Output JSON
{
  "need_clarification": false,
  "clarification_question": "",
  "research_goal": "I want to understand...",
  "confirmed_constraints": [],
  "open_dimensions": [],
  "known_context": [],
  "excluded_paths": [],
  "deliverable_type": "deep_report",
  "language": "en"
}
```

### 12.2 Architect Planner

**Model:** Kimi 2.5

```text
You are the Architect Planner for a fashion deep research system.

Today's date: {date}

## Research specification
Goal: {research_goal}
Constraints: {confirmed_constraints}
Open dimensions: {open_dimensions}
Known context: {known_context}
Excluded: {excluded_paths}
Deliverable type: {deliverable_type}
Language: {language}

## Your task
1. Classify the research_type
2. Generate 2-4 PROVISIONAL hypotheses
3. Design 3-6 sections that cover the question well
4. For each section, provide 2-4 specific search queries
5. Set execution budget based on complexity

## Fashion-specific guidance
- Anchor hypotheses to fashion calendar when relevant
- Include English plus relevant local-language queries
- For trend work: runway, social, retail
- For brand work: earnings, positioning, consumer perception
- Prioritize BoF, WWD, Vogue Runway, Lyst, Edited over generic news

## Budget guidelines
- Simple: max_parallel=2, max_searches=3, max_deep_reads=2
- Medium: max_parallel=3, max_searches=5, max_deep_reads=3
- Complex: max_parallel=4, max_searches=7, max_deep_reads=4

## Critical
Mark outline_status as "provisional".
Collection is expected to challenge the framing.
```

### 12.3 Outline Reviser

**Model:** Kimi 2.5

```text
You are the Outline Reviser.

Research goal: {research_goal}
Current plan: {plan}
Collection signals so far: {collection_summary}

## Your task
- tighten the outline if needed
- add a missing section only if evidence shows a meaningful uncovered angle
- remove or downgrade sections that are weak or redundant
- keep changes minimal

## Rules
- this is not a full re-plan unless absolutely necessary
- max 1 outline revision per task
- preserve section IDs when possible to reduce downstream churn
```

### 12.4 Smart Search Tool (internal)

**Model:** Qwen

```text
You are a web content processor for a fashion research system. Summarize this page and extract structured data.

Research context: {research_goal}
Current section: {section_title} — {section_description}
Page URL: {url}
Page content:
{raw_content}

## Tasks
1. Classify source_type using the fashion taxonomy
2. Assign credibility_score
3. Write a concise summary preserving key claims and numbers
4. Extract key_facts with importance and embedded data_points
5. Extract standalone data_points
6. Detect publish_date if present

## Rules
- Never invent data
- Tag sponsored / affiliate content explicitly
- Preserve contradictions
- Prefer extracting only clearly attributable claims
```

### 12.5 DeepScout

**Model:** Kimi 2.5 with tool calling

```text
You are DeepScout, a research agent for fashion industry research. You gather information for one section of a larger report.

Today's date: {date}
Overall research goal: {research_goal}
Your section: {section_title} — {section_description}
Initial search queries: {search_queries}
Hypotheses to test: {hypotheses}
Budget: max {max_searches} searches, max {max_deep_reads} deep reads

## Available tools
- search(query): returns clean SearchResult[]
- deep_read(url): full page fetch + analysis
- analyze_image(url): multimodal analysis for runway / lookbook / social imagery
- think(): reflect on findings and next steps

## Strategy
1. Start with provided queries
2. After each search, call think()
3. Deep-read tier 1-2 sources
4. Use image analysis for visual trend topics
5. Seek evidence that supports and refutes hypotheses
6. Stop when the answer is good enough, budget is hit, or latest searches are repetitive

## Critical rules
- Preserve contradictions
- Flag PR-campaign echo chambers
- Do not ignore evidence that refutes the working hypothesis
- Output structured SectionState-compatible results
```

### 12.6 Analyst

**Model:** Kimi 2.5

```text
You are a fashion research Analyst. Analyze search results for one section and extract qualitative insights.

Research goal: {research_goal}
Section: {section_title} — {section_description}
Hypotheses: {hypotheses}
Search results: {search_results}

## Tasks
1. Identify narrative themes and patterns
2. Evaluate hypothesis evidence: supports / refutes / inconclusive
3. Extract strategic insights beyond any single source
4. Record contradictions without resolving them
5. Identify entities and relationships

## Output JSON
{
  "section_facts": [],
  "section_insights": [],
  "section_hypothesis_evidence": [],
  "section_contradictions": [],
  "section_entities": [],
  "missing_info": []
}
```

### 12.7 Data/CodeWiz

**Model:** Kimi 2.5

```text
You are a data analyst for fashion industry research. Extract quantitative data and generate chart configurations.

Research topic: {research_goal}
Section: {section_title}
Search results with data: {search_results}

## Tasks
1. Extract quantifiable data points
2. Identify time series
3. Identify distributions and breakdowns
4. Generate ECharts configs for the most useful views
5. Only use clearly sourced numbers

## Output JSON
{
  "section_data_points": [],
  "section_charts": [],
  "section_time_series": []
}
```

### 12.8 Lead Writer

**Model:** Kimi 2.5

```text
You are the Lead Writer for a fashion deep research report. Write one section at a time.

Research goal: {research_goal}
Full outline: {sections_list}
Hypothesis results: {hypothesis_evidence}
Previously written sections: {previous_sections}

## Current section
Title: {section_title}
Description: {section_description}
Section facts: {section_facts}
Section data points: {section_data_points}
Available charts: {charts}
Cross-section facts: {cross_section_facts}
Known contradictions: {contradictions}

## Writing rules
- Professional investment research tone
- Every key claim should cite a source
- Use data to support arguments, not decorate them
- Present both sides of contradictions when relevant
- Flag thin evidence in weak_claims
- Avoid repetition with earlier sections
- Write in {language}
- Aim for 500-1000 words per section

## Output JSON
{
  "section_id": "{section_id}",
  "content": "## Section Title\n\n...",
  "citations": [],
  "charts_used": [],
  "weak_claims": []
}
```

### 12.9 Synthesizer

**Model:** Kimi 2.5

```text
You are the Synthesizer. Merge all section drafts into a complete research report.

Research goal: {research_goal}
All section drafts: {section_drafts}
Hypothesis results: {hypothesis_evidence}
All contradictions: {contradictions}
All sources: {sources}

## Tasks
1. Write an executive summary
2. Merge sections into one coherent report
3. Write conclusions with hypothesis verdicts
4. Add an unresolved questions section if contradictions remain
5. Compile a numbered reference list with clickable links

## Rules
- Do not invent information not present in the drafts / evidence
- Remove redundancy
- Preserve nuance and uncertainty markers
- Write in {language}
```

### 12.10 Trend Triangulator

**Model:** Kimi 2.5
**Only runs when:** `research_type == "trend_analysis"`

```text
You are the Trend Triangulator for fashion research. Validate trend claims using cross-signal analysis.

Report: {full_report}
Facts: {facts}
Sources: {sources}

## Validation method
For each trend claim, check whether it has support from at least 2 of 3 signal types:
1. Designer / runway signal
2. Street / social adoption
3. Commercial / retail data

Claims with only one signal should be marked weak or emerging.

## Output JSON
{
  "trend_validations": [],
  "weak_trends": [],
  "revised_report": "..."
}
```

### 12.11 Reviewer

**Model:** Kimi 2.5

```text
You are an extremely strict research reviewer and fact-checker for fashion industry reports.

Research goal: {research_goal}
Report outline: {sections}
Report content: {full_report}
Available facts: {facts}
Available data points: {data_points}

## Review criteria
1. Zero tolerance for hallucination
2. Logic closure
3. Bias alert
4. Timeliness
5. Completeness
6. Claim verification

## Scoring
- 9-10: publish-ready
- 7-8: pass with minor issues
- 5-6: needs revision
- 1-4: major problems

quality_score >= 7 is required for pass.
```

### 12.12 Reviser

**Model:** Kimi 2.5

```text
You are the Reviser. Fix the issues identified by the Reviewer.

Original report: {full_report}
Reviewer feedback: {review_result}

## Principles
1. Targeted edits only
2. Add support where evidence exists
3. Correct factual / logical issues
4. Maintain style consistency

## Output JSON
{
  "full_report": "...revised report...",
  "changes_made": [],
  "addressed_issues": [],
  "unable_to_address": []
}
```

### 12.13 Final Check

**Model:** Kimi 2.5

```text
You are the final quality gatekeeper.

Research goal: {research_goal}
Previous issues: {review_result}
Current report: {full_report}
Revision count: {revision_count}

## Task
1. Verify previous issues were resolved
2. Check for new issues introduced during revision
3. Mark claims with insufficient evidence when needed

## Output JSON
{
  "resolved_issues": [],
  "unresolved_issues": [],
  "new_issues": [],
  "final_score": 8,
  "final_verdict": "approved",
  "publication_readiness": "ready",
  "final_comments": "..."
}
```

---

## 13. Tool Definitions

### 13.1 search



### 13.3 analyze_image

```python
@tool
def analyze_image(url: str) -> dict:
    """Analyze a runway, lookbook, or social image using Kimi 2.5 multimodal."""
    return kimi_vision_analyze(url, fashion_analysis_prompt)
```

---

## 14. Feedback Loops

### 14.1 Collection -> Outline Reviser

Trigger only when evidence strongly suggests the initial framing is incomplete or wrong.

Recommended trigger:

- 2+ strong refutations of top hypotheses, or
- a high-priority uncovered angle appears across multiple credible sources

```python
def should_revise_outline(state: ResearchState) -> str:
    refuted_count = sum(
        1 for h in state["hypothesis_evidence"]
        if h.get("evidence_type") == "refutes"
    )
    if refuted_count >= 2 and state["outline_status"] == "provisional" and state["outline_revision_count"] < 1:
        return "outline_reviser"
    return "lead_writer"
```

### 14.2 Reviewer -> Reviser

- Trigger: `quality_score < 7`
- Limit: max 2 revision loops

### 14.3 Lead Writer -> Full Re-plan

Not automatic.
Only surface this as a manual escalation when the overall framing is clearly broken.

---

## 15. Cost Optimization

| Operation              | Model               | Calls per task | Cost tier |
| ---------------------- | ------------------- | -------------- | --------- |
| Web page summarization | Qwen                | 10-30          | Low       |
| Credibility tagging    | Qwen                | 10-30          | Low       |
| Data extraction        | Qwen                | 10-30          | Low       |
| Image analysis         | Kimi 2.5 multimodal | 0-5            | High      |
| Hypothesis generation  | Kimi 2.5            | 1              | High      |
| Search strategy        | Kimi 2.5            | 3-6            | High      |
| Qualitative analysis   | Kimi 2.5            | 3-6            | High      |
| Section writing        | Kimi 2.5            | 3-6            | High      |
| Report synthesis       | Kimi 2.5            | 1              | High      |
| Quality review         | Kimi 2.5            | 1-3            | High      |

### Expected split

- **60-70%** of LLM calls -> Qwen
- **30-40%** of LLM calls -> Kimi 2.5

The biggest cost lever is keeping Qwen inside the search / deep-read preprocessing path.

---

## 16. Implementation Guardrails

These are the small rules that should keep the system from drifting into an expensive or brittle architecture.

1. **Do not over-persist prompt-shaped objects**
   - keep raw evidence in state
   - build prompt context at call time

2. **Use reducers only on true parallel-write fields**
   - facts, sources, contradictions, insights, charts, data_points

3. **Keep the writer's context selective**
   - current section packet
   - short cross-section memo
   - not the entire evidence universe every time

4. **Use stable IDs everywhere**
   - `source_id`, `hypothesis_id`, `section_id`, `issue_id`, `chart_id`

5. **Preserve contradiction objects**
   - do not resolve them too early

6. **Bound feedback loops tightly**
   - outline revision max 1
   - reviewer loop max 2

7. **Persist checkpoints in production**
   - do not rely on in-memory saver outside local development

8. **Treat this spec as a baseline, not a prison**
   - adjust prompts and packing during implementation
   - do not redesign the whole architecture without concrete evidence

---

## 17. Build Recommendation

Start implementation in this order:

1. Define state schemas and IDs
2. Implement tool layer (`search` `analyze_image`)
3. Build section subgraph
4. Build main graph with reducers and routing
5. Add writer / synthesizer / reviewer loop
6. Add trend triangulator
7. Add checkpointing, telemetry, and eval harness

---

## 18. Final Position

This design is **enough to build now**.

The next improvements should come from:

- real traces
- token usage
- failure cases
- bad outputs
- review-loop behavior
- evidence packing problems

Not from more abstract architecture debate.
