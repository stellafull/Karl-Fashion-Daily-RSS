# Command + asyncio.gather Refactor Design

**Date:** 2026-04-01  
**Branch:** feat/deep-research-task1-4  
**Status:** Approved, pending implementation

---

## Background

The current graph uses LangGraph `Send` to fan out section research in parallel, with five `route_after_*` functions as conditional edges. This design has two problems:

1. **Parallel-write collision:** `Send` calls `section_pipeline_node` once per section concurrently. Any field in `SectionState` that isn't annotated with `operator.add` in `ResearchState` causes `InvalidUpdateError` when multiple parallel nodes write to it simultaneously.

2. **Routing scattered:** Control flow logic lives in standalone functions (`route_after_clarify`, `route_after_collection`, etc.) disconnected from the nodes that have the context to make those decisions.

LangChain's `open_deep_research` solves both: routing moves into nodes via `Command`, and parallel fan-out uses `asyncio.gather` inside a single node — one atomic write to parent state.

---

## Design

### 1. Command-Based Routing

Nodes that make routing decisions return `Command(goto=next_node, update={...})` instead of a plain dict. This co-locates routing logic with the node that has the information.

**Affected nodes:** `clarify`, `synthesizer`, `reviewer`, `section_pipeline`

**Before:**
```python
# graph.py
def route_after_clarify(state: ResearchState) -> str:
    if state.get("need_clarification"):
        return END
    return "planner"

graph.add_conditional_edges("clarify", route_after_clarify)

# clarify.py
async def clarify_node(state, config) -> dict:
    ...
    return {"need_clarification": True, "clarification_question": q}
```

**After:**
```python
# clarify.py
async def clarify_node(state, config) -> Command:
    ...
    if need_clarification:
        return Command(goto=END, update={"need_clarification": True, "clarification_question": q})
    return Command(goto="planner", update={"research_goal": ..., "language": ...})

# graph.py — no routing function, plain edge not needed (Command handles it)
```

**Routing functions removed:** `route_after_clarify`, `route_after_collection`, `route_after_synthesis`, `route_after_review`, `fan_out_sections`

---

### 2. Explicit Researcher Loop (deep_scout)

Replace `create_react_agent` with an explicit `researcher → researcher_tools` Command loop inside `deep_scout_node`. This matches LangChain's pattern and avoids the deprecated `create_react_agent` import.

**Loop structure:**

```
deep_scout_node
  └─ internal loop:
       researcher (LLM with tools bound) → calls tools or ResearchComplete
       researcher_tools (tool executor)
       exit when: ResearchComplete called | max_react_tool_calls reached | no tool calls
```

**Exit tool:**
```python
class ResearchComplete(BaseModel):
    """Signal that research for this section is complete."""
    summary: str
```

The researcher binds search tools + `ResearchComplete`. When it calls `ResearchComplete`, the loop exits and `deep_scout_node` returns its collected search results.

---

### 3. `section_pipeline_node` with asyncio.gather

Replace `Send`-based fan-out with `asyncio.gather` inside the existing `section_pipeline_node`. The node is called once, gathers all sections concurrently, merges results atomically, then routes via `Command`.

```python
async def section_pipeline_node(state: ResearchState, config: RunnableConfig) -> Command:
    section_inputs = [build_section_input(s, state) for s in state["sections"]]
    results = await asyncio.gather(*[section_subgraph.ainvoke(inp, config) for inp in section_inputs])

    merged = {key: [] for key in ["facts", "data_points", "hypothesis_evidence",
                                   "charts", "insights", "contradictions", "sources", "open_questions"]}
    for r in results:
        for key in merged:
            merged[key].extend(r.get(key, []))

    refuted = sum(1 for h in merged["hypothesis_evidence"] if h.get("evidence_type") == "refutes")
    next_node = "outline_reviser" if refuted >= 2 and state.get("outline_revision_count", 0) < 1 else "lead_writer"

    return Command(goto=next_node, update=merged)
```

**Why this eliminates the collision:** All section results are merged in-process before any write to `ResearchState`. LangGraph sees one `Command.update` from one node — no concurrent writes.

**`ResearchState` reducers:** All parallel list fields keep `Annotated[list, operator.add]` reducers — consistent with LangChain's pattern, safe if collection strategy changes.

**Graph topology:** Unchanged. `section_pipeline` is still a single node. `Send` and `fan_out_sections` are removed.

---

### 4. State & Schema Changes

**`ResearchState`** — minimal:
- All parallel list fields keep `Annotated[list, operator.add]` (no change)
- Ensure `revision_count: int` and `outline_revision_count: int` are present (Command nodes increment these in `update`)

**`SectionState`** — unchanged.

**`schemas.py`** — add `ResearchComplete` tool schema (see Section 2).

---

### 5. File Map

**Changed (6 files):**

| File | Change |
|------|--------|
| `graph.py` | Remove 5 routing functions. Replace `add_conditional_edges` with plain edges where routing moves into nodes. `section_pipeline_node` becomes gather-based. |
| `agents/clarify.py` | Return `Command` with goto and update |
| `agents/synthesizer.py` | Return `Command` routing to `trend_triangulator` or `reviewer` |
| `agents/reviewer.py` | Return `Command` routing to `reviser` or `final_check` |
| `agents/deep_scout.py` | Replace `create_react_agent` with explicit researcher loop |
| `schemas.py` | Add `ResearchComplete` |

**Unchanged:** `planner.py`, `outline_reviser.py`, `analyst.py`, `data_wiz.py`, `writer.py`, `reviser.py`, `final_check.py`, `trend_triangulator.py`, `api.py`, `prompts.py`, `configuration.py`, `utils.py`, `state.py`

**No new files. No deleted files.**

---

## What This Is Not

- No supervisor LLM / `ConductResearch` tool (we keep planner-structured sections)
- No removal of `SectionState` or the section subgraph
- No changes to prompts, configuration, or API layer
