# Deep Research Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the current Deep Research pipeline bugs so the LangGraph server run has correct model configuration, section-scoped evidence, usable citations, failfast behavior, and accurate progress streaming.

**Architecture:** Keep the current graph topology, but tighten the node contracts. The fixes are grouped into three sequential tasks: model/configuration correctness, evidence/source correctness, and runtime/streaming correctness. Each task adds or adjusts regression tests first, then makes the minimal code changes required to satisfy the tested behavior.

**Tech Stack:** Python 3.12, LangChain, LangGraph, LangSmith, FastAPI, pytest, uv

---

### Task 1: Configuration And Structured Output Contracts

**Files:**
- Modify: `src/deep_agents/configuration.py`
- Modify: `src/deep_agents/agents/clarify.py`
- Modify: `src/deep_agents/agents/planner.py`
- Modify: `src/deep_agents/agents/outline_reviser.py`
- Modify: `src/deep_agents/agents/analyst.py`
- Modify: `src/deep_agents/agents/data_wiz.py`
- Modify: `src/deep_agents/agents/writer.py`
- Modify: `src/deep_agents/agents/reviewer.py`
- Modify: `src/deep_agents/agents/reviser.py`
- Modify: `src/deep_agents/agents/final_check.py`
- Test: `tests/test_configuration.py`
- Test: `tests/agents/test_planner.py`

- [ ] Add or adjust tests covering the expected provider base URL default and rejection of request-endpoint URLs.
- [ ] Run the configuration tests alone and confirm they fail for the current implementation.
- [ ] Add or adjust tests covering structured-output nodes disabling streaming where they call `with_structured_output(...)`.
- [ ] Run the planner test alone and confirm it fails for the current implementation.
- [ ] Update configuration defaults and validation so `openai_compatible_base_url` is a provider base URL, not a completions endpoint.
- [ ] Update each structured-output node to initialize its model with streaming disabled.
- [ ] Re-run the targeted configuration and planner tests until they pass.

### Task 2: Evidence, Source, And Section-Scoping Contracts

**Files:**
- Modify: `src/deep_agents/utils.py`
- Modify: `src/deep_agents/agents/deep_scout.py`
- Modify: `src/deep_agents/graph.py`
- Modify: `src/deep_agents/agents/writer.py`
- Modify: `src/deep_agents/prompts.py`
- Modify: `tests/agents/test_section_pipeline.py`
- Modify: `tests/agents/test_writing.py`
- Add or modify: `tests/agents/test_deep_scout.py` or `tests/agents/test_section_pipeline.py`

- [ ] Add or adjust tests that prove search results produce structured source records, globally unique source IDs, and section-tagged evidence after section merge.
- [ ] Add or adjust tests that prove writer input filtering does not leak facts or data points from one section into another.
- [ ] Add or adjust tests that prove a second section-pipeline pass replaces merged evidence instead of appending stale data when override semantics are required.
- [ ] Run the new targeted tests and confirm they fail against the current implementation.
- [ ] Change the search-result flow so downstream code receives machine-readable source metadata instead of only a formatted string.
- [ ] Ensure section-level evidence and sources carry `section_id` before they are merged into `ResearchState`.
- [ ] Ensure merged source IDs are globally unique across sections.
- [ ] Ensure section-pipeline merged collection fields use explicit override updates where re-collection should replace prior evidence.
- [ ] Re-run the targeted evidence and writer tests until they pass.

### Task 3: Runtime Failfast, Streaming, And MCP Store Guards

**Files:**
- Modify: `src/deep_agents/agents/deep_scout.py`
- Modify: `src/deep_agents/agents/writer.py`
- Modify: `src/deep_agents/api.py`
- Modify: `src/deep_agents/utils.py`
- Modify: `tests/test_api.py`
- Modify: `tests/agents/test_section_pipeline.py`

- [ ] Add or adjust tests that prove `writer_node` emits `section_done` custom stream events during real node execution.
- [ ] Add or adjust tests that prove authenticated MCP token loading safely handles missing runtime store instead of crashing with `AttributeError`.
- [ ] Add or adjust tests that prove section failures surface as hard failures where the design requires failfast behavior, rather than silently degrading into empty evidence.
- [ ] Run the new targeted tests and confirm they fail against the current implementation.
- [ ] Implement real custom stream writes for completed sections.
- [ ] Replace silent failure swallowing in section collection paths with explicit failures or minimal guarded behavior aligned with failfast semantics.
- [ ] Guard MCP token helpers against `get_store()` returning `None`.
- [ ] Re-run the targeted runtime tests until they pass.

### Final Verification

**Files:**
- Verify only: `src/deep_agents/**/*.py`
- Verify only: `tests/**/*.py`

- [ ] Run `uv run pytest -q` from the repository root and confirm the suite is green.
- [ ] Run `uv run python -m compileall src` and confirm the source tree compiles cleanly.
- [ ] Re-check the implemented behavior against the original bug list: configuration contract, citation/source contract, section scoping, override semantics, section progress streaming, failfast behavior, and MCP store safety.
