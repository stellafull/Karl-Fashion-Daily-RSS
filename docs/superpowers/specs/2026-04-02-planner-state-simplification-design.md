# Planner State Simplification Design

## Goal

Reduce planner schema complexity and remove planner-owned execution budgeting, while keeping downstream behavior stable and minimizing code changes.

## Agreed Decisions

- The planner LLM output should contain content only.
- `budget` should be removed from graph state.
- Runtime-generated IDs should remain.
- Downstream nodes should keep consuming the existing internal normalized shape as much as possible.

## Current Problem

The planner currently asks the model to emit a detailed `ArchitectPlan` object containing:

- planner content
- runtime defaults
- enum-like control fields
- execution budgeting

This makes the model responsible for values that are not part of the research content itself, which has already caused repeated schema failures such as:

- hypothesis status values like `pending_validation`
- section priorities like `high` / `medium`
- outline status values like `draft` / `ready`

These outputs are semantically close but structurally invalid.

## Target Design

### 1. Simplified Planner LLM Contract

The planner model should output only:

- `research_type`
- `hypotheses`: list of strings
- `sections`: list of objects with:
  - `title`
  - `description`
  - `search_queries`

The planner prompt must explicitly stop requesting:

- `evidence_needed`
- `status`
- `priority`
- `outline_status`
- `budget`
- any ID fields

The planner model should no longer output:

- hypothesis IDs
- hypothesis status
- hypothesis evidence-needed lists
- section IDs
- section priority
- outline status variations
- budget

### 2. Local Normalization In `planner_node`

`planner_node` should convert the simplified model output into the existing runtime shape expected by downstream code.

It should synthesize:

- hypothesis IDs: `h_1`, `h_2`, ...
- hypothesis status: `"untested"`
- hypothesis `evidence_needed`: `[]`
- section IDs: `sec_1`, `sec_2`, ...
- section priority: `1..n`
- outline status: `"provisional"`
- outline revision count: `0`

This keeps the runtime interface stable without requiring the model to generate operational metadata.

This synthesis is deterministic construction of runtime fields, not a compatibility layer for malformed planner output.
`planner_node` should not coerce invalid legacy-style values such as:

- `pending_validation`
- `high` / `medium` / `low`
- `draft` / `ready`

Those fields should no longer exist in the simplified planner contract. If they still appear after the simplification, the run should fail fast and the planner contract should be revisited rather than patched around.

Any existing tests that assert coercion of malformed planner values must be deleted rather than updated to pass.

### 3. Remove `budget` From State

`budget` should be removed from:

- planner output schema
- `ResearchState.budget`
- `SectionState.budget`
- graph fan-out payloads
- tests and fixtures that currently include planner budget

After this change, `section_pipeline_node` must stop copying `state["budget"]` into section inputs.

### 4. Execution Limits Stay In Configuration

Execution limits belong to runtime policy, not planner content.

For this refactor:

- Do not add `max_searches` to `Configuration`
- Remove the `max_searches` hint from `deep_scout_prompt`
- `deep_scout` should no longer depend on planner state for any execution-budget field
- `deep_scout_node` must stop reading `state["budget"]` directly
- `deep_scout` execution remains bounded by configuration-owned ceilings, specifically:
  - `max_researcher_iterations` as the turn cap
  - `max_react_tool_calls` as the tool-call cap

## Why Runtime IDs Stay

The LLM does not need to emit IDs, but runtime-generated IDs remain useful because current downstream logic still uses them for:

- section fan-out / merge association
- section-scoped evidence aggregation
- writer-side section filtering
- hypothesis-evidence linkage

Removing runtime IDs would expand scope and force broader downstream rewrites. That is explicitly out of scope for this change.

## Impacted Files

Primary expected changes:

- `src/deep_agents/agents/planner.py`
- `src/deep_agents/agents/deep_scout.py`
- `src/deep_agents/prompts.py`
- `src/deep_agents/schemas.py`
- `src/deep_agents/state.py`
- `src/deep_agents/graph.py`
- related planner / graph / fixture tests

## Non-Goals

- No rewrite of downstream analyst/writer/reviewer flow
- No removal of runtime-generated IDs
- No compatibility layer for multiple planner output formats
- No redesign of the wider research graph

## Verification

The change is complete when:

- planner tests pass with the simplified output contract
- graph/state tests pass after removing `budget`
- a live planner smoke run succeeds without the previous schema failures
- raw planner LLM output in LangSmith shows only the simplified planner contract, without `id`, `status`, `priority`, `budget`, `outline_status`, or `evidence_needed`
- state after `planner_node` contains the normalized runtime shape with generated IDs, `status="untested"`, `evidence_needed=[]`, `outline_status="provisional"`, and `outline_revision_count=0`

## Validation Scope

- `research_type` remains prompt-constrained in this refactor rather than being tightened further at the schema layer
- the goal is to remove fragile planner-output metadata, not to add new schema strictness elsewhere
