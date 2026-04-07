# Pipeline Cleanup and Bug Fixes

## Goal

Fix cascading bugs in the deep research pipeline caused by dead code, schema mismatches, broken data transport, and overly strict validation. Simplify the data pipeline to match actual usage.

## Root Causes Identified

1. `tavily_search` returns formatted markdown, but `_extract_results` expects JSON — `section_sources` is always empty, causing URL validation to always crash
2. Custom `override_reducer` is a pre-`Overwrite` workaround from outdated LangGraph examples
3. Multiple schemas fight what LLMs naturally produce (`key_excerpts: str` vs list output)
4. Dead code and unused fields waste prompt tokens and obscure debugging
5. Hypothesis dicts carry unused `id/status/evidence_needed` metadata noise
6. `list[dict]` schemas provide no validation, letting bad LLM output propagate silently
7. `deep_scout` loop exits immediately if model doesn't call tools on first turn

## Changes

### Change 1: Remove Dead Code

Delete from `state.py`:
- `ResearchPhase` enum — replaced by LangGraph routing per PRD
- `AgentLog` TypedDict — never populated or read
- `AgentState` class — never used by any node or graph

Remove from `ResearchState`:
- `open_questions` field — populated by section_pipeline but never consumed downstream
- `insights` field — same

Remove from `SectionState`:
- `section_entities` — analyst produces it but section_pipeline never merges into ResearchState
- `section_time_series` — data_wiz produces it but section_pipeline never merges into ResearchState

Remove from `schemas.py`:
- `section_entities` from `AnalystOutput`
- `section_time_series` from `DataWizOutput`

Remove from agent nodes:
- `analyst.py`: stop returning `section_entities`
- `data_wiz.py`: stop returning `section_time_series`

Remove from `graph.py`:
- Merge logic for `insights` and `open_questions` in `section_pipeline_node`

Remove from `prompts.py`:
- `section_entities` from `analyst_prompt` output spec
- `section_time_series` from `data_wiz_prompt` output spec

### Change 2: Replace `override_reducer` with LangGraph `Overwrite`

In `state.py`:
- Delete `override_reducer` function
- Change all `Annotated[list[dict], override_reducer]` to `Annotated[list[dict], operator.add]`
- Affected fields: `facts`, `data_points`, `hypothesis_evidence`, `charts`, `contradictions`, `sources`, `section_drafts`

In `graph.py`:
- Import `Overwrite` from `langgraph.types`
- Replace `{"type": "override", "value": merged["facts"]}` with `Overwrite(merged["facts"])` for all merged fields in `section_pipeline_node`

### Change 3a: Fix `summarize_webpage_prompt` for `key_excerpts`

Keep `key_excerpts: str` in `Summary` schema. Fix the prompt wording:
- Change "逐条列出" to "合并为一段文字" in `summarize_webpage_prompt`

### Change 3b: Simplify Hypotheses to `list[str]`

In `state.py`:
- Change `ResearchState.hypotheses` type from `list[dict]` to `list[str]`
- Change `SectionState.hypotheses` type from `list[dict]` to `list[str]`

In `planner.py`:
- Return `plan.hypotheses` directly instead of wrapping in dicts with `id/status/evidence_needed`
- Remove the dict comprehension that creates `{"id": "h_1", "statement": h, ...}`

Downstream impact: none. No code accesses hypothesis `id`, `statement`, `status`, or `evidence_needed`. Evidence linkage uses `hypothesis_statement` text matching, not IDs.

### Change 3c: Type `ReviewResult` and `FinalResult` Loose Dicts

Add to `schemas.py`:

```python
class ReviewIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    severity: str
    description: str
    suggestion: str = ""

class ClaimCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_text: str
    source_url: str
    status: str

class ResolutionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    status: str = ""
```

Update `ReviewResult`:
- `issues: list[ReviewIssue]`
- `claim_checks: list[ClaimCheck]`

Update `FinalResult`:
- `resolved_issues: list[ResolutionItem]`
- `unresolved_issues: list[ResolutionItem]`
- `new_issues: list[ResolutionItem]`

Remove `_validate_claim_checks_use_source_url` from `reviewer.py` — `ClaimCheck` model handles validation.

Prompt changes:
- `reviewer_prompt`: remove `id` from issues field spec
- `final_check_prompt`: add `description/status` field spec for resolution items

### Change 4: Fix `tavily_search` Output and Remove `Source` Schema

Keep `Summary` schema as-is (`summary: str, key_excerpts: str`) — the LLM producing Summary via `with_structured_output(Summary)` doesn't know the URL/title. Instead, `tavily_search` constructs the structured JSON output by combining LLM summary with url/title from tavily search result metadata.

Change `tavily_search` to return structured JSON instead of formatted markdown:
```python
results_list = [
    {"url": url, "title": result["title"], "summary": result["content"] if summary is None else summary}
    for url, result, summary in zip(unique_results.keys(), unique_results.values(), summaries)
]
return json.dumps(results_list, ensure_ascii=False)
```

Remove `Source` schema from `schemas.py` — unused as a Pydantic model.

`_extract_results` in `deep_scout.py` already handles JSON lists with url/title/summary fields — structured JSON from tavily_search is now parseable, so source extraction works correctly. `section_sources` gets properly populated.

Remove from `graph.py` `section_pipeline_node`:
- `legacy_keys` set and stripping logic — no more legacy IDs exist

### Change 5: Section Pipeline Validation — Crash to Warn+Skip

In `_map_source_refs` in `graph.py`:
- Return `None` instead of raising `ValueError` on unresolved URLs
- Filter `None` results in merge loops

Keep crash for truly broken state (no sections, subgraph returns None).

### Change 6: Fix `deep_scout` Loop Robustness

In `deep_scout.py`:
- If model produces no tool calls and `search_results` is empty, append a `HumanMessage` re-prompting the model to use search tools
- Keep `max_react_tool_calls` as the total iteration cap

### Change 7: Prompt Alignment

All prompt changes to match schema changes:
- `analyst_prompt`: remove `section_entities` from output spec
- `data_wiz_prompt`: remove `section_time_series` from output spec
- `reviewer_prompt`: remove `id` from issues spec
- `final_check_prompt`: add `description/status` spec for resolution items
- `summarize_webpage_prompt`: change `key_excerpts` wording

## Files Impacted

| File | Changes |
|------|---------|
| `state.py` | Remove dead code, remove override_reducer, simplify hypotheses type, remove unused fields |
| `schemas.py` | Remove Source, remove dead fields from AnalystOutput/DataWizOutput, add ReviewIssue/ClaimCheck/ResolutionItem, update Summary with url/title |
| `graph.py` | Use Overwrite, simplify merge logic, remove legacy_keys, warn+skip validation |
| `utils.py` | Update tavily_search to return structured JSON, update Summary usage |
| `prompts.py` | Align all prompt output specs with schema changes |
| `agents/planner.py` | Simplify hypothesis return |
| `agents/analyst.py` | Remove section_entities from return |
| `agents/data_wiz.py` | Remove section_time_series from return |
| `agents/deep_scout.py` | Simplify _extract_results, fix loop robustness |
| `agents/reviewer.py` | Remove _validate_claim_checks_use_source_url |
| `tests/` | Update all affected tests |

## Non-Goals

- No redesign of the overall graph topology
- No provider/model changes
- No new features
- No changes to api.py streaming logic

## Execution Order

1. Change 1 (dead code removal) — no dependencies
2. Change 2 (Overwrite) — depends on Change 1 removing insights/open_questions
3. Change 3a (key_excerpts prompt) — independent
4. Change 3b (hypotheses simplification) — independent
5. Change 4 (tavily_search + Source removal) — independent, but should be done before Change 5
6. Change 5 (warn+skip validation) — depends on Change 4 for full effect
7. Change 6 (deep_scout loop) — independent
8. Change 3c (typed review schemas) + Change 7 (prompt alignment) — depends on Change 1
9. Test updates — after all code changes
