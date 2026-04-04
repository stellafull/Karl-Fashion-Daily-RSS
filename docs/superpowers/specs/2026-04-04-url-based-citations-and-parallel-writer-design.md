# URL-Based Citations And Parallel Writer Design

## Goal

Pull the research pipeline back to its intended simple citation model:

- citations are URL-based, not `source_id`-based
- canonical source objects contain only `url`, `title`, and `summary`
- writer runs per-section in parallel
- cross-section de-duplication and stylistic unification happen only in `synthesizer`

This design intentionally removes provenance machinery that drifted into the implementation and is not required by the product goal.

## Design Decisions

- Canonical source identity is the `url`
- Canonical source shape is:

```python
{
    "url": str,
    "title": str,
    "summary": str,
}
```

- `source_id`, `source_id_a`, `source_id_b`, and source credibility scoring are removed from business-facing contracts
- `section_id` remains a runtime routing field where needed, but is not part of the model-facing citation contract
- `writer` receives only section-local evidence plus a global `sections_list`
- `writer` does not receive prior section body text or memoized content
- `synthesizer` becomes the only layer responsible for:
  - global de-duplication
  - section-to-section consistency
  - ordering polish
  - unified terminology and tone

## Main State Machine

The top-level LangGraph topology stays the same:

```text
START
  -> clarify
  -> planner
  -> section_pipeline
       -> deep_scout
       -> analyst
       -> data_wiz
  -> outline_reviser (optional once)
  -> lead_writer
  -> synthesizer
  -> trend_triangulator (conditional)
  -> reviewer
       -> reviser -> reviewer
       -> final_check
END
```

The graph architecture is not the problem. The problem is the current data contract between collection, merge, and writing.

## Section Subgraph Contract

The section subgraph should produce:

```python
section_sources: list[Source]
section_facts: list[SectionFact]
section_hypothesis_evidence: list[HypothesisEvidence]
section_contradictions: list[Contradiction]
section_data_points: list[DataPoint]
section_charts: list[dict]
section_time_series: list[dict]
```

Where:

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
    hypothesis_id: str
    evidence_type: str
    content: str
    source_url: str


class Contradiction(BaseModel):
    claim_a: str
    claim_b: str
    source_url_a: str
    source_url_b: str


class DataPoint(BaseModel):
    id: str
    name: str
    value: str | int | float
    unit: str | None = None
    year: int | None = None
    source_url: str
    category: str | None = None
    confidence: str | None = None
```

These are the product-facing evidence contracts. None of them require `source_id`.

## Field Flow

### `deep_scout`

Responsibilities:

- run search/tool loop
- collect raw search outputs
- normalize source objects into `url/title/summary`

Allowed source object:

```python
{
    "url": ...,
    "title": ...,
    "summary": ...,
}
```

Not allowed:

- `source_id`
- `credibility_score`
- synthetic ranking metadata

### `analyst`

Responsibilities:

- read `search_results`
- produce facts, hypothesis evidence, contradictions, entities, and open questions

Required citation behavior:

- every fact must carry `source_url`
- every hypothesis evidence record must carry `source_url`
- every contradiction must carry `source_url_a` and `source_url_b`

The schema, not only the prompt, must enforce this.

### `data_wiz`

Responsibilities:

- read `search_results`
- produce data points, chart definitions, time series

Required citation behavior:

- every data point must carry `source_url`

Again, this must be schema-level, not prompt-only.

### `section_pipeline` merge

Responsibilities:

- tag merged records with `section_id` for runtime filtering
- merge section-local lists into main graph state
- optionally validate that any `source_url` referenced by a record exists in the section's `section_sources`

Not responsible for:

- generating synthetic IDs
- globally remapping source identifiers
- teaching the model an internal linkage scheme

Validation should be URL-based if retained:

- facts: `source_url` must exist in section sources
- hypothesis evidence: same
- data points: same
- contradictions: `source_url_a/b` must exist in section sources

## Writer Design

### Current decision

`writer` should become parallel, not sequential.

Why:

- section drafts are logically independent
- serial writing forces unnecessary context coupling
- serial writing caused drift such as `previous_sections_memo` and prompt-level `section_id`
- global coherence is already the job of `synthesizer`

### Writer input contract

Each parallel writer task gets:

- `research_goal`
- `sections_list`
- current `section_title`
- current `section_description`
- current section `facts`
- current section `data_points`
- current section `charts`
- current section `contradictions`
- current section `hypothesis_evidence`
- `language`

It does **not** get:

- previous section body text
- previous section memo content
- `section_id` in the prompt

`section_id` may still be attached by runtime after model output, but it is not model-facing.

### Writer output contract

The model returns:

```python
class Citation(BaseModel):
    claim: str
    url: str
    title: str


class SectionDraft(BaseModel):
    content: str
    citations: list[Citation]
    charts_used: list[str]
    weak_claims: list[str]
```

Runtime may append `section_id` after validation.

## Synthesizer Design

`synthesizer` becomes the only global unification layer.

Responsibilities:

- merge section drafts
- remove repeated points across sections
- smooth transitions
- unify terminology and tone
- write executive summary
- write conclusion and hypothesis judgments

This is the correct place for whole-report coherence. It should not be approximated by serial writer context bleed.

## Prompt Changes

### Remove

- fixed source white-lists such as BoF / WWD / Vogue Runway / Lyst / Edited
- tier-1 / tier-2 source ranking language
- all `source_id` field requirements
- `章节ID：{section_id}` from writer prompt
- any writer prompt dependency on previous section body content

### Replace with

- evidence-quality wording:
  - prefer directly relevant, citable, information-dense sources
  - prefer official / first-party / primary data where available
- URL-based citation wording:
  - facts/evidence/data points must include `source_url`
  - citations must include `url` and `title`

## Keep / Modify / Delete

### Keep

- top-level graph topology
- section subgraph topology
- runtime `section_id` for section scoping and aggregation
- source fields: `url`, `title`, `summary`
- `sections_list` in writer prompt

### Modify

- `AnalystOutput` and `DataWizOutput` from loose `list[dict]` into strong typed models
- graph merge validation from `source_id`-based to URL-based
- writer from serial loop with contextual memoing to parallel section drafting
- citations and claim checks from `source_id`-based to URL-based

### Delete

- `source_id`, `source_id_a`, `source_id_b`
- `_global_source_id`
- source ID remapping tables in graph merge
- `credibility_score`
- previous section body memoing
- fixed source white-lists in prompts
- model-facing `section_id`
- `failed_sections` if fail-fast remains the graph policy

## File-Level Change Intent

| File | Intent |
|------|--------|
| `src/deep_agents/agents/deep_scout.py` | remove synthetic source metadata; keep `url/title/summary` only |
| `src/deep_agents/schemas.py` | introduce strong typed URL-based evidence models |
| `src/deep_agents/prompts.py` | remove `source_id` wording, remove source white-list, remove writer `section_id` and previous-content dependency |
| `src/deep_agents/agents/analyst.py` | switch to strong typed URL-based output |
| `src/deep_agents/agents/data_wiz.py` | switch to strong typed URL-based output |
| `src/deep_agents/graph.py` | remove source-id remap; perform only URL-based merge validation and section tagging |
| `src/deep_agents/agents/writer.py` | parallelize section writing; keep `sections_list`; drop previous section content dependency |
| `src/deep_agents/agents/reviewer.py` | consume URL-based claim references if needed |
| tests | rewrite around URL-based citations and parallel writer semantics |

## Non-Goals

- no redesign of the overall research graph
- no provider/model changes
- no extra ranking or trust-scoring system for sources
- no attempt to preserve `source_id` compatibility

## Review Notes

- The current implementation drifted from a simple citation design into a synthetic provenance system.
- The bug now surfacing (`Missing required source_id`) is evidence of that drift, not an isolated defect.
- The clean fix is to simplify the contract, not to keep patching `source_id` enforcement deeper into the graph.
