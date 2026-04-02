import operator
from importlib import import_module
from typing import Annotated, get_args, get_origin, get_type_hints

from langgraph.graph import add_messages

from deep_agents.state import (
    ResearchPhase,
    ResearchState,
    SectionState,
    override_reducer,
)


def _assert_annotated_reducer(field_annotation: object, expected_reducer: object) -> None:
    assert get_origin(field_annotation) is Annotated
    metadata = get_args(field_annotation)[1:]
    assert expected_reducer in metadata


def test_state_types_are_importable() -> None:
    assert ResearchState.__name__ == "ResearchState"
    assert SectionState.__name__ == "SectionState"
    assert ResearchState.__total__ is True
    assert SectionState.__total__ is True


def test_research_phase_compatibility_values() -> None:
    assert ResearchPhase.INIT.value == "init"
    assert ResearchPhase.PLANNING.value == "planning"
    assert ResearchPhase.RESEARCHING.value == "researching"
    assert ResearchPhase.ANALYZING.value == "analyzing"
    assert ResearchPhase.WRITING.value == "writing"
    assert ResearchPhase.REVIEWING.value == "reviewing"
    assert ResearchPhase.REVISING.value == "revising"
    assert ResearchPhase.RE_RESEARCHING.value == "re_researching"
    assert ResearchPhase.COMPLETED.value == "completed"


def test_research_state_messages_uses_add_messages_reducer() -> None:
    hints = get_type_hints(ResearchState, include_extras=True)
    _assert_annotated_reducer(hints["messages"], add_messages)


def test_research_state_has_outline_revision_count() -> None:
    hints = get_type_hints(ResearchState, include_extras=True)
    assert "outline_revision_count" in hints


def test_failed_sections_field_exists() -> None:
    hints = get_type_hints(ResearchState, include_extras=True)
    assert "failed_sections" in hints


def test_research_state_collection_fields_use_operator_add_reducer() -> None:
    hints = get_type_hints(ResearchState, include_extras=True)
    reducer_fields = [
        "facts",
        "data_points",
        "hypothesis_evidence",
        "charts",
        "insights",
        "contradictions",
        "sources",
        "open_questions",
        "section_drafts",
    ]

    for field_name in reducer_fields:
        _assert_annotated_reducer(hints[field_name], override_reducer)


def test_override_reducer_supports_add_and_explicit_override() -> None:
    current = [{"id": 1}]
    appended = override_reducer(current, [{"id": 2}])
    overridden = override_reducer(current, {"type": "override", "value": [{"id": 9}]})

    assert appended == [{"id": 1}, {"id": 2}]
    assert overridden == [{"id": 9}]


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


def test_research_state_has_no_budget_field() -> None:
    hints = get_type_hints(ResearchState, include_extras=True)
    assert "budget" not in hints


def test_state_compatibility_exports_smoke() -> None:
    state_module = import_module("deep_agents.state")
    assert hasattr(state_module, "ResearchPhase")
    assert hasattr(state_module, "AgentLog")
    assert hasattr(state_module, "AgentState")
    assert state_module.ResearchPhase.INIT.value == "init"
    assert "timestamp" in state_module.AgentLog.__annotations__
    agent_state_hints = get_type_hints(state_module.AgentState, include_extras=True)
    assert "phase" in agent_state_hints
    assert "raw_notes" in agent_state_hints
    assert "notes" in agent_state_hints
