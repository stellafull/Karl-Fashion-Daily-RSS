import operator
from typing import Annotated, get_args, get_origin, get_type_hints

from langgraph.graph import add_messages

from deep_agents.state import ResearchState, SectionState


def _assert_annotated_reducer(field_annotation: object, expected_reducer: object) -> None:
    assert get_origin(field_annotation) is Annotated
    metadata = get_args(field_annotation)[1:]
    assert expected_reducer in metadata


def test_state_types_are_importable() -> None:
    assert ResearchState.__name__ == "ResearchState"
    assert SectionState.__name__ == "SectionState"


def test_research_state_messages_uses_add_messages_reducer() -> None:
    hints = get_type_hints(ResearchState, include_extras=True)
    _assert_annotated_reducer(hints["messages"], add_messages)


def test_research_state_has_outline_revision_count() -> None:
    hints = get_type_hints(ResearchState, include_extras=True)
    assert "outline_revision_count" in hints


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
        _assert_annotated_reducer(hints[field_name], operator.add)


def test_section_state_matches_prd_shape() -> None:
    hints = get_type_hints(SectionState, include_extras=True)
    required_keys = {
        "section_id",
        "section_title",
        "section_description",
        "search_queries",
        "research_goal",
        "hypotheses",
        "budget",
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
        "section_priority",
        "section_queries",
        "scout_output",
        "analyst_output",
        "data_wiz_output",
    }

    assert required_keys.issubset(set(hints.keys()))
    assert removed_keys.isdisjoint(set(hints.keys()))
