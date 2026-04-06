import operator
from enum import Enum
from typing import Annotated

from langgraph.graph import MessagesState
from typing_extensions import TypedDict


def override_reducer(current_value, new_value):
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    return operator.add(current_value, new_value)


class ResearchPhase(str, Enum):
    INIT = "init"
    PLANNING = "planning"
    RESEARCHING = "researching"
    ANALYZING = "analyzing"
    WRITING = "writing"
    REVIEWING = "reviewing"
    REVISING = "revising"
    RE_RESEARCHING = "re_researching"
    COMPLETED = "completed"


class AgentLog(TypedDict):
    timestamp: str
    agent: str
    action: str
    input_summary: str
    output_summary: str
    duration_ms: int
    tokens_used: int


class ResearchInputState(MessagesState, total=False):
    object_context: str | None


class ResearchState(ResearchInputState):
    need_clarification: bool
    clarification_question: str

    research_goal: str
    confirmed_constraints: list[str]
    open_dimensions: list[str]
    language: str

    research_type: str
    hypotheses: list[dict]
    sections: list[dict]
    outline_status: str
    outline_revision_count: int

    facts: Annotated[list[dict], override_reducer]
    data_points: Annotated[list[dict], override_reducer]
    hypothesis_evidence: Annotated[list[dict], override_reducer]
    charts: Annotated[list[dict], override_reducer]
    insights: Annotated[list[dict], override_reducer]
    contradictions: Annotated[list[dict], override_reducer]
    sources: Annotated[list[dict], override_reducer]
    open_questions: Annotated[list[dict], override_reducer]
    section_drafts: Annotated[list[dict], override_reducer]

    full_report: str
    review_result: dict | None
    revision_count: int
    final_result: dict | None


class SectionState(TypedDict):
    section_id: str
    section_title: str
    section_description: str
    search_queries: list[str]
    research_goal: str
    hypotheses: list[dict]
    language: str

    search_results: list[dict]
    section_facts: list[dict]
    section_insights: list[str]
    section_hypothesis_evidence: list[dict]
    section_contradictions: list[dict]
    section_entities: list[dict]
    missing_info: list[str]
    section_data_points: list[dict]
    section_charts: list[dict]
    section_time_series: list[dict]
    section_sources: list[dict]


class AgentState(ResearchState, total=False):
    phase: str
    logs: list[AgentLog]
    raw_notes: Annotated[list[str], override_reducer]
    notes: Annotated[list[str], override_reducer]
