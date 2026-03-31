import operator
from enum import Enum
from typing import Annotated

from langgraph.graph import add_messages
from typing_extensions import TypedDict


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


class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    object_context: str | None

    need_clarification: bool
    clarification_question: str

    research_goal: str
    confirmed_constraints: list[str]
    open_dimensions: list[str]
    language: str

    research_type: str
    hypotheses: list[dict]
    sections: list[dict]
    budget: dict
    outline_status: str
    outline_revision_count: int

    facts: Annotated[list[dict], operator.add]
    data_points: Annotated[list[dict], operator.add]
    hypothesis_evidence: Annotated[list[dict], operator.add]
    charts: Annotated[list[dict], operator.add]
    insights: Annotated[list[dict], operator.add]
    contradictions: Annotated[list[dict], operator.add]
    sources: Annotated[list[dict], operator.add]
    open_questions: Annotated[list[dict], operator.add]
    section_drafts: Annotated[list[dict], operator.add]

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
    budget: dict
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
