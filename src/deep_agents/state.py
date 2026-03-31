import operator
from typing import Annotated

from langgraph.graph import add_messages
from typing_extensions import TypedDict


class ResearchState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    object_context: str | None

    need_clarification: bool
    clarification_question: str
    clarification_answer: str

    research_goal: str
    confirmed_constraints: list[str]
    open_dimensions: list[str]
    language: str

    research_type: str
    hypotheses: list[dict]
    sections: list[dict]
    budget: dict
    outline_status: str

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


class SectionState(TypedDict, total=False):
    section_id: str
    section_title: str
    section_description: str
    section_priority: int
    section_queries: list[str]

    scout_output: dict
    analyst_output: dict
    data_wiz_output: dict
    section_sources: list[dict]
