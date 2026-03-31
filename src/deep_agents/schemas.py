from typing import Literal

from pydantic import BaseModel, Field, StrictInt


class Summary(BaseModel):
    summary: str
    key_excerpts: str


class ResearchComplete(BaseModel):
    reason: str


class ResearchBrief(BaseModel):
    need_clarification: bool
    clarification_question: str = ""
    research_goal: str = ""
    confirmed_constraints: list[str] = Field(default_factory=list)
    open_dimensions: list[str] = Field(default_factory=list)
    language: str = "zh"


class Hypothesis(BaseModel):
    id: str
    statement: str
    evidence_needed: list[str]
    status: Literal["untested", "supported", "refuted", "partial"] = "untested"


class Section(BaseModel):
    id: str
    title: str
    description: str
    search_queries: list[str]
    priority: int


class ArchitectPlan(BaseModel):
    research_type: str
    hypotheses: list[Hypothesis]
    sections: list[Section]
    budget: dict
    outline_status: Literal["provisional"] = "provisional"


class RevisedOutline(BaseModel):
    sections: list[Section]
    outline_status: Literal["revised"] = "revised"


class AnalystOutput(BaseModel):
    section_facts: list[dict] = Field(default_factory=list)
    section_insights: list[str] = Field(default_factory=list)
    section_hypothesis_evidence: list[dict] = Field(default_factory=list)
    section_contradictions: list[dict] = Field(default_factory=list)
    section_entities: list[dict] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)


class DataWizOutput(BaseModel):
    section_data_points: list[dict] = Field(default_factory=list)
    section_charts: list[dict] = Field(default_factory=list)
    section_time_series: list[dict] = Field(default_factory=list)


class SectionDraft(BaseModel):
    section_id: str
    content: str
    citations: list[dict] = Field(default_factory=list)
    charts_used: list[str] = Field(default_factory=list)
    weak_claims: list[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    quality_score: StrictInt = Field(ge=1, le=10)
    verdict: Literal["pass", "fail"]
    issues: list[dict] = Field(default_factory=list)
    claim_checks: list[dict] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)


class ReviserOutput(BaseModel):
    full_report: str
    changes_made: list[str] = Field(default_factory=list)
    addressed_issues: list[str] = Field(default_factory=list)
    unable_to_address: list[str] = Field(default_factory=list)


class FinalResult(BaseModel):
    resolved_issues: list[dict] = Field(default_factory=list)
    unresolved_issues: list[dict] = Field(default_factory=list)
    new_issues: list[dict] = Field(default_factory=list)
    final_score: StrictInt = Field(ge=1, le=10)
    final_verdict: Literal["approved", "rejected"]
    publication_readiness: Literal["ready", "needs_review"]
    final_comments: str
