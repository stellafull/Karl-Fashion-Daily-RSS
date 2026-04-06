from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt


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


class PlannerSection(BaseModel):
    title: str
    description: str
    search_queries: list[str]


class SimplifiedPlan(BaseModel):
    research_type: str
    hypotheses: list[str]
    sections: list[PlannerSection] = Field(min_length=1)


class Section(BaseModel):
    id: str
    title: str
    description: str
    search_queries: list[str]
    priority: int


class RevisedOutline(BaseModel):
    sections: list[Section]
    outline_status: Literal["revised"] = "revised"


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    title: str
    summary: str


class SectionFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    source_url: str
    importance: str


class HypothesisEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_statement: str
    evidence_type: str
    content: str
    source_url: str


class Contradiction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_a: str
    claim_b: str
    source_url_a: str
    source_url_b: str


class DataPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: str | int | float
    unit: str | None = None
    year: int | None = None
    source_url: str
    category: str | None = None
    confidence: str | None = None


class AnalystOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_facts: list[SectionFact] = Field(default_factory=list)
    section_insights: list[str] = Field(default_factory=list)
    section_hypothesis_evidence: list[HypothesisEvidence] = Field(default_factory=list)
    section_contradictions: list[Contradiction] = Field(default_factory=list)
    section_entities: list[dict] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)


class DataWizOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_data_points: list[DataPoint] = Field(default_factory=list)
    section_charts: list[dict] = Field(default_factory=list)
    section_time_series: list[dict] = Field(default_factory=list)


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    url: str
    title: str


class SectionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    citations: list[Citation] = Field(default_factory=list)
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
