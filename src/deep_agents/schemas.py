from pydantic import BaseModel, Field


class Summary(BaseModel):
    summary: str
    key_excerpts: str


class ResearchComplete(BaseModel):
    reason: str


class ResearchBrief(BaseModel):
    need_clarification: bool
    clarification_question: str
    research_goal: str
    confirmed_constraints: list[str]
    open_dimensions: list[str]
    language: str = "zh"


class Hypothesis(BaseModel):
    id: str
    statement: str
    evidence_needed: list[str]
    status: str = "untested"


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
    outline_status: str = "provisional"


class RevisedOutline(BaseModel):
    sections: list[Section]
    outline_status: str = "revised"


class AnalystOutput(BaseModel):
    facts: list[str] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class DataWizOutput(BaseModel):
    data_points: list[str] = Field(default_factory=list)
    charts: list[str] = Field(default_factory=list)
    hypothesis_evidence: list[str] = Field(default_factory=list)


class SectionDraft(BaseModel):
    section_id: str
    content: str
    citations: list[str] = Field(default_factory=list)
    charts_used: list[str] = Field(default_factory=list)
    weak_claims: list[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    quality_score: float
    verdict: str
    issues: list[str] = Field(default_factory=list)
    claim_checks: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)


class ReviserOutput(BaseModel):
    full_report: str
    changes_made: list[str] = Field(default_factory=list)
    addressed_issues: list[str] = Field(default_factory=list)
    unable_to_address: list[str] = Field(default_factory=list)


class FinalResult(BaseModel):
    resolved_issues: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    new_issues: list[str] = Field(default_factory=list)
    final_score: float
    final_verdict: str
    publication_readiness: str
    final_comments: str
