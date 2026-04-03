import pytest
from pydantic import ValidationError

from deep_agents.schemas import (
    AnalystOutput,
    DataWizOutput,
    FinalResult,
    PlannerSection,
    ResearchBrief,
    ResearchComplete,
    ReviewResult,
    RevisedOutline,
    ReviserOutput,
    Section,
    SectionDraft,
    SimplifiedPlan,
    Summary,
)


def test_summary_schema() -> None:
    model = Summary(summary="short", key_excerpts="quote")
    assert model.summary == "short"
    assert model.key_excerpts == "quote"


def test_research_complete_schema() -> None:
    model = ResearchComplete(reason="enough evidence")
    assert model.reason == "enough evidence"


def test_research_brief_defaults() -> None:
    model = ResearchBrief(need_clarification=False)
    assert model.clarification_question == ""
    assert model.research_goal == ""
    assert model.confirmed_constraints == []
    assert model.open_dimensions == []
    assert model.language == "zh"


def test_planner_section_schema() -> None:
    model = PlannerSection(
        title="Market Snapshot",
        description="Topline market movement",
        search_queries=["fashion market 2026", "luxury sales report"],
    )
    assert model.title == "Market Snapshot"
    assert len(model.search_queries) == 2


def test_simplified_plan_schema() -> None:
    model = SimplifiedPlan(
        research_type="trend_analysis",
        hypotheses=["A is rising"],
        sections=[
            PlannerSection(
                title="T1",
                description="D1",
                search_queries=["q1"],
            )
        ],
    )
    assert model.research_type == "trend_analysis"
    assert len(model.hypotheses) == 1
    assert len(model.sections) == 1


def test_section_schema() -> None:
    model = Section(
        id="sec-1",
        title="Market Snapshot",
        description="Topline market movement",
        search_queries=["fashion market 2026", "luxury sales report"],
        priority=1,
    )
    assert model.title == "Market Snapshot"
    assert model.search_queries[0] == "fashion market 2026"


def test_revised_outline_defaults() -> None:
    model = RevisedOutline(
        sections=[
            Section(
                id="s1",
                title="T1",
                description="D1",
                search_queries=["q1"],
                priority=1,
            )
        ]
    )
    assert model.outline_status == "revised"
    assert len(model.sections) == 1


def test_analyst_output_defaults() -> None:
    model = AnalystOutput()
    assert model.section_facts == []
    assert model.section_insights == []
    assert model.section_hypothesis_evidence == []
    assert model.section_contradictions == []
    assert model.section_entities == []
    assert model.missing_info == []


def test_data_wiz_output_defaults() -> None:
    model = DataWizOutput()
    assert model.section_data_points == []
    assert model.section_charts == []
    assert model.section_time_series == []


def test_section_draft_defaults() -> None:
    model = SectionDraft(section_id="s1", content="body", citations=[{"url": "x"}])
    assert model.citations == [{"url": "x"}]
    assert model.charts_used == []
    assert model.weak_claims == []


def test_review_result_defaults() -> None:
    model = ReviewResult(
        quality_score=9,
        verdict="pass",
        issues=[{"issue": "weak evidence"}],
        claim_checks=[{"claim": "c1", "ok": False}],
    )
    assert isinstance(model.quality_score, int)
    assert model.issues == [{"issue": "weak evidence"}]
    assert model.claim_checks == [{"claim": "c1", "ok": False}]
    assert model.missing_aspects == []


def test_reviser_output_defaults() -> None:
    model = ReviserOutput(full_report="new draft")
    assert model.changes_made == []
    assert model.addressed_issues == []
    assert model.unable_to_address == []


def test_final_result_schema() -> None:
    model = FinalResult(
        resolved_issues=[{"issue": "i1"}],
        unresolved_issues=[{"issue": "i2"}],
        new_issues=[{"issue": "i3"}],
        final_score=9,
        final_verdict="approved",
        publication_readiness="ready",
        final_comments="looks good",
    )
    assert isinstance(model.final_score, int)
    assert model.resolved_issues == [{"issue": "i1"}]
    assert model.unresolved_issues == [{"issue": "i2"}]
    assert model.new_issues == [{"issue": "i3"}]
    assert model.final_verdict == "approved"


def test_revised_outline_status_invalid_value_raises() -> None:
    with pytest.raises(ValidationError):
        RevisedOutline(
            sections=[
                Section(
                    id="s1",
                    title="T1",
                    description="D1",
                    search_queries=["q1"],
                    priority=1,
                )
            ],
            outline_status="provisional",
        )


def test_review_result_verdict_invalid_value_raises() -> None:
    with pytest.raises(ValidationError):
        ReviewResult(quality_score=5, verdict="revise")


def test_final_result_invalid_enum_like_values_raise() -> None:
    with pytest.raises(ValidationError):
        FinalResult(
            final_score=8,
            final_verdict="pass",
            publication_readiness="pending",
            final_comments="x",
        )


def test_score_range_validation_raises_for_out_of_range_values() -> None:
    with pytest.raises(ValidationError):
        ReviewResult(quality_score=11, verdict="pass")
    with pytest.raises(ValidationError):
        FinalResult(
            final_score=0,
            final_verdict="approved",
            publication_readiness="ready",
            final_comments="x",
        )


def test_score_fields_reject_boolean_values() -> None:
    with pytest.raises(ValidationError):
        ReviewResult(quality_score=True, verdict="pass")
    with pytest.raises(ValidationError):
        FinalResult(
            final_score=False,
            final_verdict="approved",
            publication_readiness="ready",
            final_comments="x",
        )
