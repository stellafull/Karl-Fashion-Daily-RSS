import pytest
from pydantic import ValidationError

from deep_agents.schemas import (
    AnalystOutput,
    Citation,
    Contradiction,
    DataPoint,
    DataWizOutput,
    FinalResult,
    HypothesisEvidence,
    PlannerSection,
    ResearchBrief,
    ResearchComplete,
    ReviewResult,
    RevisedOutline,
    ReviserOutput,
    Section,
    SectionDraft,
    SectionFact,
    SimplifiedPlan,
    Source,
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


def test_simplified_plan_rejects_empty_sections() -> None:
    with pytest.raises(ValidationError):
        SimplifiedPlan(
            research_type="trend_analysis",
            hypotheses=["A is rising"],
            sections=[],
        )


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


def test_analyst_output_rejects_unexpected_top_level_fields() -> None:
    with pytest.raises(ValidationError):
        AnalystOutput(unexpected_field="x")


def test_data_wiz_output_rejects_unexpected_top_level_fields() -> None:
    with pytest.raises(ValidationError):
        DataWizOutput(unexpected_field="x")


def test_source_schema_requires_url_title_summary() -> None:
    model = Source(
        url="https://example.com/a",
        title="Example",
        summary="Summary text",
    )
    assert model.url == "https://example.com/a"


def test_source_rejects_legacy_source_id_field() -> None:
    with pytest.raises(ValidationError):
        Source(
            source_id="src_001",
            url="https://example.com/a",
            title="Example",
            summary="Summary text",
        )


def test_section_fact_requires_source_url() -> None:
    model = SectionFact(
        content="销量增长",
        source_url="https://example.com/a",
        importance="high",
    )
    assert model.source_url == "https://example.com/a"


def test_hypothesis_evidence_uses_statement_not_id() -> None:
    model = HypothesisEvidence(
        hypothesis_statement="消费者偏好转向功能性服饰",
        evidence_type="supports",
        content="多个来源提到功能性需求增强",
        source_url="https://example.com/b",
    )
    assert model.hypothesis_statement.startswith("消费者偏好")


def test_contradiction_requires_dual_source_urls() -> None:
    model = Contradiction(
        claim_a="线上渠道增速放缓",
        claim_b="线上渠道依然高速增长",
        source_url_a="https://example.com/a",
        source_url_b="https://example.com/b",
    )
    assert model.source_url_a == "https://example.com/a"
    assert model.source_url_b == "https://example.com/b"


def test_contradiction_rejects_legacy_source_id_fields() -> None:
    with pytest.raises(ValidationError):
        Contradiction(
            claim_a="线上渠道增速放缓",
            claim_b="线上渠道依然高速增长",
            source_url_a="https://example.com/a",
            source_url_b="https://example.com/b",
            source_id_a="src_a",
            source_id_b="src_b",
        )


def test_data_point_has_no_runtime_id_field() -> None:
    model = DataPoint(
        name="market_size",
        value=3457,
        source_url="https://example.com/c",
    )
    dumped = model.model_dump()
    assert "id" not in dumped


def test_citation_schema_requires_claim_url_title() -> None:
    model = Citation(
        claim="市场规模提升",
        url="https://example.com/citation",
        title="2026 Market Report",
    )
    assert model.url == "https://example.com/citation"


def test_citation_rejects_legacy_source_id_field() -> None:
    with pytest.raises(ValidationError):
        Citation(
            claim="市场规模提升",
            url="https://example.com/citation",
            title="2026 Market Report",
            source_id="src_001",
        )


def test_section_draft_has_no_model_facing_section_id() -> None:
    model = SectionDraft(
        content="## 市场概况",
        citations=[{"claim": "增长", "url": "https://example.com/a", "title": "A"}],
        charts_used=[],
        weak_claims=[],
    )
    dumped = model.model_dump()
    assert "section_id" not in dumped


def test_old_source_id_payload_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SectionFact(
            content="增长",
            source_url="https://example.com/a",
            source_id="src_001",
            importance="high",
        )


def test_old_hypothesis_id_payload_is_rejected() -> None:
    with pytest.raises(ValidationError):
        HypothesisEvidence(
            hypothesis_statement="消费者偏好转向功能性服饰",
            hypothesis_id="h_001",
            evidence_type="supports",
            content="多个来源提到功能性需求增强",
            source_url="https://example.com/b",
        )


def test_data_point_rejects_legacy_id_field() -> None:
    with pytest.raises(ValidationError):
        DataPoint(
            id="dp_001",
            name="market_size",
            value=3457,
            source_url="https://example.com/c",
        )


def test_section_draft_rejects_legacy_section_id_field() -> None:
    with pytest.raises(ValidationError):
        SectionDraft(
            section_id="s1",
            content="## 市场概况",
            citations=[{"claim": "增长", "url": "https://example.com/a", "title": "A"}],
            charts_used=[],
            weak_claims=[],
        )


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
