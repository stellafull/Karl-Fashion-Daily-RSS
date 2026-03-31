from deep_agents.schemas import (
    AnalystOutput,
    ArchitectPlan,
    DataWizOutput,
    FinalResult,
    Hypothesis,
    ResearchBrief,
    ResearchComplete,
    ReviewResult,
    ReviserOutput,
    RevisedOutline,
    Section,
    SectionDraft,
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


def test_hypothesis_defaults() -> None:
    model = Hypothesis(id="h1", statement="X is rising", evidence_needed=["sales data"])
    assert model.status == "untested"
    assert model.id == "h1"


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


def test_architect_plan_defaults() -> None:
    model = ArchitectPlan(
        research_type="trend_analysis",
        hypotheses=[Hypothesis(id="h1", statement="A", evidence_needed=["B"])],
        sections=[
            Section(
                id="s1",
                title="T1",
                description="D1",
                search_queries=["q1"],
                priority=1,
            )
        ],
        budget={"max_sections": 6},
    )
    assert model.outline_status == "provisional"
    assert len(model.hypotheses) == 1


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
        verdict="revise",
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
        final_score=95,
        final_verdict="approved",
        publication_readiness="ready",
        final_comments="looks good",
    )
    assert isinstance(model.final_score, int)
    assert model.resolved_issues == [{"issue": "i1"}]
    assert model.unresolved_issues == [{"issue": "i2"}]
    assert model.new_issues == [{"issue": "i3"}]
    assert model.final_verdict == "approved"
