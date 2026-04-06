"""Tests for planner_node and outline_reviser_node."""
from unittest.mock import AsyncMock, MagicMock, patch
from deep_agents.schemas import PlannerSection, RevisedOutline, Section, SimplifiedPlan


async def test_planner_node_returns_normalized_shape(mock_config):
    from deep_agents.agents.planner import planner_node

    mock_plan = SimplifiedPlan(
        research_type="trend_analysis",
        hypotheses=["X is rising"],
        sections=[
            PlannerSection(title="T", description="D", search_queries=["q"])
        ],
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_plan)
    state = {
        "research_goal": "了解趋势",
        "confirmed_constraints": [],
        "open_dimensions": [],
        "language": "zh",
    }
    with patch("deep_agents.agents.planner.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        result = await planner_node(state, mock_config)

    assert result["research_type"] == "trend_analysis"
    assert result["outline_status"] == "provisional"
    assert result["outline_revision_count"] == 0
    assert "budget" not in result

    # Normalized hypothesis shape
    assert len(result["hypotheses"]) == 1
    h = result["hypotheses"][0]
    assert h["id"] == "h_1"
    assert h["statement"] == "X is rising"
    assert h["status"] == "untested"
    assert h["evidence_needed"] == []

    # Normalized section shape
    assert len(result["sections"]) == 1
    s = result["sections"][0]
    assert s["id"] == "sec_1"
    assert s["title"] == "T"
    assert s["priority"] == 1


async def test_planner_node_ids_increment_correctly(mock_config):
    from deep_agents.agents.planner import planner_node

    mock_plan = SimplifiedPlan(
        research_type="market_overview",
        hypotheses=["H1", "H2"],
        sections=[
            PlannerSection(title="S1", description="D1", search_queries=["q1"]),
            PlannerSection(title="S2", description="D2", search_queries=["q2"]),
            PlannerSection(title="S3", description="D3", search_queries=["q3"]),
        ],
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_plan)
    state = {
        "research_goal": "市场",
        "confirmed_constraints": [],
        "open_dimensions": [],
        "language": "zh",
    }
    with patch("deep_agents.agents.planner.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        result = await planner_node(state, mock_config)

    assert [h["id"] for h in result["hypotheses"]] == ["h_1", "h_2"]
    assert [s["id"] for s in result["sections"]] == ["sec_1", "sec_2", "sec_3"]
    assert [s["priority"] for s in result["sections"]] == [1, 2, 3]


async def test_planner_node_disables_streaming_for_structured_output(mock_config):
    from deep_agents.agents.planner import planner_node

    mock_plan = SimplifiedPlan(
        research_type="market_overview",
        hypotheses=[],
        sections=[
            PlannerSection(title="S1", description="D1", search_queries=["q1"]),
        ],
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_plan)
    state = {
        "research_goal": "市场",
        "confirmed_constraints": [],
        "open_dimensions": [],
        "language": "zh",
    }
    with patch("deep_agents.agents.planner.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        await planner_node(state, mock_config)

    assert m.call_args.kwargs["disable_streaming"] is True


def test_planner_prompt_requires_non_empty_sections_and_no_legacy_source_bias() -> None:
    from deep_agents.prompts import planner_prompt

    assert "设计 3-6 个研究章节" in planner_prompt
    assert "如果无法给出 3 个章节，宁可重试推理，也不要输出空 sections" in planner_prompt
    assert "BoF、WWD、Vogue Runway、Lyst、Edited" not in planner_prompt


async def test_outline_reviser_increments_count(mock_config):
    from deep_agents.agents.outline_reviser import outline_reviser_node

    mock_revised = RevisedOutline(
        sections=[Section(id="sec_1", title="T", description="D", search_queries=["q"], priority=1)]
    )
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_revised)
    state = {
        "research_goal": "了解趋势",
        "sections": [{"id": "sec_1", "title": "T", "description": "D", "search_queries": ["q"], "priority": 1}],
        "hypothesis_evidence": [],
        "outline_revision_count": 0,
    }
    with patch("deep_agents.agents.outline_reviser.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        result = await outline_reviser_node(state, mock_config)

    assert result["outline_revision_count"] == 1
    assert result["outline_status"] == "revised"


async def test_outline_reviser_disables_streaming_for_structured_output(mock_config):
    from deep_agents.agents.outline_reviser import outline_reviser_node

    mock_revised = RevisedOutline(sections=[])
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=mock_revised)
    state = {
        "research_goal": "了解趋势",
        "sections": [],
        "hypothesis_evidence": [],
        "outline_revision_count": 0,
    }
    with patch("deep_agents.agents.outline_reviser.init_chat_model") as m:
        m.return_value.with_structured_output.return_value.with_retry.return_value = mock_chain
        await outline_reviser_node(state, mock_config)

    assert m.call_args.kwargs["disable_streaming"] is True
