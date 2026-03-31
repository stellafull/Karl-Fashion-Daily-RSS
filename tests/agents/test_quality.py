# tests/agents/test_quality.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from deep_agents.schemas import ReviewResult, ReviserOutput, FinalResult


async def test_reviewer_pass_on_high_score(sample_state, mock_config):
    from deep_agents.agents.reviewer import reviewer_node

    mock_result = ReviewResult(
        quality_score=8,
        verdict="pass",
        issues=[],
        claim_checks=[{"claim_text": "市场规模3620亿", "source_id": "src_001", "status": "verified"}],
        missing_aspects=[],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_result)
    sample_state["full_report"] = "# 报告\n\n内容..."

    with patch("deep_agents.agents.reviewer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await reviewer_node(sample_state, mock_config)

    assert result["review_result"]["quality_score"] == 8
    assert result["review_result"]["verdict"] == "pass"


async def test_reviser_updates_report_and_increments_count(sample_state, mock_config):
    from deep_agents.agents.reviser import reviser_node

    mock_output = ReviserOutput(
        full_report="# 修订后报告\n\n改进内容...",
        changes_made=["添加来源引用"],
        addressed_issues=["issue_1"],
        unable_to_address=[],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_output)

    sample_state["full_report"] = "# 原始报告"
    sample_state["review_result"] = {"quality_score": 5, "verdict": "fail", "issues": [{"id": "issue_1"}]}
    sample_state["revision_count"] = 0

    with patch("deep_agents.agents.reviser.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await reviser_node(sample_state, mock_config)

    assert "修订后" in result["full_report"]
    assert result["revision_count"] == 1


async def test_final_check_approved(sample_state, mock_config):
    from deep_agents.agents.final_check import final_check_node

    mock_result = FinalResult(
        resolved_issues=[{"id": "issue_1"}],
        unresolved_issues=[],
        new_issues=[],
        final_score=8,
        final_verdict="approved",
        publication_readiness="ready",
        final_comments="报告质量良好，可以发布。",
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_result)

    sample_state["full_report"] = "# 最终报告"
    sample_state["review_result"] = {"quality_score": 8, "verdict": "pass", "issues": []}
    sample_state["revision_count"] = 1

    with patch("deep_agents.agents.final_check.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await final_check_node(sample_state, mock_config)

    assert result["final_result"]["publication_readiness"] == "ready"
    assert result["final_result"]["final_verdict"] == "approved"
