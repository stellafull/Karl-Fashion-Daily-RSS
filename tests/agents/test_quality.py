# tests/agents/test_quality.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from deep_agents.schemas import (
    ClaimCheck,
    FinalResult,
    ResolutionItem,
    ReviewIssue,
    ReviewResult,
    ReviserOutput,
)


def test_reviewer_prompt_uses_url_based_claim_checks_contract() -> None:
    from deep_agents.prompts import reviewer_prompt

    assert "claim_text/source_url/status" in reviewer_prompt
    assert "source_id" not in reviewer_prompt


async def test_reviewer_pass_returns_command_to_final_check(
    sample_state, mock_config
) -> None:
    from deep_agents.agents.reviewer import reviewer_node
    from langgraph.types import Command

    mock_result = ReviewResult(
        quality_score=8,
        verdict="pass",
        issues=[],
        claim_checks=[
            ClaimCheck(
                claim_text="报告中的销量声明与证据一致",
                source_url="https://example.com/fact",
                status="verified",
            )
        ],
        missing_aspects=[],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_result)
    sample_state["full_report"] = "# 报告\n\n内容..."
    sample_state["revision_count"] = 0

    with patch("deep_agents.agents.reviewer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await reviewer_node(sample_state, mock_config)

    assert isinstance(result, Command)
    assert result.goto == "final_check"
    assert result.update["review_result"]["verdict"] == "pass"
    assert result.update["review_result"]["quality_score"] == 8
    assert result.update["review_result"]["claim_checks"][0]["source_url"] == "https://example.com/fact"
    assert mock_init.call_args.kwargs["disable_streaming"] is True


async def test_reviewer_fail_returns_command_to_reviser(
    sample_state, mock_config
) -> None:
    from deep_agents.agents.reviewer import reviewer_node
    from langgraph.types import Command

    mock_result = ReviewResult(
        quality_score=4,
        verdict="fail",
        issues=[ReviewIssue(type="evidence", severity="major", description="缺少数据支撑")],
        claim_checks=[],
        missing_aspects=["数据来源"],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_result)
    sample_state["full_report"] = "# 报告"
    sample_state["revision_count"] = 0

    with patch("deep_agents.agents.reviewer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await reviewer_node(sample_state, mock_config)

    assert isinstance(result, Command)
    assert result.goto == "reviser"
    assert result.update["review_result"]["verdict"] == "fail"


async def test_reviewer_max_revisions_goes_to_final_check(
    sample_state, mock_config
) -> None:
    from deep_agents.agents.reviewer import reviewer_node
    from langgraph.types import Command

    mock_result = ReviewResult(
        quality_score=4,
        verdict="fail",
        issues=[],
        claim_checks=[],
        missing_aspects=[],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_result)
    sample_state["full_report"] = "# 报告"
    sample_state["revision_count"] = 2

    with patch("deep_agents.agents.reviewer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await reviewer_node(sample_state, mock_config)

    assert isinstance(result, Command)
    assert result.goto == "final_check"


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
    sample_state["review_result"] = {"quality_score": 5, "verdict": "fail", "issues": []}
    sample_state["revision_count"] = 0

    with patch("deep_agents.agents.reviser.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = mock_chained
        result = await reviser_node(sample_state, mock_config)

    assert "修订后" in result["full_report"]
    assert result["revision_count"] == 1
    assert mock_init.call_args.kwargs["disable_streaming"] is True


async def test_final_check_approved(sample_state, mock_config):
    from deep_agents.agents.final_check import final_check_node

    mock_result = FinalResult(
        resolved_issues=[ResolutionItem(description="issue_1", status="fixed")],
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
    assert mock_init.call_args.kwargs["disable_streaming"] is True
