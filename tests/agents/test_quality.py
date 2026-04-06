# tests/agents/test_quality.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from deep_agents.schemas import ReviewResult, ReviserOutput, FinalResult


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
            {
                "claim_text": "报告中的销量声明与证据一致",
                "source_url": "https://example.com/fact",
                "status": "verified",
            }
        ],
        missing_aspects=[],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_result)
    sample_state["full_report"] = "# 报告\n\n内容..."
    sample_state["revision_count"] = 0

    with patch("deep_agents.agents.reviewer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = (
            mock_chained
        )
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
        issues=[{"id": "issue_1", "description": "缺少数据支撑"}],
        claim_checks=[],
        missing_aspects=["数据来源"],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_result)
    sample_state["full_report"] = "# 报告"
    sample_state["revision_count"] = 0

    with patch("deep_agents.agents.reviewer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = (
            mock_chained
        )
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
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = (
            mock_chained
        )
        result = await reviewer_node(sample_state, mock_config)

    assert isinstance(result, Command)
    assert result.goto == "final_check"


async def test_reviewer_rejects_legacy_source_id_claim_checks(
    sample_state, mock_config
) -> None:
    from deep_agents.agents.reviewer import reviewer_node

    mock_result = ReviewResult(
        quality_score=6,
        verdict="fail",
        issues=[],
        claim_checks=[
            {
                "claim_text": "旧版契约",
                "source_id": "src_legacy",
                "status": "unsupported",
            }
        ],
        missing_aspects=[],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_result)
    sample_state["full_report"] = "# 报告"
    sample_state["revision_count"] = 0

    with patch("deep_agents.agents.reviewer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = (
            mock_chained
        )
        with pytest.raises(ValueError, match="source_url"):
            await reviewer_node(sample_state, mock_config)


@pytest.mark.parametrize(
    "claim_check",
    [
        {"claim_text": "缺失 source_url", "status": "unsupported"},
        {"claim_text": "空 source_url", "source_url": "", "status": "unsupported"},
        {"claim_text": "空白 source_url", "source_url": "   ", "status": "unsupported"},
    ],
)
async def test_reviewer_rejects_missing_or_blank_source_url_in_claim_checks(
    sample_state, mock_config, claim_check
) -> None:
    from deep_agents.agents.reviewer import reviewer_node

    mock_result = ReviewResult(
        quality_score=6,
        verdict="fail",
        issues=[],
        claim_checks=[claim_check],
        missing_aspects=[],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_result)
    sample_state["full_report"] = "# 报告"
    sample_state["revision_count"] = 0

    with patch("deep_agents.agents.reviewer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = (
            mock_chained
        )
        with pytest.raises(ValueError, match="source_url"):
            await reviewer_node(sample_state, mock_config)


async def test_reviewer_rejects_unresolved_claim_check_source_url(
    sample_state, mock_config
) -> None:
    from deep_agents.agents.reviewer import reviewer_node

    sample_state["sources"] = [
        {"url": "https://example.com/known", "title": "known", "summary": ""}
    ]
    mock_result = ReviewResult(
        quality_score=6,
        verdict="fail",
        issues=[],
        claim_checks=[
            {
                "claim_text": "引用了未知来源",
                "source_url": "https://example.com/unknown",
                "status": "unsupported",
            }
        ],
        missing_aspects=[],
    )
    mock_chained = MagicMock()
    mock_chained.ainvoke = AsyncMock(return_value=mock_result)
    sample_state["full_report"] = "# 报告"
    sample_state["revision_count"] = 0

    with patch("deep_agents.agents.reviewer.init_chat_model") as mock_init:
        mock_init.return_value.with_structured_output.return_value.with_retry.return_value = (
            mock_chained
        )
        with pytest.raises(ValueError, match="Unresolved source_url"):
            await reviewer_node(sample_state, mock_config)


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
    assert mock_init.call_args.kwargs["disable_streaming"] is True


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
    assert mock_init.call_args.kwargs["disable_streaming"] is True
