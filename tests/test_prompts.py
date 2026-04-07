from deep_agents.prompts import (
    STRICT_JSON_OUTPUT_RULES,
    analyst_prompt,
    clarify_prompt,
    compress_search_prompt,
    data_wiz_prompt,
    deep_scout_prompt,
    final_check_prompt,
    planner_prompt,
    reviewer_prompt,
    reviser_prompt,
    summarize_webpage_prompt,
    writer_prompt,
)


def test_planner_prompt_output_schema() -> None:
    """Verify planner_prompt enforces simplified plan output schema."""
    assert 'research_type' in planner_prompt
    assert 'hypotheses' in planner_prompt
    assert 'sections' in planner_prompt
    # Verify it does NOT mention old fields that were removed in simplification
    assert 'evidence_needed' not in planner_prompt
    assert 'outline_status' not in planner_prompt
    assert 'budget' not in planner_prompt
    assert 'max_searches' not in planner_prompt


def test_deep_scout_prompt_removes_max_searches() -> None:
    """Verify deep_scout_prompt does not reference {max_searches}."""
    assert '{max_searches}' not in deep_scout_prompt
    assert '预算：最多' not in deep_scout_prompt
    # Verify it still has the core strategy elements
    assert 'tavily_search' in deep_scout_prompt
    assert 'think_tool' in deep_scout_prompt
    assert 'analyze_image' in deep_scout_prompt


def test_structured_prompts_use_strict_json_contract() -> None:
    for prompt in (
        clarify_prompt,
        planner_prompt,
        analyst_prompt,
        data_wiz_prompt,
        writer_prompt,
        reviewer_prompt,
        reviser_prompt,
        final_check_prompt,
        summarize_webpage_prompt,
    ):
        assert STRICT_JSON_OUTPUT_RULES in prompt
        assert "现在直接输出最终 JSON。" in prompt
        assert "```json" in prompt


def test_compress_search_prompt_outputs_plain_markdown_text() -> None:
    assert "直接输出压缩后的 Markdown 文本" in compress_search_prompt
    assert "不要输出 JSON" in compress_search_prompt
