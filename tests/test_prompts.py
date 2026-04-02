from deep_agents.prompts import planner_prompt, deep_scout_prompt


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
