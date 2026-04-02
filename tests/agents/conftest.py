import pytest


@pytest.fixture
def mock_config():
    return {"configurable": {"thread_id": "test-001"}}


@pytest.fixture
def sample_state():
    return {
        "research_goal": "2025年中国奢侈品市场深度研究",
        "language": "zh",
        "research_type": "market_overview",
        "sections": [
            {
                "id": "sec_1",
                "title": "市场概况",
                "description": "规模与增速分析",
                "search_queries": ["luxury market china 2025"],
                "priority": 1,
            }
        ],
        "hypotheses": [],
        "facts": [],
        "data_points": [],
        "hypothesis_evidence": [],
        "charts": [],
        "insights": [],
        "contradictions": [],
        "sources": [],
        "open_questions": [],
        "section_drafts": [],
        "failed_sections": [],
        "full_report": "",
        "review_result": None,
        "revision_count": 0,
        "final_result": None,
    }
