"""Tests for graph topology and section_pipeline_node gather behavior."""
import asyncio
import gc
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.graph import END
from langgraph.types import Command


def test_graph_compiles_without_error() -> None:
    from deep_agents.graph import build_research_graph

    graph = build_research_graph()
    assert graph is not None


def test_graph_exports_chat_friendly_input_schema() -> None:
    from deep_agents.graph import build_research_graph

    schema = build_research_graph().get_input_jsonschema()

    assert set(schema["properties"]) == {"messages", "object_context"}
    assert schema["required"] == ["messages"]
    assert "oneOf" in schema["properties"]["messages"]["items"]


def test_section_subgraph_compiles() -> None:
    from deep_agents.graph import build_section_subgraph

    sg = build_section_subgraph()
    assert sg is not None


def test_no_routing_functions_exported() -> None:
    """Routing functions are removed — they should not be importable from graph."""
    import deep_agents.graph as g

    for name in (
        "route_after_clarify",
        "fan_out_sections",
        "route_after_collection",
        "route_after_synthesis",
        "route_after_review",
    ):
        assert not hasattr(g, name), f"{name} should be removed from graph.py"


async def test_section_pipeline_node_merges_results() -> None:
    """section_pipeline_node gathers results from all sections and returns Command."""
    from deep_agents.graph import section_pipeline_node

    section_result = {
        "section_facts": [{"content": "fact1", "source_url": "https://example.com"}],
        "section_data_points": [{"name": "dp1", "value": 1, "source_url": "https://example.com"}],
        "section_hypothesis_evidence": [],
        "section_charts": [],
        "section_insights": ["insight1"],
        "section_contradictions": [],
        "section_sources": [{"url": "https://example.com", "title": "example", "summary": ""}],
        "missing_info": ["question1"],
    }

    state = {
        "sections": [
            {
                "id": "sec_1",
                "title": "市场概况",
                "description": "规模",
                "search_queries": ["q1"],
                "priority": 1,
            }
        ],
        "research_goal": "了解市场",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = AsyncMock(return_value=section_result)

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        result = await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})

    assert isinstance(result, Command)
    assert result.goto == "lead_writer"
    assert result.update["facts"]["type"] == "override"
    assert result.update["sources"]["type"] == "override"
    assert result.update["insights"]["type"] == "override"
    assert result.update["open_questions"]["type"] == "override"
    assert len(result.update["facts"]["value"]) == 1
    assert len(result.update["sources"]["value"]) == 1
    assert result.update["facts"]["value"][0]["section_id"] == "sec_1"
    assert result.update["sources"]["value"][0]["section_id"] == "sec_1"
    assert "source_id" not in result.update["facts"]["value"][0]
    assert "source_id" not in result.update["sources"]["value"][0]
    assert result.update["insights"]["value"][0]["section_id"] == "sec_1"
    assert result.update["open_questions"]["value"][0]["section_id"] == "sec_1"
    assert "failed_sections" not in result.update


async def test_section_pipeline_node_raises_when_any_section_subgraph_fails() -> None:
    """A section subgraph failure should hard-fail section_pipeline_node."""
    from deep_agents.graph import section_pipeline_node

    good_result = {
        "section_facts": [{"content": "fact from sec_1", "source_url": "https://example.com/1"}],
        "section_data_points": [],
        "section_hypothesis_evidence": [],
        "section_charts": [],
        "section_insights": [],
        "section_contradictions": [],
        "section_sources": [{"url": "https://example.com/1", "title": "one", "summary": ""}],
        "missing_info": [],
    }

    call_count = 0

    async def mock_ainvoke(inp, config=None):
        nonlocal call_count
        call_count += 1
        if inp["section_id"] == "sec_2":
            raise RuntimeError("LLM failed")
        return good_result

    state = {
        "sections": [
            {
                "id": "sec_1",
                "title": "市场概况",
                "description": "规模",
                "search_queries": ["q1"],
                "priority": 1,
            },
            {
                "id": "sec_2",
                "title": "竞争格局",
                "description": "品牌",
                "search_queries": ["q2"],
                "priority": 2,
            },
        ],
        "research_goal": "了解市场",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = mock_ainvoke

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        with pytest.raises(RuntimeError, match="LLM failed"):
            await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})


async def test_section_pipeline_cancels_sibling_sections_on_first_failure() -> None:
    """When one section fails, sibling section tasks should be cancelled immediately."""
    from deep_agents.graph import section_pipeline_node

    sibling_cancelled = asyncio.Event()
    sibling_started = asyncio.Event()

    async def mock_ainvoke(inp, config=None):
        if inp["section_id"] == "sec_1":
            sibling_started.set()
            try:
                await asyncio.sleep(0.2)
                return {
                    "section_facts": [],
                    "section_data_points": [],
                    "section_hypothesis_evidence": [],
                    "section_charts": [],
                    "section_insights": [],
                    "section_contradictions": [],
                    "section_sources": [],
                    "missing_info": [],
                }
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

        await sibling_started.wait()
        raise RuntimeError("section failed")

    state = {
        "sections": [
            {
                "id": "sec_1",
                "title": "市场概况",
                "description": "规模",
                "search_queries": ["q1"],
                "priority": 1,
            },
            {
                "id": "sec_2",
                "title": "竞争格局",
                "description": "品牌",
                "search_queries": ["q2"],
                "priority": 2,
            },
        ],
        "research_goal": "了解市场",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = mock_ainvoke

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        with pytest.raises(RuntimeError, match="section failed"):
            await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})

    await asyncio.sleep(0)
    assert sibling_cancelled.is_set()


async def test_section_pipeline_cancels_and_drains_children_on_external_cancellation() -> None:
    """External cancellation of section_pipeline_node should cancel and drain section tasks."""
    from deep_agents.graph import section_pipeline_node

    section_1_started = asyncio.Event()
    section_2_started = asyncio.Event()
    section_1_cancelled = asyncio.Event()
    section_2_cancelled = asyncio.Event()
    never_set = asyncio.Event()
    child_tasks: set[asyncio.Task] = set()

    async def mock_ainvoke(inp, config=None):
        current = asyncio.current_task()
        if current is not None:
            child_tasks.add(current)

        if inp["section_id"] == "sec_1":
            section_1_started.set()
            try:
                await never_set.wait()
            except asyncio.CancelledError:
                section_1_cancelled.set()
                raise
        else:
            section_2_started.set()
            try:
                await never_set.wait()
            except asyncio.CancelledError:
                section_2_cancelled.set()
                raise

        return {
            "section_facts": [],
            "section_data_points": [],
            "section_hypothesis_evidence": [],
            "section_charts": [],
            "section_insights": [],
            "section_contradictions": [],
            "section_sources": [],
            "missing_info": [],
        }

    state = {
        "sections": [
            {
                "id": "sec_1",
                "title": "市场概况",
                "description": "规模",
                "search_queries": ["q1"],
                "priority": 1,
            },
            {
                "id": "sec_2",
                "title": "竞争格局",
                "description": "品牌",
                "search_queries": ["q2"],
                "priority": 2,
            },
        ],
        "research_goal": "了解市场",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = mock_ainvoke

    try:
        with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
            pipeline_task = asyncio.create_task(
                section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})
            )
            await asyncio.wait_for(section_1_started.wait(), timeout=0.2)
            await asyncio.wait_for(section_2_started.wait(), timeout=0.2)
            pipeline_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pipeline_task

        await asyncio.wait_for(section_1_cancelled.wait(), timeout=0.2)
        await asyncio.wait_for(section_2_cancelled.wait(), timeout=0.2)
    finally:
        leftovers = [task for task in child_tasks if not task.done()]
        for task in leftovers:
            task.cancel()
        if leftovers:
            await asyncio.gather(*leftovers, return_exceptions=True)


async def test_section_pipeline_external_cancellation_drains_completed_child_exceptions() -> None:
    """External cancellation should also consume exceptions from already-finished child tasks."""
    from deep_agents.graph import section_pipeline_node

    real_wait = asyncio.wait
    real_gather = asyncio.gather
    failure_started = asyncio.Event()
    blocking_started = asyncio.Event()
    blocking_cancelled = asyncio.Event()
    release_blocker = asyncio.Event()
    child_tasks: set[asyncio.Task] = set()
    drained_tasks: set[asyncio.Task] = set()

    async def mock_ainvoke(inp, config=None):
        current = asyncio.current_task()
        if current is not None:
            child_tasks.add(current)

        if inp["section_id"] == "sec_1":
            failure_started.set()
            raise RuntimeError("child exploded")

        blocking_started.set()
        try:
            await release_blocker.wait()
        except asyncio.CancelledError:
            blocking_cancelled.set()
            raise

        return {
            "section_facts": [],
            "section_data_points": [],
            "section_hypothesis_evidence": [],
            "section_charts": [],
            "section_insights": [],
            "section_contradictions": [],
            "section_sources": [],
            "missing_info": [],
        }

    async def cancelling_wait(pending, *, return_when):
        done, still_pending = await real_wait(pending, return_when=return_when)
        await asyncio.wait_for(failure_started.wait(), timeout=0.2)
        await asyncio.wait_for(blocking_started.wait(), timeout=0.2)
        asyncio.current_task().cancel()
        await asyncio.sleep(0)
        return done, still_pending

    async def recording_gather(*aws, return_exceptions=False):
        for awaitable in aws:
            if isinstance(awaitable, asyncio.Task):
                drained_tasks.add(awaitable)
        return await real_gather(*aws, return_exceptions=return_exceptions)

    state = {
        "sections": [
            {
                "id": "sec_1",
                "title": "市场概况",
                "description": "规模",
                "search_queries": ["q1"],
                "priority": 1,
            },
            {
                "id": "sec_2",
                "title": "竞争格局",
                "description": "品牌",
                "search_queries": ["q2"],
                "priority": 2,
            },
        ],
        "research_goal": "了解市场",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = mock_ainvoke
    try:
        with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
            with patch("deep_agents.graph.asyncio.wait", new=cancelling_wait):
                with patch("deep_agents.graph.asyncio.gather", new=recording_gather):
                    pipeline_task = asyncio.create_task(
                        section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})
                    )
                    with pytest.raises(asyncio.CancelledError):
                        await pipeline_task

        await asyncio.wait_for(blocking_cancelled.wait(), timeout=0.2)
        assert child_tasks
        assert all(task.done() for task in child_tasks)
        assert drained_tasks == child_tasks
    finally:
        child_tasks.clear()
        gc.collect()
        await asyncio.sleep(0)


async def test_section_pipeline_fails_fast_on_unresolved_source_url_reference() -> None:
    """Dangling source_url references should fail fast during section merge."""
    from deep_agents.graph import section_pipeline_node

    section_result = {
        "section_facts": [{"content": "fact1", "source_url": "https://example.com/missing"}],
        "section_data_points": [],
        "section_hypothesis_evidence": [],
        "section_charts": [],
        "section_insights": [],
        "section_contradictions": [],
        "section_sources": [{"url": "https://example.com", "title": "example", "summary": ""}],
        "missing_info": [],
    }
    state = {
        "sections": [
            {"id": "sec_1", "title": "T", "description": "D", "search_queries": ["q"], "priority": 1}
        ],
        "research_goal": "研究",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = AsyncMock(return_value=section_result)

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        with pytest.raises(ValueError, match="Unresolved source_url"):
            await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})


@pytest.mark.parametrize(
    ("field_name", "record"),
    [
        ("section_facts", {"content": "fact-without-source"}),
        ("section_data_points", {"name": "dp-without-source", "value": 1}),
        (
            "section_hypothesis_evidence",
            {
                "hypothesis_statement": "h1",
                "evidence_type": "supports",
                "content": "evidence-without-source",
            },
        ),
    ],
)
async def test_section_pipeline_fails_fast_when_required_source_url_missing(
    field_name: str, record: dict
) -> None:
    """Facts/data/evidence records must include source_url and fail fast if missing."""
    from deep_agents.graph import section_pipeline_node

    section_result = {
        "section_facts": [],
        "section_data_points": [],
        "section_hypothesis_evidence": [],
        "section_charts": [],
        "section_insights": [],
        "section_contradictions": [],
        "section_sources": [{"url": "https://example.com", "title": "example", "summary": ""}],
        "missing_info": [],
    }
    section_result[field_name] = [record]
    state = {
        "sections": [
            {"id": "sec_1", "title": "T", "description": "D", "search_queries": ["q"], "priority": 1}
        ],
        "research_goal": "研究",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = AsyncMock(return_value=section_result)

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        with pytest.raises(ValueError, match="Missing required source_url"):
            await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})


async def test_section_pipeline_rejects_legacy_source_id_only_records() -> None:
    """Legacy source_id-only records are no longer accepted by active merge contract."""
    from deep_agents.graph import section_pipeline_node

    section_result = {
        "section_facts": [{"content": "legacy fact", "source_id": "src_000"}],
        "section_data_points": [],
        "section_hypothesis_evidence": [],
        "section_charts": [],
        "section_insights": [],
        "section_contradictions": [],
        "section_sources": [{"url": "https://example.com", "title": "example", "summary": ""}],
        "missing_info": [],
    }
    state = {
        "sections": [
            {"id": "sec_1", "title": "T", "description": "D", "search_queries": ["q"], "priority": 1}
        ],
        "research_goal": "研究",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = AsyncMock(return_value=section_result)

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        with pytest.raises(ValueError, match="Missing required source_url"):
            await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})


async def test_section_pipeline_trims_incidental_whitespace_in_source_url_refs() -> None:
    """Whitespace around URL references should be trimmed before source matching."""
    from deep_agents.graph import section_pipeline_node

    section_result = {
        "section_facts": [{"content": "fact", "source_url": " https://example.com/a "}],
        "section_data_points": [{"name": "dp", "value": 1, "source_url": " https://example.com/a "}],
        "section_hypothesis_evidence": [
            {
                "hypothesis_statement": "h1",
                "evidence_type": "supports",
                "content": "e",
                "source_url": " https://example.com/a ",
            }
        ],
        "section_charts": [],
        "section_insights": [],
        "section_contradictions": [
            {
                "claim_a": "a",
                "claim_b": "b",
                "source_url_a": " https://example.com/a ",
                "source_url_b": " https://example.com/a ",
            }
        ],
        "section_sources": [{"url": "https://example.com/a", "title": "a", "summary": ""}],
        "missing_info": [],
    }
    state = {
        "sections": [
            {"id": "sec_1", "title": "T", "description": "D", "search_queries": ["q"], "priority": 1}
        ],
        "research_goal": "研究",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = AsyncMock(return_value=section_result)

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        result = await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})

    assert result.update["facts"]["value"][0]["source_url"] == "https://example.com/a"
    assert result.update["data_points"]["value"][0]["source_url"] == "https://example.com/a"
    assert (
        result.update["hypothesis_evidence"]["value"][0]["source_url"]
        == "https://example.com/a"
    )
    assert (
        result.update["contradictions"]["value"][0]["source_url_a"]
        == "https://example.com/a"
    )
    assert (
        result.update["contradictions"]["value"][0]["source_url_b"]
        == "https://example.com/a"
    )


async def test_section_pipeline_drops_legacy_keys_from_hybrid_url_records() -> None:
    """Hybrid URL records with stale legacy keys should not leak legacy keys into merged output."""
    from deep_agents.graph import section_pipeline_node

    section_result = {
        "section_facts": [
            {"content": "fact", "source_url": "https://example.com/a", "source_id": "src_legacy"}
        ],
        "section_data_points": [
            {
                "name": "dp",
                "value": 1,
                "source_url": "https://example.com/a",
                "id": "dp_legacy",
                "source_id": "src_legacy",
            }
        ],
        "section_hypothesis_evidence": [
            {
                "hypothesis_statement": "h1",
                "evidence_type": "supports",
                "content": "e",
                "source_url": "https://example.com/a",
                "hypothesis_id": "h_legacy",
                "source_id": "src_legacy",
            }
        ],
        "section_charts": [],
        "section_insights": [],
        "section_contradictions": [
            {
                "claim_a": "a",
                "claim_b": "b",
                "source_url_a": "https://example.com/a",
                "source_url_b": "https://example.com/a",
                "source_id_a": "src_legacy_a",
                "source_id_b": "src_legacy_b",
            }
        ],
        "section_sources": [
            {
                "url": "https://example.com/a",
                "title": "a",
                "summary": "",
                "source_id": "src_legacy",
            }
        ],
        "missing_info": [],
    }
    state = {
        "sections": [
            {"id": "sec_1", "title": "T", "description": "D", "search_queries": ["q"], "priority": 1}
        ],
        "research_goal": "研究",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = AsyncMock(return_value=section_result)

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        result = await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})

    merged_fact = result.update["facts"]["value"][0]
    merged_data_point = result.update["data_points"]["value"][0]
    merged_evidence = result.update["hypothesis_evidence"]["value"][0]
    merged_contradiction = result.update["contradictions"]["value"][0]
    merged_source = result.update["sources"]["value"][0]

    assert "source_id" not in merged_fact
    assert "source_id" not in merged_data_point
    assert "id" not in merged_data_point
    assert "source_id" not in merged_evidence
    assert "hypothesis_id" not in merged_evidence
    assert "source_id_a" not in merged_contradiction
    assert "source_id_b" not in merged_contradiction
    assert "source_id" not in merged_source


async def test_section_pipeline_fails_fast_on_unresolved_contradiction_source_urls() -> None:
    """Contradiction source_url_a/source_url_b must resolve from section sources."""
    from deep_agents.graph import section_pipeline_node

    section_result = {
        "section_facts": [],
        "section_data_points": [],
        "section_hypothesis_evidence": [],
        "section_charts": [],
        "section_insights": [],
        "section_contradictions": [
            {
                "claim_a": "a",
                "claim_b": "b",
                "source_url_a": "https://example.com/missing-a",
                "source_url_b": "https://example.com/missing-b",
            }
        ],
        "section_sources": [{"url": "https://example.com", "title": "example", "summary": ""}],
        "missing_info": [],
    }
    state = {
        "sections": [
            {"id": "sec_1", "title": "T", "description": "D", "search_queries": ["q"], "priority": 1}
        ],
        "research_goal": "研究",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = AsyncMock(return_value=section_result)

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        with pytest.raises(ValueError, match="Unresolved source_url"):
            await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})


async def test_section_pipeline_routes_to_outline_reviser_when_refuted() -> None:
    """section_pipeline routes to outline_reviser when ≥2 hypotheses are refuted and count < 1."""
    from deep_agents.graph import section_pipeline_node

    refuted_evidence = [
        {
            "evidence_type": "refutes",
            "hypothesis_statement": "h1",
            "content": "e1",
            "source_url": "https://example.com/1",
        },
        {
            "evidence_type": "refutes",
            "hypothesis_statement": "h2",
            "content": "e2",
            "source_url": "https://example.com/2",
        },
    ]
    section_result = {
        "section_facts": [],
        "section_data_points": [],
        "section_hypothesis_evidence": refuted_evidence,
        "section_charts": [],
        "section_insights": [],
        "section_contradictions": [],
        "section_sources": [
            {"url": "https://example.com/1", "title": "one", "summary": ""},
            {"url": "https://example.com/2", "title": "two", "summary": ""},
        ],
        "missing_info": [],
    }

    state = {
        "sections": [
            {"id": "sec_1", "title": "T", "description": "D", "search_queries": ["q"], "priority": 1}
        ],
        "research_goal": "研究",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 0,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = AsyncMock(return_value=section_result)

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        result = await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})

    assert result.goto == "outline_reviser"


async def test_section_pipeline_does_not_route_to_outline_reviser_when_count_at_1() -> None:
    """After one outline revision (count=1), re-outline is blocked even with refuted hypotheses."""
    from deep_agents.graph import section_pipeline_node

    refuted_evidence = [
        {
            "evidence_type": "refutes",
            "hypothesis_statement": "h1",
            "content": "e1",
            "source_url": "https://example.com/1",
        },
        {
            "evidence_type": "refutes",
            "hypothesis_statement": "h2",
            "content": "e2",
            "source_url": "https://example.com/2",
        },
    ]
    section_result = {
        "section_facts": [],
        "section_data_points": [],
        "section_hypothesis_evidence": refuted_evidence,
        "section_charts": [],
        "section_insights": [],
        "section_contradictions": [],
        "section_sources": [
            {"url": "https://example.com/1", "title": "one", "summary": ""},
            {"url": "https://example.com/2", "title": "two", "summary": ""},
        ],
        "missing_info": [],
    }

    state = {
        "sections": [
            {"id": "sec_1", "title": "T", "description": "D", "search_queries": ["q"], "priority": 1}
        ],
        "research_goal": "研究",
        "hypotheses": [],
        "language": "zh",
        "outline_revision_count": 1,
        "hypothesis_evidence": [],
    }

    mock_sg = MagicMock()
    mock_sg.ainvoke = AsyncMock(return_value=section_result)

    with patch("deep_agents.graph._get_section_subgraph", return_value=mock_sg):
        result = await section_pipeline_node(state, {"configurable": {"thread_id": "t1"}})

    assert result.goto == "lead_writer"
