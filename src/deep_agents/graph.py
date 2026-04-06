"""LangGraph builder: main research graph + section subgraph."""
import asyncio
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from deep_agents.state import ResearchInputState, ResearchState, SectionState

from deep_agents.agents.clarify import clarify_node
from deep_agents.agents.planner import planner_node
from deep_agents.agents.outline_reviser import outline_reviser_node
from deep_agents.agents.analyst import analyst_node
from deep_agents.agents.data_wiz import data_wiz_node
from deep_agents.agents.deep_scout import deep_scout_node
from deep_agents.agents.writer import writer_node
from deep_agents.agents.synthesizer import synthesizer_node
from deep_agents.agents.trend_triangulator import trend_triangulator_node
from deep_agents.agents.reviewer import reviewer_node
from deep_agents.agents.reviser import reviser_node
from deep_agents.agents.final_check import final_check_node

_section_subgraph = None


def _get_section_subgraph():
    global _section_subgraph
    if _section_subgraph is None:
        _section_subgraph = build_section_subgraph()
    return _section_subgraph


def _tag_section_records(records: list[dict], section_id: str) -> list[dict]:
    """Attach section_id to every section-level record before global merge."""
    tagged: list[dict] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        tagged.append({**record, "section_id": section_id})
    return tagged


async def section_pipeline_node(
    state: ResearchState, config: RunnableConfig
) -> Command[Literal["lead_writer", "outline_reviser"]]:
    """Fan out to all section subgraphs concurrently; merge results; route via Command."""
    sections = state.get("sections", [])
    if not sections:
        raise ValueError("Planner produced no sections")

    sg = _get_section_subgraph()

    section_inputs = [
        {
            "section_id": s["id"],
            "section_title": s["title"],
            "section_description": s["description"],
            "search_queries": s["search_queries"],
            "research_goal": state["research_goal"],
            "hypotheses": state.get("hypotheses", []),
            "language": state.get("language", "zh"),
            "search_results": [],
            "section_facts": [],
            "section_insights": [],
            "section_hypothesis_evidence": [],
            "section_contradictions": [],
            "section_entities": [],
            "missing_info": [],
            "section_data_points": [],
            "section_charts": [],
            "section_time_series": [],
            "section_sources": [],
        }
        for s in sections
    ]

    async def _run_section(index: int, section_input: dict) -> tuple[int, dict]:
        return index, await sg.ainvoke(section_input, config)

    tasks = [
        asyncio.create_task(_run_section(index, section_input))
        for index, section_input in enumerate(section_inputs)
    ]
    raw: list[dict | None] = [None] * len(section_inputs)
    pending = set(tasks)
    try:
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_EXCEPTION,
            )

            first_error: Exception | None = None
            for task in done:
                exc = task.exception()
                if exc is not None:
                    if first_error is None:
                        first_error = exc
                    continue

                index, section_result = task.result()
                raw[index] = section_result

            if first_error is not None:
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                raise first_error
    except asyncio.CancelledError as cancel_exc:
        for task in tasks:
            if task.done():
                continue
            task.cancel()
        if tasks:
            current_task = asyncio.current_task()
            if current_task is not None and hasattr(current_task, "uncancel"):
                current_task.uncancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        raise cancel_exc

    if any(result is None for result in raw):
        raise RuntimeError("Section subgraph returned no result")

    merged: dict = {
        "facts": [],
        "data_points": [],
        "hypothesis_evidence": [],
        "charts": [],
        "insights": [],
        "contradictions": [],
        "sources": [],
        "open_questions": [],
    }

    for s, r in zip(sections, raw):
        if r is None:
            raise RuntimeError("Section subgraph returned no result")
        section_id = s["id"]
        source_urls: set[str] = set()
        legacy_keys = {"source_id", "source_id_a", "source_id_b", "hypothesis_id", "id"}
        section_sources = r.get("section_sources", [])
        for src in section_sources:
            if not isinstance(src, dict):
                continue
            source_url = src.get("url")
            if not isinstance(source_url, str) or not source_url.strip():
                raise ValueError(f"Missing required url in section_sources for section '{section_id}'")
            source_url = source_url.strip()
            source_urls.add(source_url)
            cleaned_src = {
                key: value for key, value in src.items() if key not in legacy_keys
            }
            merged["sources"].append(
                {
                    **cleaned_src,
                    "url": source_url,
                    "section_id": section_id,
                }
            )

        def _map_source_refs(record: dict, required_url_fields: tuple[str, ...] = ()) -> dict:
            mapped = {**record, "section_id": section_id}

            for field_name in required_url_fields:
                source_url = mapped.get(field_name)
                if not isinstance(source_url, str) or not source_url.strip():
                    raise ValueError(
                        f"Missing required {field_name} in section '{section_id}'"
                    )
                source_url = source_url.strip()
                if source_url not in source_urls:
                    raise ValueError(
                        f"Unresolved {field_name} '{source_url}' in section '{section_id}'"
                    )
                mapped[field_name] = source_url

            for optional_field in ("source_url", "source_url_a", "source_url_b"):
                if optional_field in required_url_fields:
                    continue
                source_url = mapped.get(optional_field)
                if source_url is None:
                    continue
                if not isinstance(source_url, str) or not source_url.strip():
                    raise ValueError(
                        f"Invalid {optional_field} in section '{section_id}'"
                    )
                source_url = source_url.strip()
                if source_url not in source_urls:
                    raise ValueError(
                        f"Unresolved {optional_field} '{source_url}' in section '{section_id}'"
                    )
                mapped[optional_field] = source_url

            return {
                key: value for key, value in mapped.items() if key not in legacy_keys
            }

        merged["facts"].extend(
            _map_source_refs(item, required_url_fields=("source_url",))
            for item in r.get("section_facts", [])
            if isinstance(item, dict)
        )
        merged["data_points"].extend(
            _map_source_refs(item, required_url_fields=("source_url",))
            for item in r.get("section_data_points", [])
            if isinstance(item, dict)
        )
        merged["hypothesis_evidence"].extend(
            _map_source_refs(item, required_url_fields=("source_url",))
            for item in r.get("section_hypothesis_evidence", [])
            if isinstance(item, dict)
        )
        merged["charts"].extend(_tag_section_records(r.get("section_charts", []), section_id))
        merged["insights"].extend(
            {"section_id": section_id, "insight": i}
            for i in r.get("section_insights", [])
        )
        merged["contradictions"].extend(
            _map_source_refs(item, required_url_fields=("source_url_a", "source_url_b"))
            for item in r.get("section_contradictions", [])
            if isinstance(item, dict)
        )
        merged["open_questions"].extend(
            {"section_id": section_id, "question": q} for q in r.get("missing_info", [])
        )

    refuted = sum(
        1 for h in merged["hypothesis_evidence"] if h.get("evidence_type") == "refutes"
    )
    next_node = (
        "outline_reviser"
        if refuted >= 2 and state.get("outline_revision_count", 0) < 1
        else "lead_writer"
    )

    return Command(
        goto=next_node,
        update={
            "facts": {"type": "override", "value": merged["facts"]},
            "data_points": {"type": "override", "value": merged["data_points"]},
            "hypothesis_evidence": {"type": "override", "value": merged["hypothesis_evidence"]},
            "charts": {"type": "override", "value": merged["charts"]},
            "insights": {"type": "override", "value": merged["insights"]},
            "contradictions": {"type": "override", "value": merged["contradictions"]},
            "sources": {"type": "override", "value": merged["sources"]},
            "open_questions": {"type": "override", "value": merged["open_questions"]},
        },
    )


def build_section_subgraph():
    graph = StateGraph(SectionState)
    graph.add_node("deep_scout", deep_scout_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("data_wiz", data_wiz_node)
    graph.add_edge(START, "deep_scout")
    graph.add_edge("deep_scout", "analyst")
    graph.add_edge("analyst", "data_wiz")
    graph.add_edge("data_wiz", END)
    return graph.compile()


def build_research_graph():
    graph = StateGraph(ResearchState, input_schema=ResearchInputState)

    graph.add_node("clarify", clarify_node)
    graph.add_node("planner", planner_node)
    graph.add_node("outline_reviser", outline_reviser_node)
    graph.add_node("section_pipeline", section_pipeline_node)
    graph.add_node("lead_writer", writer_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("trend_triangulator", trend_triangulator_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("reviser", reviser_node)
    graph.add_node("final_check", final_check_node)

    graph.add_edge(START, "clarify")
    graph.add_edge("planner", "section_pipeline")
    graph.add_edge("outline_reviser", "section_pipeline")
    graph.add_edge("lead_writer", "synthesizer")
    graph.add_edge("trend_triangulator", "reviewer")
    graph.add_edge("reviser", "reviewer")
    graph.add_edge("final_check", END)

    return graph.compile(checkpointer=MemorySaver())
