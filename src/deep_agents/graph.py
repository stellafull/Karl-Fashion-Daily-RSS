"""LangGraph builder: main research graph + section subgraph."""
import asyncio
import logging
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from deep_agents.state import ResearchState, SectionState

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

logger = logging.getLogger(__name__)

_section_subgraph = None


def _get_section_subgraph():
    global _section_subgraph
    if _section_subgraph is None:
        _section_subgraph = build_section_subgraph()
    return _section_subgraph


async def section_pipeline_node(
    state: ResearchState, config: RunnableConfig
) -> Command[Literal["lead_writer", "outline_reviser"]]:
    """Fan out to all section subgraphs concurrently; merge results; route via Command."""
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
        for s in state.get("sections", [])
    ]

    raw = await asyncio.gather(
        *[sg.ainvoke(inp, config) for inp in section_inputs],
        return_exceptions=True,
    )

    merged: dict = {
        "facts": [],
        "data_points": [],
        "hypothesis_evidence": [],
        "charts": [],
        "insights": [],
        "contradictions": [],
        "sources": [],
        "open_questions": [],
        "failed_sections": [],
    }

    for s, r in zip(state.get("sections", []), raw):
        if isinstance(r, Exception):
            logger.warning("Section %s failed: %s", s["id"], r)
            merged["failed_sections"].append(s["id"])
            continue
        merged["facts"].extend(r.get("section_facts", []))
        merged["data_points"].extend(r.get("section_data_points", []))
        merged["hypothesis_evidence"].extend(r.get("section_hypothesis_evidence", []))
        merged["charts"].extend(r.get("section_charts", []))
        merged["insights"].extend(
            {"section_id": s["id"], "insight": i}
            for i in r.get("section_insights", [])
        )
        merged["contradictions"].extend(r.get("section_contradictions", []))
        merged["sources"].extend(r.get("section_sources", []))
        merged["open_questions"].extend(
            {"section_id": s["id"], "question": q} for q in r.get("missing_info", [])
        )

    refuted = sum(
        1 for h in merged["hypothesis_evidence"] if h.get("evidence_type") == "refutes"
    )
    next_node = (
        "outline_reviser"
        if refuted >= 2 and state.get("outline_revision_count", 0) < 1
        else "lead_writer"
    )

    return Command(goto=next_node, update=merged)


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
    graph = StateGraph(ResearchState)

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
