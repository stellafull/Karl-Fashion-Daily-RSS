# src/deep_agents/graph.py
"""LangGraph builder: main research graph + section subgraph + conditional edge functions."""

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from deep_agents.state import ResearchState, SectionState

# Agent node imports
from deep_agents.agents.clarify import clarify_node
from deep_agents.agents.planner import planner_node
from deep_agents.agents.outline_reviser import outline_reviser_node
from deep_agents.agents.deep_scout import deep_scout_node
from deep_agents.agents.analyst import analyst_node
from deep_agents.agents.data_wiz import data_wiz_node
from deep_agents.agents.writer import writer_node
from deep_agents.agents.synthesizer import synthesizer_node
from deep_agents.agents.trend_triangulator import trend_triangulator_node
from deep_agents.agents.reviewer import reviewer_node
from deep_agents.agents.reviser import reviser_node
from deep_agents.agents.final_check import final_check_node


# ── Conditional edge functions ──────────────────────────────────────────────

def route_after_clarify(state: ResearchState) -> str:
    if state.get("need_clarification"):
        return END
    return "planner"


def fan_out_sections(state: ResearchState) -> list:
    return [
        Send("section_pipeline", {
            "section_id": s["id"],
            "section_title": s["title"],
            "section_description": s["description"],
            "search_queries": s["search_queries"],
            "research_goal": state["research_goal"],
            "hypotheses": state["hypotheses"],
            "budget": state["budget"],
            "language": state.get("language", "zh"),
            # Initialize empty output fields
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
        })
        for s in state.get("sections", [])
    ]


def route_after_collection(state: ResearchState) -> str:
    refuted_count = sum(
        1 for h in state.get("hypothesis_evidence", [])
        if h.get("evidence_type") == "refutes"
    )
    if refuted_count >= 2 and state.get("outline_revision_count", 0) < 1:
        return "outline_reviser"
    return "lead_writer"


def route_after_synthesis(state: ResearchState) -> str:
    if state.get("research_type") == "trend_analysis":
        return "trend_triangulator"
    return "reviewer"


def route_after_review(state: ResearchState) -> str:
    review = state.get("review_result") or {}
    if review.get("verdict") != "pass" and state.get("revision_count", 0) < 2:
        return "reviser"
    return "final_check"


# ── Section pipeline node (wraps section subgraph) ──────────────────────────

_section_subgraph = None


def _get_section_subgraph():
    global _section_subgraph
    if _section_subgraph is None:
        _section_subgraph = build_section_subgraph()
    return _section_subgraph


async def section_pipeline_node(state: dict) -> dict:
    """Run deep_scout → analyst → data_wiz for one section; merge outputs to ResearchState fields."""
    sg = _get_section_subgraph()
    result = await sg.ainvoke(state)
    return {
        "facts": result.get("section_facts", []),
        "data_points": result.get("section_data_points", []),
        "hypothesis_evidence": result.get("section_hypothesis_evidence", []),
        "charts": result.get("section_charts", []),
        "insights": [
            {"section_id": state["section_id"], "insight": i}
            for i in result.get("section_insights", [])
        ],
        "contradictions": result.get("section_contradictions", []),
        "sources": result.get("section_sources", []),
        "open_questions": [
            {"section_id": state["section_id"], "question": q}
            for q in result.get("missing_info", [])
        ],
    }


# ── Graph builders ────────────────────────────────────────────────────────────

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

    # Nodes
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

    # Edges
    graph.add_edge(START, "clarify")
    graph.add_conditional_edges("clarify", route_after_clarify)
    graph.add_edge("planner", "outline_reviser")
    graph.add_conditional_edges("outline_reviser", fan_out_sections)
    graph.add_conditional_edges("section_pipeline", route_after_collection)
    graph.add_edge("lead_writer", "synthesizer")
    graph.add_conditional_edges("synthesizer", route_after_synthesis)
    graph.add_edge("trend_triangulator", "reviewer")
    graph.add_conditional_edges("reviewer", route_after_review)
    graph.add_edge("reviser", "reviewer")
    graph.add_edge("final_check", END)

    return graph.compile(checkpointer=MemorySaver())
