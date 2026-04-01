# src/deep_agents/api.py
"""FastAPI app: single POST /research endpoint with SSE streaming."""

import json
import logging
from typing import Any, List, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from deep_agents.graph import build_research_graph

logger = logging.getLogger(__name__)

app = FastAPI(title="Fashion Deep Research API", version="1.0.0")

_graph = None

NODE_NAMES = {
    "clarify", "planner", "outline_reviser", "section_pipeline",
    "lead_writer", "synthesizer", "trend_triangulator",
    "reviewer", "reviser", "final_check",
}


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_research_graph()
    return _graph


class ResearchRequest(BaseModel):
    messages: List[Any]
    object_context: Optional[str] = None
    thread_id: str


@app.post("/research")
async def research(request: ResearchRequest):
    """Start a research job. Returns SSE stream of typed events."""

    async def event_stream():
        graph = get_graph()
        config = {"configurable": {"thread_id": request.thread_id}}
        input_state = {
            "messages": request.messages,
            "object_context": request.object_context,
        }

        final_state_output = {}

        try:
            async for event in graph.astream_events(input_state, config=config, version="v2"):
                kind = event.get("event")
                name = event.get("name", "")
                data = event.get("data", {})

                if kind == "on_chain_end" and name in NODE_NAMES:
                    output = data.get("output") or {}

                    # Progress event for every completed node
                    yield f"data: {json.dumps({'type': 'progress', 'node': name, 'status': 'done'})}\n\n"

                    # Section done event
                    if name == "section_pipeline":
                        section_id = output.get("section_id", "")
                        yield f"data: {json.dumps({'type': 'section_done', 'section_id': section_id})}\n\n"

                    # Clarification event
                    if name == "clarify" and output.get("need_clarification"):
                        yield f"data: {json.dumps({'type': 'clarification', 'question': output.get('clarification_question', '')})}\n\n"

                    # Track final_check output for report event
                    if name == "final_check":
                        final_state_output = output

        except Exception as e:
            logger.error(f"Research pipeline error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        # Emit report event after stream completes
        if final_state_output:
            report = final_state_output.get("full_report", "")
            yield f"data: {json.dumps({'type': 'report', 'content': report})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}
