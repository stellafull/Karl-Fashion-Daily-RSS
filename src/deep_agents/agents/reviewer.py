# src/deep_agents/agents/reviewer.py
"""Reviewer: strict quality review → ReviewResult with quality_score and issues."""

import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import reviewer_prompt
from deep_agents.schemas import ReviewResult
from deep_agents.state import ResearchState
from deep_agents.utils import get_api_key_for_model


async def reviewer_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Review the full report: score quality and identify issues."""
    configurable = Configuration.from_runnable_config(config)
    model = (
        init_chat_model(
            model=configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            base_url=configurable.openai_compatible_base_url,
        )
        .with_structured_output(ReviewResult)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    prompt_text = reviewer_prompt.format(
        research_goal=state["research_goal"],
        sections=json.dumps(
            [{"id": s["id"], "title": s["title"]} for s in state.get("sections", [])],
            ensure_ascii=False,
        ),
        full_report=state.get("full_report", ""),
        facts=json.dumps(state.get("facts", [])[:30], ensure_ascii=False),
        data_points=json.dumps(state.get("data_points", [])[:20], ensure_ascii=False),
    )

    result: ReviewResult = await model.ainvoke([HumanMessage(content=prompt_text)])
    return {"review_result": result.model_dump()}
