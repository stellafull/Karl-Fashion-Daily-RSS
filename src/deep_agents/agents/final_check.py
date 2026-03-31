# src/deep_agents/agents/final_check.py
"""Final check: verify fixes, detect new issues, assign publication_readiness."""

import json
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import final_check_prompt
from deep_agents.schemas import FinalResult
from deep_agents.state import ResearchState
from deep_agents.utils import get_api_key_for_model


async def final_check_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Final quality gate: verify revisions, assign publication readiness."""
    configurable = Configuration.from_runnable_config(config)
    model = (
        init_chat_model(
            model=configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            base_url=configurable.openai_compatible_base_url,
        )
        .with_structured_output(FinalResult)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    prompt_text = final_check_prompt.format(
        research_goal=state["research_goal"],
        review_result=json.dumps(state.get("review_result", {}), ensure_ascii=False, indent=2),
        full_report=state.get("full_report", ""),
        revision_count=state.get("revision_count", 0),
    )

    result: FinalResult = await model.ainvoke([HumanMessage(content=prompt_text)])
    return {"final_result": result.model_dump()}
