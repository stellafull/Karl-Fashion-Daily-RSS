"""DataWiz node — quantitative data extraction from section search results.

Reads search_results, section_title, and research_goal from SectionState,
calls the LLM with structured output to produce a DataWizOutput, then returns
the relevant state fields as plain dicts/lists.
"""

import json

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import data_wiz_prompt
from deep_agents.schemas import DataWizOutput
from deep_agents.state import SectionState
from deep_agents.utils import get_api_key_for_model


async def data_wiz_node(state: SectionState, config: RunnableConfig) -> dict:
    """Extract quantitative data points and chart configurations from search results."""
    configurable = Configuration.from_runnable_config(config)

    model = (
        init_chat_model(
            model=configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            base_url=configurable.openai_compatible_base_url,
        )
        .with_structured_output(DataWizOutput)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    research_goal = state.get("research_goal", "")
    section_title = state.get("section_title", "")
    search_results = state.get("search_results", [])

    prompt_text = data_wiz_prompt.format(
        research_goal=research_goal,
        section_title=section_title,
        search_results=json.dumps(search_results, ensure_ascii=False),
    )

    output: DataWizOutput = await model.ainvoke([HumanMessage(content=prompt_text)])

    return {
        "section_data_points": output.section_data_points,
        "section_charts": output.section_charts,
        "section_time_series": output.section_time_series,
    }
