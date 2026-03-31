"""Analyst node — qualitative analysis of search results for a section.

Reads search_results, hypotheses, section_title, section_description, and research_goal
from SectionState, calls the LLM with structured output to produce an AnalystOutput, then
returns the relevant state fields as plain dicts/lists.
"""

import json

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import analyst_prompt
from deep_agents.schemas import AnalystOutput
from deep_agents.state import SectionState
from deep_agents.utils import get_api_key_for_model


async def analyst_node(state: SectionState, config: RunnableConfig) -> dict:
    """Perform qualitative analysis of collected search results for the section."""
    configurable = Configuration.from_runnable_config(config)

    model = (
        init_chat_model(
            model=configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            base_url=configurable.openai_compatible_base_url,
        )
        .with_structured_output(AnalystOutput)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    research_goal = state.get("research_goal", "")
    section_title = state.get("section_title", "")
    section_description = state.get("section_description", "")
    hypotheses = state.get("hypotheses", [])
    search_results = state.get("search_results", [])

    prompt_text = analyst_prompt.format(
        research_goal=research_goal,
        section_title=section_title,
        section_description=section_description,
        hypotheses=json.dumps(hypotheses, ensure_ascii=False),
        search_results=json.dumps(search_results, ensure_ascii=False),
    )

    output: AnalystOutput = await model.ainvoke([HumanMessage(content=prompt_text)])

    return {
        "section_facts": output.section_facts,
        "section_insights": output.section_insights,
        "section_hypothesis_evidence": output.section_hypothesis_evidence,
        "section_contradictions": output.section_contradictions,
        "section_entities": output.section_entities,
        "missing_info": output.missing_info,
    }
