"""Planner node — generates the initial research plan (ArchitectPlan).

Reads research_goal, confirmed_constraints, open_dimensions, and language from state,
calls the LLM with structured output to produce an ArchitectPlan, then returns the
relevant state fields.
"""

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import planner_prompt
from deep_agents.schemas import ArchitectPlan
from deep_agents.state import ResearchState
from deep_agents.utils import get_api_key_for_model, get_today_str


async def planner_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Generate the initial research plan from the research goal and constraints."""
    configurable = Configuration.from_runnable_config(config)

    model = (
        init_chat_model(
            model=configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            base_url=configurable.openai_compatible_base_url,
        )
        .with_structured_output(ArchitectPlan)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    research_goal = state.get("research_goal", "")
    confirmed_constraints = state.get("confirmed_constraints", [])
    open_dimensions = state.get("open_dimensions", [])
    language = state.get("language", "zh")

    prompt_text = planner_prompt.format(
        date=get_today_str(),
        research_goal=research_goal,
        confirmed_constraints=confirmed_constraints,
        open_dimensions=open_dimensions,
        language=language,
    )

    plan: ArchitectPlan = await model.ainvoke([HumanMessage(content=prompt_text)])

    return {
        "research_type": plan.research_type,
        "hypotheses": [h.model_dump() for h in plan.hypotheses],
        "sections": [s.model_dump() for s in plan.sections],
        "budget": plan.budget,
        "outline_status": plan.outline_status,
        "outline_revision_count": 0,
    }
