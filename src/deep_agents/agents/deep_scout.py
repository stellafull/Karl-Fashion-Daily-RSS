"""DeepScout node — explicit tool-calling loop for section-level evidence collection.

Runs a researcher -> researcher_tools loop: the LLM calls search tools and reflects
until it calls ResearchComplete or hits max_deep_scout_iterations.

After each tavily_search call the raw results are compressed (like open_deep_research's
compress_research) before being fed back to the LLM and stored for downstream nodes.
"""
import json
import logging

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.agents.compress_search import compress_search
from deep_agents.configuration import Configuration
from deep_agents.prompts import deep_scout_prompt
from deep_agents.state import SectionState
from deep_agents.utils import (
    _strip_ctrl,
    get_all_tools,
    get_api_key_for_model,
    get_today_str,
)

logger = logging.getLogger(__name__)

_RESEARCH_COMPLETE_TOOL_NAME = "ResearchComplete"
_THINK_TOOL_NAME = "think_tool"


async def deep_scout_node(state: SectionState, config: RunnableConfig) -> dict:
    """Run an explicit researcher loop to collect evidence for a single research section."""
    configurable = Configuration.from_runnable_config(config)
    tools = await get_all_tools(config)

    callable_tools = [
        t for t in tools if hasattr(t, "name") and callable(getattr(t, "ainvoke", None))
    ]
    tool_map = {t.name: t for t in callable_tools}

    model = init_chat_model(
        model=configurable.research_model,
        max_tokens=configurable.research_model_max_tokens,
        api_key=get_api_key_for_model(configurable.research_model, config),
        base_url=configurable.openai_compatible_base_url,
        max_retries=configurable.provider_max_retries,
    )
    bound_model = model.bind_tools(callable_tools)

    research_goal = state.get("research_goal", "")
    section_title = state.get("section_title", "")
    section_description = state.get("section_description", "")
    hypotheses = state.get("hypotheses", [])

    system_prompt = deep_scout_prompt.format(
        date=get_today_str(),
        research_goal=research_goal,
        section_title=section_title,
        section_description=section_description,
        search_queries="\n".join(state.get("search_queries", [])),
        hypotheses=json.dumps(hypotheses, ensure_ascii=False),
    )
    system_prompt = _strip_ctrl(system_prompt)

    messages: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="请开始研究当前章节，收集足够的证据。"),
    ]

    search_results: list[str] = []

    for _ in range(configurable.max_deep_scout_iterations):
        response: AIMessage = await bound_model.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        done = any(
            tc["name"] == _RESEARCH_COMPLETE_TOOL_NAME
            for tc in response.tool_calls
        )
        tool_messages = []
        for tc in response.tool_calls:
            if tc["name"] == _RESEARCH_COMPLETE_TOOL_NAME:
                continue

            tool = tool_map.get(tc["name"])
            if tool is None:
                tool_messages.append(
                    ToolMessage(
                        content=f"Unknown tool: {tc['name']}",
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    )
                )
                continue

            try:
                raw_result = await tool.ainvoke(tc, config=config)

                # tool.ainvoke with a ToolCall dict returns a ToolMessage,
                # not the raw string.  Extract .content so downstream
                # processing and json.dumps don't choke.
                if isinstance(raw_result, ToolMessage):
                    content = raw_result.content
                elif isinstance(raw_result, str):
                    content = raw_result
                else:
                    content = json.dumps(raw_result, ensure_ascii=False)

                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)

                # Compress search/image results before feeding to LLM.
                # think_tool reflections are internal — skip compression.
                if tc["name"] != _THINK_TOOL_NAME and content:
                    try:
                        compressed = await compress_search(
                            raw_content=content,
                            research_goal=research_goal,
                            section_title=section_title,
                            section_description=section_description,
                            hypotheses=hypotheses,
                            config=config,
                        )
                        search_results.append(compressed)
                        content = compressed  # LLM sees compressed version
                    except Exception as compress_exc:
                        logger.warning(
                            "Compression failed for %s, using raw content: %s",
                            tc["name"],
                            compress_exc,
                        )
                        search_results.append(content)

                tool_messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    )
                )
            except Exception as tool_exc:
                logger.warning("Tool %s failed: %s", tc["name"], tool_exc)
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool error: {tool_exc}",
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    )
                )

        messages.extend(tool_messages)

        if done:
            break

    return {"search_results": search_results}
