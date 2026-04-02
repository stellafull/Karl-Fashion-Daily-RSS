"""DeepScout node — explicit tool-calling loop for section-level evidence collection.

Runs a researcher -> researcher_tools loop: the LLM calls search tools and reflects
until it calls ResearchComplete or hits max_react_tool_calls.
"""
import json
import logging
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import deep_scout_prompt
from deep_agents.state import SectionState
from deep_agents.utils import get_all_tools, get_api_key_for_model, get_today_str

logger = logging.getLogger(__name__)

_RESEARCH_COMPLETE_TOOL_NAME = "ResearchComplete"


def _extract_results(tool_result_content: str, sources: list, seen_urls: set) -> dict:
    """Parse a tool result string, extract URL sources, return search_result entry."""
    search_result = {"raw": tool_result_content[:3000]}
    try:
        data = json.loads(tool_result_content)
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                url = item.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append(
                        {
                            "source_id": f"src_{len(sources):03d}",
                            "url": url,
                            "title": item.get("title", ""),
                            "credibility_score": 0.7,
                        }
                    )
    except Exception:
        pass
    return search_result


async def deep_scout_node(state: SectionState, config: RunnableConfig) -> dict:
    """Run an explicit researcher loop to collect evidence for a single research section."""
    configurable = Configuration.from_runnable_config(config)

    try:
        tools = await get_all_tools(config)
    except Exception as exc:
        logger.warning("deep_scout_node: get_all_tools failed: %s", exc)
        tools = []

    callable_tools = [
        t for t in tools if hasattr(t, "name") and callable(getattr(t, "ainvoke", None))
    ]
    tool_map = {t.name: t for t in callable_tools}

    model = init_chat_model(
        model=configurable.research_model,
        max_tokens=configurable.research_model_max_tokens,
        api_key=get_api_key_for_model(configurable.research_model, config),
        base_url=configurable.openai_compatible_base_url,
    )
    bound_model = model.bind_tools(callable_tools)

    system_prompt = deep_scout_prompt.format(
        date=get_today_str(),
        research_goal=state.get("research_goal", ""),
        section_title=state.get("section_title", ""),
        section_description=state.get("section_description", ""),
        search_queries="\n".join(state.get("search_queries", [])),
        hypotheses=json.dumps(state.get("hypotheses", []), ensure_ascii=False),
    )

    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="请开始研究当前章节，收集足够的证据。"),
    ]

    search_results = []
    sources: list[dict] = []
    seen_urls: set[str] = set()

    try:
        for _ in range(configurable.max_react_tool_calls):
            response: AIMessage = await bound_model.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            done = False
            tool_messages = []
            for tc in response.tool_calls:
                if tc["name"] == _RESEARCH_COMPLETE_TOOL_NAME:
                    done = True
                    break

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
                    content = (
                        raw_result
                        if isinstance(raw_result, str)
                        else json.dumps(raw_result, ensure_ascii=False)
                    )
                    search_results.append(_extract_results(content, sources, seen_urls))
                    tool_messages.append(
                        ToolMessage(
                            content=content[:3000],
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

    except Exception as exc:
        logger.warning("deep_scout_node failed: %s", exc)
        return {"search_results": [], "section_sources": []}

    return {
        "search_results": search_results,
        "section_sources": sources,
    }
