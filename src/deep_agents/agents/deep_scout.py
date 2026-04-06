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
from deep_agents.utils import (
    _strip_ctrl,
    get_all_tools,
    get_api_key_for_model,
    get_today_str,
)

logger = logging.getLogger(__name__)

_RESEARCH_COMPLETE_TOOL_NAME = "ResearchComplete"


def _serialize_tool_result(raw_result: Any) -> str:
    """Convert a tool result to a bounded string for ToolMessage content."""
    if isinstance(raw_result, str):
        return raw_result[:3000]
    return json.dumps(raw_result, ensure_ascii=False)[:3000]


def _extract_results(
    raw_tool_result: Any,
    sources: list[dict[str, Any]],
    seen_urls: set[str],
) -> dict[str, Any]:
    """Parse tool results and emit canonical source metadata."""
    preview = _serialize_tool_result(raw_tool_result)
    search_result: dict[str, Any] = {"raw": preview, "sources": []}

    data: Any = raw_tool_result
    if isinstance(raw_tool_result, str):
        try:
            data = json.loads(raw_tool_result)
        except Exception:
            return search_result

    candidates: list[dict[str, Any]] = []
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        candidates = [item for item in data["results"] if isinstance(item, dict)]
    elif isinstance(data, list):
        candidates = [item for item in data if isinstance(item, dict)]
    else:
        return search_result

    for item in candidates:
        raw_url = item.get("url")
        if not isinstance(raw_url, str):
            continue
        url = raw_url.strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        raw_summary = item.get("summary")
        raw_content = item.get("content")
        if isinstance(raw_summary, str) and raw_summary.strip():
            summary = raw_summary
        elif isinstance(raw_content, str):
            summary = raw_content
        else:
            summary = ""
        source_record = {
            "url": url,
            "title": item.get("title", "") if isinstance(item.get("title"), str) else "",
            "summary": summary,
        }
        sources.append(source_record)
        search_result["sources"].append(source_record)
    return search_result


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
    system_prompt = _strip_ctrl(system_prompt)

    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="请开始研究当前章节，收集足够的证据。"),
    ]

    search_results: list[dict[str, Any]] = []
    sources: list[dict] = []
    seen_urls: set[str] = set()

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
                content = _serialize_tool_result(raw_result)
                search_results.append(
                    _extract_results(raw_result, sources, seen_urls)
                )
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

    return {
        "search_results": search_results,
        "section_sources": sources,
    }
