"""DeepScout node — ReAct tool-calling agent for section-level evidence collection.

Uses create_react_agent from langgraph.prebuilt to run a ReAct loop that searches
the web, reflects, and collects evidence for a single research section. Parses tool
call responses to extract search_results and section_sources.
"""

import json
import logging

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent

from deep_agents.configuration import Configuration
from deep_agents.prompts import deep_scout_prompt
from deep_agents.state import SectionState
from deep_agents.utils import get_all_tools, get_api_key_for_model, get_today_str

logger = logging.getLogger(__name__)


def _parse_agent_messages(messages):
    """Parse tool messages from agent output to extract search results and sources."""
    search_results = []
    sources = []
    seen_urls = set()

    for msg in messages:
        msg_type = getattr(msg, "type", None) or (
            msg.get("type") if isinstance(msg, dict) else None
        )
        if msg_type == "tool":
            content = getattr(msg, "content", "") or (
                msg.get("content", "") if isinstance(msg, dict) else ""
            )
            if content:
                search_results.append({"raw": str(content)[:3000]})
                # Try to extract URLs from JSON content
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        for item in data:
                            url = item.get("url", "") if isinstance(item, dict) else ""
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                sources.append(
                                    {
                                        "source_id": f"src_{len(sources):03d}",
                                        "url": url,
                                        "title": (
                                            item.get("title", "")
                                            if isinstance(item, dict)
                                            else ""
                                        ),
                                        "credibility_score": 0.7,
                                    }
                                )
                except Exception:
                    pass

    return search_results, sources


async def deep_scout_node(state: SectionState, config: RunnableConfig) -> dict:
    """Run a ReAct agent to collect evidence for a single research section."""
    configurable = Configuration.from_runnable_config(config)

    tools = await get_all_tools(config)

    model = init_chat_model(
        model=configurable.research_model,
        max_tokens=configurable.research_model_max_tokens,
        api_key=get_api_key_for_model(configurable.research_model, config),
        base_url=configurable.openai_compatible_base_url,
    )

    research_goal = state.get("research_goal", "")
    section_title = state.get("section_title", "")
    section_description = state.get("section_description", "")
    search_queries = state.get("search_queries", [])
    hypotheses = state.get("hypotheses", [])
    budget = state.get("budget", {})
    max_searches = budget.get("max_searches", 5)

    system_prompt = deep_scout_prompt.format(
        date=get_today_str(),
        research_goal=research_goal,
        section_title=section_title,
        section_description=section_description,
        search_queries="\n".join(search_queries),
        hypotheses=json.dumps(hypotheses, ensure_ascii=False),
        max_searches=max_searches,
    )

    agent = create_react_agent(model, tools, prompt=system_prompt)

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="请开始研究当前章节，收集足够的证据。")]},
            config=config,
        )
        search_results, section_sources = _parse_agent_messages(result["messages"])
        return {
            "search_results": search_results,
            "section_sources": section_sources,
        }
    except Exception as exc:
        logger.warning("deep_scout_node failed: %s", exc)
        return {
            "search_results": [],
            "section_sources": [],
        }
