"""Compress search — condense raw search results before think_tool reflection.

Mirrors the compress_research pattern from open_deep_research: called inside
deep_scout's tool loop after each tavily_search call, compressing results
before the LLM sees them via think_tool. Preserves URLs, data, and key
evidence while fitting within downstream context windows.
"""

import json
import logging

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import compress_search_prompt
from deep_agents.utils import _strip_ctrl, get_api_key_for_model, get_today_str

logger = logging.getLogger(__name__)


def _message_content_to_text(content: object) -> str:
    """Normalize provider-specific content payloads into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


async def compress_search(
    raw_content: str,
    research_goal: str,
    section_title: str,
    section_description: str,
    hypotheses: list[str],
    config: RunnableConfig,
) -> str:
    """Compress raw search results into a concise summary."""
    if not raw_content or not raw_content.strip():
        return raw_content

    configurable = Configuration.from_runnable_config(config)

    model = init_chat_model(
        model=configurable.compression_model,
        max_tokens=configurable.compression_model_max_tokens,
        api_key=get_api_key_for_model(configurable.compression_model, config),
        base_url=configurable.openai_compatible_base_url,
        max_retries=configurable.provider_max_retries,
        disable_streaming=True,
    )

    prompt_text = compress_search_prompt.format(
        date=get_today_str(),
        research_goal=research_goal,
        section_title=section_title,
        section_description=section_description,
        hypotheses=json.dumps(hypotheses, ensure_ascii=False),
        raw_search_results=raw_content,
    )
    prompt_text = _strip_ctrl(prompt_text)

    response = await model.ainvoke([HumanMessage(content=prompt_text)])
    return _message_content_to_text(response.content)
