"""Reviewer: strict quality review -> ReviewResult with quality_score and issues."""
import json
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from deep_agents.configuration import Configuration
from deep_agents.prompts import reviewer_prompt
from deep_agents.schemas import ReviewResult
from deep_agents.state import ResearchState
from deep_agents.utils import _strip_ctrl, get_api_key_for_model


def _known_source_urls_from_state(state: ResearchState) -> set[str]:
    known_urls: set[str] = set()

    for source in state.get("sources", []):
        if not isinstance(source, dict):
            continue
        raw = source.get("url")
        if isinstance(raw, str) and raw.strip():
            known_urls.add(raw.strip())

    for field_name in ("facts", "data_points", "hypothesis_evidence"):
        for item in state.get(field_name, []):
            if not isinstance(item, dict):
                continue
            raw = item.get("source_url")
            if isinstance(raw, str) and raw.strip():
                known_urls.add(raw.strip())

    for item in state.get("contradictions", []):
        if not isinstance(item, dict):
            continue
        for field_name in ("source_url_a", "source_url_b"):
            raw = item.get(field_name)
            if isinstance(raw, str) and raw.strip():
                known_urls.add(raw.strip())

    return known_urls


def _validate_claim_checks_use_source_url(
    review_result: dict, state: ResearchState
) -> None:
    known_urls = _known_source_urls_from_state(state)

    for item in review_result.get("claim_checks", []):
        if any(key in item for key in ("source_id", "source_id_a", "source_id_b")):
            raise ValueError("reviewer claim_checks must use source_url, not source_id fields")
        source_url = item.get("source_url")
        if not isinstance(source_url, str) or not source_url.strip():
            raise ValueError("reviewer claim_checks must include source_url")
        source_url = source_url.strip()
        item["source_url"] = source_url
        if known_urls and source_url not in known_urls:
            raise ValueError(f"Unresolved source_url '{source_url}' in reviewer claim_checks")


async def reviewer_node(
    state: ResearchState, config: RunnableConfig
) -> Command[Literal["reviser", "final_check"]]:
    """Review the full report; route to reviser if quality insufficient, else final_check."""
    configurable = Configuration.from_runnable_config(config)
    model = (
        init_chat_model(
            model=configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            base_url=configurable.openai_compatible_base_url,
            disable_streaming=True,
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
    prompt_text = _strip_ctrl(prompt_text)

    result: ReviewResult = await model.ainvoke([HumanMessage(content=prompt_text)])
    review_result = result.model_dump()
    _validate_claim_checks_use_source_url(review_result, state)

    if review_result.get("verdict") != "pass" and state.get("revision_count", 0) < 2:
        return Command(goto="reviser", update={"review_result": review_result})
    return Command(goto="final_check", update={"review_result": review_result})
