"""Lead writer: iterates over all sections in one call, returns all SectionDrafts."""

import json

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from deep_agents.configuration import Configuration
from deep_agents.prompts import writer_prompt
from deep_agents.schemas import SectionDraft
from deep_agents.state import ResearchState
from deep_agents.utils import get_api_key_for_model


async def writer_node(state: ResearchState, config: RunnableConfig) -> dict:
    """Write all section drafts sequentially, accumulating context between sections."""
    configurable = Configuration.from_runnable_config(config)
    model = (
        init_chat_model(
            model=configurable.research_model,
            max_tokens=configurable.research_model_max_tokens,
            api_key=get_api_key_for_model(configurable.research_model, config),
            base_url=configurable.openai_compatible_base_url,
        )
        .with_structured_output(SectionDraft)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    sections = state.get("sections", [])
    all_facts = state.get("facts", [])
    all_data_points = state.get("data_points", [])
    all_charts = state.get("charts", [])
    all_contradictions = state.get("contradictions", [])
    hypothesis_evidence = state.get("hypothesis_evidence", [])
    language = state.get("language", "zh")

    sections_list = json.dumps(
        [{"id": s["id"], "title": s["title"]} for s in sections],
        ensure_ascii=False,
    )

    drafts = []
    previous_sections_memo = "无"

    for section in sections:
        sec_id = section["id"]

        # Filter evidence to this section
        sec_facts = [f for f in all_facts if f.get("section_id") == sec_id or not f.get("section_id")]
        sec_data = [d for d in all_data_points if d.get("section_id") == sec_id or not d.get("section_id")]
        sec_charts = [c for c in all_charts if c.get("section_id") == sec_id or not c.get("section_id")]
        sec_contradictions = [c for c in all_contradictions if c.get("section_id") == sec_id]

        prompt_text = writer_prompt.format(
            research_goal=state["research_goal"],
            sections_list=sections_list,
            hypothesis_evidence=json.dumps(hypothesis_evidence[:10], ensure_ascii=False),
            previous_sections_memo=previous_sections_memo,
            section_title=section["title"],
            section_description=section["description"],
            section_id=sec_id,
            section_facts=json.dumps(sec_facts[:20], ensure_ascii=False),
            section_data_points=json.dumps(sec_data[:10], ensure_ascii=False),
            charts=json.dumps(sec_charts[:5], ensure_ascii=False),
            contradictions=json.dumps(sec_contradictions[:5], ensure_ascii=False),
            language=language,
        )

        draft: SectionDraft = await model.ainvoke([HumanMessage(content=prompt_text)])
        draft_dict = draft.model_dump()
        draft_dict["section_id"] = sec_id
        drafts.append(draft_dict)

        # Update memo for next section
        previous_sections_memo = "; ".join(
            f"{d['section_id']}({d['content'][:80]}...)" for d in drafts
        )

    return {"section_drafts": drafts}
