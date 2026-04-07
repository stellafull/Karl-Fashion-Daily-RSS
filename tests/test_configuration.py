from pydantic import ValidationError

from deep_agents.configuration import Configuration


def test_configuration_uses_provider_base_url_not_request_endpoint() -> None:
    config = Configuration()

    assert config.openai_compatible_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_configuration_rejects_chat_completions_endpoint_as_base_url() -> None:
    try:
        Configuration(
            openai_compatible_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        )
    except ValidationError as exc:
        assert "must be a provider base URL" in str(exc)
    else:
        raise AssertionError("Expected ValidationError for endpoint URL")


def test_configuration_allows_non_request_endpoint_paths() -> None:
    config = Configuration(
        openai_compatible_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/images"
    )

    assert config.openai_compatible_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1/images"


def test_configuration_uses_single_deep_scout_loop_cap() -> None:
    config = Configuration()

    assert config.max_deep_scout_iterations == 5
    assert "max_researcher_iterations" not in Configuration.model_fields
    assert "max_react_tool_calls" not in Configuration.model_fields
