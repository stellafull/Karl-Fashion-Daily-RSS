import pytest


@pytest.fixture
def mock_config():
    return {"configurable": {"thread_id": "test-001"}}
