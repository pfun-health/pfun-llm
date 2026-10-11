import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import ollama
from ollama import AsyncClient


@pytest.fixture
def mock_ollama_client():
    """Return a mock AsyncClient that returns a standard chat response."""
    client = AsyncMock(spec=AsyncClient)
    client.chat.return_value = {"message": {"content": "Hello, world!"}}
    return client


@pytest.fixture
def mock_settings():
    """Return a mock settings object with typical Ollama config."""
    settings = MagicMock()
    settings.ollama_host = "http://testhost:1234"
    settings.ollama_api_key = ""
    settings.ollama_model = "gpt-oss:120b-cloud"
    return settings


@pytest.fixture
def mock_get_settings(mock_settings):
    """Patch pfun_common.settings.get_settings to return mock_settings."""
    with patch("pfun_common.settings.get_settings", return_value=mock_settings) as gs:
        yield gs


@pytest.fixture
def model(mock_ollama_client, mock_get_settings):
    """Create an OllamaGenerativeModel with a mocked client (no network)."""
    from pfun_llm.backend.ollama import OllamaGenerativeModel

    m = OllamaGenerativeModel()
    m._client = mock_ollama_client
    return m


@pytest.fixture
def stream_model(mock_ollama_client, mock_get_settings):
    """Create a model whose client.chat returns an async iterator."""
    from pfun_llm.backend.ollama import OllamaGenerativeModel

    async def fake_stream(*args, **kwargs):
        async def _gen():
            yield {"message": {"content": "Hello "}}
            yield {"message": {"content": "world"}}
        return _gen()

    mock_ollama_client.chat.side_effect = fake_stream
    m = OllamaGenerativeModel()
    m._client = mock_ollama_client
    return m
