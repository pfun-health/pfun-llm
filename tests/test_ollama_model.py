"""Tests for OllamaGenerativeModel."""

import pytest

import ollama
from ollama._types import ProgressResponse, ResponseError

from pfun_llm.backend.ollama import (
    _OLLAMA_DEFAULT_MODEL,
    OllamaGenerativeModel,
    OllamaMessage,
    OllamaMessages,
)


class TestInstantiation:
    def test_default_model_from_settings(self, mock_get_settings, mock_settings):
        mock_settings.ollama_model = "gemma3:4b-cloud"
        m = OllamaGenerativeModel()
        assert m._model == "gemma3:4b-cloud"

    def test_model_set_to_class_default_when_settings_blank(self, mock_get_settings):
        m = OllamaGenerativeModel()
        assert m._model == _OLLAMA_DEFAULT_MODEL

    def test_explicit_keyword_model(self, mock_get_settings):
        m = OllamaGenerativeModel(model="deepseek-v3.2:cloud")
        assert m._model == "deepseek-v3.2:cloud"

    def test_positional_model(self, mock_get_settings):
        m = OllamaGenerativeModel("gemma3:4b-cloud")
        assert m._model == "gemma3:4b-cloud"

    def test_extra_kwargs_saved(self, mock_get_settings):
        m = OllamaGenerativeModel(temperature=0.5, num_predict=128)
        assert m._extra_kwds == {"temperature": 0.5, "num_predict": 128}


class TestSetupGenaiClient:
    def test_uses_settings_host_and_api_key(
        self, mock_get_settings, mock_settings, monkeypatch
    ):
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        mock_settings.ollama_host = "http://ollama.internal:1234"
        mock_settings.ollama_api_key = "secret-key"
        client = OllamaGenerativeModel.setup_genai_client()
        assert isinstance(client, ollama.AsyncClient)
        assert str(client._client.base_url) == "http://ollama.internal:1234"
        assert client._client.headers.get("authorization") == "Bearer secret-key"

    def test_no_auth_header_when_api_key_absent(
        self, mock_get_settings, mock_settings, monkeypatch
    ):
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        mock_settings.ollama_api_key = ""
        client = OllamaGenerativeModel.setup_genai_client()
        assert client._client.headers.get("authorization") is None


class TestChat:
    @pytest.mark.asyncio
    async def test_chat_uses_model_and_messages(self, model):
        await model.chat(messages=[{"role": "user", "content": "hi"}])
        model._client.chat.assert_awaited_once_with(
            model="gpt-oss:120b-cloud",
            messages=[{"role": "user", "content": "hi"}],
        )

    @pytest.mark.asyncio
    async def test_chat_passes_extra_kwds(self, mock_get_settings, mock_ollama_client):
        m = OllamaGenerativeModel(options={"temperature": 0.2})
        m._client = mock_ollama_client
        await m.chat(messages=[{"role": "user", "content": "hi"}])
        mock_ollama_client.chat.assert_awaited_once_with(
            model="gpt-oss:120b-cloud",
            messages=[{"role": "user", "content": "hi"}],
            options={"temperature": 0.2},
        )

    @pytest.mark.asyncio
    async def test_chat_explicit_model_overrides_default(self, model):
        await model.chat(messages=[{"role": "user", "content": "hi"}], model="gemma3:4b-cloud")
        model._client.chat.assert_awaited_once_with(
            model="gemma3:4b-cloud",
            messages=[{"role": "user", "content": "hi"}],
        )

    @pytest.mark.asyncio
    async def test_chat_returns_client_response(self, model):
        model._client.chat.return_value = {"message": {"role": "assistant", "content": "hi!"}}
        resp = await model.chat(messages=[{"role": "user", "content": "hi"}])
        assert resp == {"message": {"role": "assistant", "content": "hi!"}}


class TestStreamChat:
    @pytest.mark.asyncio
    async def test_stream_chat_yields_content(self, stream_model):
        chunks = [chunk async for chunk in stream_model.stream_chat(messages=[{"role": "user", "content": "hi"}])]
        assert chunks == ["Hello ", "world"]

    @pytest.mark.asyncio
    async def test_stream_chat_forwards_stream_flag(self, stream_model):
        async for _ in stream_model.stream_chat(messages=[{"role": "user", "content": "hi"}]):
            pass
        assert stream_model._client.chat.await_args.kwargs["stream"] is True


class TestCallGenaiClient:
    @pytest.mark.asyncio
    async def test_formats_str_contents_and_returns_response(self, model):
        resp = await model.call_genai_client(contents="hello")
        model._client.chat.assert_awaited_once_with(
            model="gpt-oss:120b-cloud",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert resp == {"message": {"content": "Hello, world!"}}

    @pytest.mark.asyncio
    async def test_formats_list_contents(self, model):
        await model.call_genai_client(contents=["a", "b"])
        model._client.chat.assert_awaited_once_with(
            model="gpt-oss:120b-cloud",
            messages=[
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
            ],
        )

    @pytest.mark.asyncio
    async def test_passes_ollamamessages_through_without_reformatting(
        self, model, mock_ollama_client
    ):
        msgs = OllamaMessages(
            messages=[
                OllamaMessage(content="q", role="user"),
                OllamaMessage(content="a", role="assistant"),
            ]
        )
        await model.call_genai_client(contents=msgs)
        model._client.chat.assert_awaited_once_with(
            model="gpt-oss:120b-cloud",
            messages=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        )

    @pytest.mark.asyncio
    async def test_explicit_model_wins(self, model, mock_ollama_client):
        await model.call_genai_client(contents="hello", model="deepseek-v3.2:cloud")
        mock_ollama_client.chat.assert_awaited_once_with(
            model="deepseek-v3.2:cloud",
            messages=[{"role": "user", "content": "hello"}],
        )

    @pytest.mark.asyncio
    async def test_404_triggers_pull_then_retries(self, model, mock_ollama_client, monkeypatch):
        mock_ollama_client.chat.side_effect = [
            ResponseError(error="model not found", status_code=404),
            {"message": {"content": "retried!"}},
        ]
        monkeypatch.setattr(
            ollama,
            "pull",
            lambda model: ProgressResponse(
                status="completed", completed=1, total=1, digest="d"
            ),
        )
        resp = await model.call_genai_client(contents="hello")
        assert resp == {"message": {"content": "retried!"}}
        assert mock_ollama_client.chat.await_count == 2

    @pytest.mark.asyncio
    async def test_404_with_pull_failure_raises_runtimeerror(
        self, model, mock_ollama_client, monkeypatch
    ):
        mock_ollama_client.chat.side_effect = ResponseError(
            error="model not found", status_code=404
        )
        monkeypatch.setattr(
            ollama,
            "pull",
            lambda model: ProgressResponse(
                status="pulling", completed=0, total=100, digest="d"
            ),
        )
        with pytest.raises(RuntimeError):
            await model.call_genai_client(contents="hello")

    @pytest.mark.asyncio
    async def test_non_404_error_raises_runtimeerror(self, model, mock_ollama_client):
        mock_ollama_client.chat.side_effect = ResponseError(
            error="bad gateway", status_code=502
        )
        with pytest.raises(RuntimeError):
            await model.call_genai_client(contents="hello")

    @pytest.mark.asyncio
    async def test_no_contents_raises_valueerror(self, model):
        with pytest.raises(ValueError):
            await model.call_genai_client(contents=None)


class TestCallAndGenerateContent:
    @pytest.mark.asyncio
    async def test_call_directly(self, model):
        resp = await model(contents="hello")
        assert resp == {"message": {"content": "Hello, world!"}}
        model._client.chat.assert_awaited_once_with(
            model="gpt-oss:120b-cloud",
            messages=[{"role": "user", "content": "hello"}],
        )

    @pytest.mark.asyncio
    async def test_call_with_missing_contents_raises(self, model):
        with pytest.raises(ValueError, match="Contents must be provided"):
            await model(contents=None)

    @pytest.mark.asyncio
    async def test_generate_content(self, model):
        resp = await model.generate_content("hello")
        assert resp == {"message": {"content": "Hello, world!"}}
        model._client.chat.assert_awaited_once_with(
            model="gpt-oss:120b-cloud",
            messages=[{"role": "user", "content": "hello"}],
        )