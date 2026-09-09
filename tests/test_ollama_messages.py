"""Tests for the Ollama message schemas."""

import pytest

from pfun_llm.backend.ollama import OllamaMessage, OllamaMessages


class TestOllamaMessage:
    def test_default_role_is_user(self):
        msg = OllamaMessage(content="hello")
        assert msg.role == "user"

    def test_content_required(self):
        with pytest.raises(Exception):
            OllamaMessage()

    def test_custom_role(self):
        msg = OllamaMessage(content="assistant reply", role="assistant")
        assert msg.role == "assistant"
        assert msg.content == "assistant reply"


class TestOllamaMessages:
    def test_empty_messages(self):
        msgs = OllamaMessages()
        assert msgs.model_dump() == {"messages": []}

    def test_serialize_single_message(self):
        msgs = OllamaMessages(messages=[OllamaMessage(content="hi")])
        assert msgs.model_dump() == {"messages": [{"role": "user", "content": "hi"}]}

    def test_serialize_multiple_messages_preserves_order(self):
        msgs = OllamaMessages(
            messages=[
                OllamaMessage(content="q1", role="user"),
                OllamaMessage(content="a1", role="assistant"),
                OllamaMessage(content="q2", role="user"),
            ]
        )
        assert msgs.model_dump() == {
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
            ]
        }

    def test_serialize_raises_on_plain_string(self):
        msgs = OllamaMessages(messages=["hi"])
        with pytest.raises(ValueError):
            msgs.model_dump()