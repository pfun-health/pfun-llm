"""Tests for the Ollama message formatting helpers."""

from pfun_llm.backend.ollama import (
    OllamaMessage,
    OllamaMessages,
    _conv_str2msg,
    _format_messages,
)


class TestConvStr2Msg:
    def test_str_to_user_message(self):
        msg = _conv_str2msg("hello")
        assert isinstance(msg, OllamaMessage)
        assert msg.content == "hello"
        assert msg.role == "user"

    def test_str_with_custom_role(self):
        msg = _conv_str2msg("hello", role="assistant")
        assert msg.role == "assistant"

    def test_ollamamsg_passthrough(self):
        orig = OllamaMessage(content="keep me", role="assistant")
        result = _conv_str2msg(orig)
        assert result is orig


class TestFormatMessages:
    def test_single_str_becomes_single_user_message(self):
        out = _format_messages("hello")
        assert isinstance(out, OllamaMessages)
        assert out.model_dump()["messages"] == [{"role": "user", "content": "hello"}]

    def test_list_str_becomes_multiple_user_messages(self):
        out = _format_messages(["a", "b"])
        assert out.model_dump()["messages"] == [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]

    def test_list_str_with_custom_role(self):
        out = _format_messages(["a", "b"], role="assistant")
        assert out.model_dump()["messages"] == [
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]

    def test_mixed_ollamamsg_list(self):
        out = _format_messages(
            [
                OllamaMessage(content="q", role="user"),
                OllamaMessage(content="a", role="assistant"),
            ]
        )
        assert out.model_dump()["messages"] == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]

    def test_returns_ollamamessages_instance(self):
        assert isinstance(_format_messages("x"), OllamaMessages)