"""Tests for src/llm/backend.py -- no live OpenAI calls, everything mocked."""

from __future__ import annotations

import pytest

from src.config import Config
from src.llm.backend import OpenAIBackend, is_quota_error


def make_backend() -> OpenAIBackend:
    cfg = Config(USE_OPENAI=True, OPENAI_API_KEY="sk-test", OPENAI_MODEL="gpt-4o-mini")
    return OpenAIBackend(cfg)


class _FakeChoice:
    def __init__(self, content: str):
        self.message = type("Msg", (), {"content": content})()


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str = "OK", capture: dict | None = None):
        self._content = content
        self._capture = capture if capture is not None else {}

    def create(self, **kwargs):
        self._capture.update(kwargs)
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions):
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions):
        self.chat = _FakeChat(completions)


def test_generate_passes_through_arguments(mocker):
    backend = make_backend()
    capture: dict = {}
    fake_client = _FakeClient(_FakeCompletions(content="hello world", capture=capture))
    mocker.patch.object(backend, "_get_client", return_value=fake_client)

    result = backend.generate("a prompt", system="a system", temperature=0.3, max_tokens=42, json_mode=False)

    assert result == "hello world"
    assert capture["model"] == "gpt-4o-mini"
    assert capture["temperature"] == 0.3
    assert capture["max_tokens"] == 42
    assert capture["messages"] == [
        {"role": "system", "content": "a system"},
        {"role": "user", "content": "a prompt"},
    ]
    assert capture["response_format"] is None


def test_generate_respects_json_mode(mocker):
    backend = make_backend()
    capture: dict = {}
    fake_client = _FakeClient(_FakeCompletions(content="{}", capture=capture))
    mocker.patch.object(backend, "_get_client", return_value=fake_client)

    backend.generate("a prompt", json_mode=True)

    assert capture["response_format"] == {"type": "json_object"}


def test_quota_error_falls_back_to_local_model(mocker):
    """Rule 1.5: OpenAI quota failure must disable OpenAI and fall back to the local model."""
    backend = make_backend()
    mocker.patch.object(
        backend,
        "_call_openai",
        side_effect=RuntimeError("Error code: 429 - You exceeded your current quota, insufficient_quota"),
    )
    fake_local = mocker.patch.object(backend, "_call_local", return_value="local fallback answer")

    result = backend.generate("a prompt")

    assert result == "local fallback answer"
    assert backend.disabled is True
    assert "429" in backend.disabled_reason
    fake_local.assert_called_once()


def test_disabled_backend_skips_openai_on_subsequent_calls(mocker):
    """Once disabled this session, no further doomed OpenAI calls should be attempted."""
    backend = make_backend()
    fake_openai = mocker.patch.object(
        backend, "_call_openai", side_effect=RuntimeError("insufficient_quota")
    )
    mocker.patch.object(backend, "_call_local", return_value="local answer")

    backend.generate("first question")
    backend.generate("second question")

    assert fake_openai.call_count == 1
    assert backend.disabled is True


def test_non_quota_error_is_reraised_not_masked(mocker):
    backend = make_backend()
    mocker.patch.object(backend, "_call_openai", side_effect=ValueError("some unrelated bug"))

    with pytest.raises(ValueError, match="some unrelated bug"):
        backend.generate("a prompt")

    assert backend.disabled is False


def test_use_openai_false_goes_straight_to_local(mocker):
    cfg = Config(USE_OPENAI=False)
    backend = OpenAIBackend(cfg)
    fake_local = mocker.patch.object(backend, "_call_local", return_value="local only")

    result = backend.generate("a prompt")

    assert result == "local only"
    fake_local.assert_called_once()


@pytest.mark.parametrize(
    "message,expected",
    [
        ("insufficient_quota", True),
        ("Rate limit exceeded", True),
        ("Error code: 429", True),
        ("You exceeded your current quota", True),
        ("billing issue on account", True),
        ("a totally unrelated ValueError", False),
    ],
)
def test_is_quota_error(message, expected):
    assert is_quota_error(RuntimeError(message)) is expected
