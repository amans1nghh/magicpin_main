"""Tests for the LLMClient class — timeout, fallback, and provider selection."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Provider selection logic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsupported_provider_raises():
    client = LLMClient(provider="unknown", api_key="key", model="m")
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        await client.complete("sys", "usr")


@pytest.mark.asyncio
async def test_openai_provider_calls_openai_endpoint():
    """Verify that provider='openai' hits the OpenAI endpoint and parses the response."""
    fake_response = httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": "Hello from OpenAI"}}
            ]
        },
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )

    with patch("src.llm_client.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=fake_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        client = LLMClient(provider="openai", api_key="test-key", model="gpt-4o-mini")
        result = await client.complete("system prompt", "user prompt")

    assert result == "Hello from OpenAI"
    mock_instance.post.assert_called_once()
    call_kwargs = mock_instance.post.call_args
    assert "api.openai.com" in call_kwargs.args[0]


@pytest.mark.asyncio
async def test_anthropic_provider_calls_anthropic_endpoint():
    """Verify that provider='anthropic' hits the Anthropic endpoint and parses the response."""
    fake_response = httpx.Response(
        200,
        json={
            "content": [
                {"text": "Hello from Anthropic"}
            ]
        },
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )

    with patch("src.llm_client.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=fake_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        client = LLMClient(provider="anthropic", api_key="test-key", model="claude-3-haiku")
        result = await client.complete("system prompt", "user prompt")

    assert result == "Hello from Anthropic"
    mock_instance.post.assert_called_once()
    call_kwargs = mock_instance.post.call_args
    assert "api.anthropic.com" in call_kwargs.args[0]


@pytest.mark.asyncio
async def test_provider_name_is_case_insensitive():
    """Provider names like 'OpenAI' or 'ANTHROPIC' should work."""
    fake_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "ok"}}]},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )

    with patch("src.llm_client.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=fake_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        client = LLMClient(provider="OpenAI", api_key="k", model="m")
        result = await client.complete("s", "u")

    assert result == "ok"


# ---------------------------------------------------------------------------
# complete_with_fallback — timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_on_timeout():
    """complete_with_fallback returns fallback when the LLM call times out."""
    with patch("src.llm_client.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        client = LLMClient(provider="openai", api_key="k", model="m", timeout=0.1)
        result = await client.complete_with_fallback("sys", "usr", "fallback text")

    assert result == "fallback text"


# ---------------------------------------------------------------------------
# complete_with_fallback — API error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_on_api_error():
    """complete_with_fallback returns fallback on HTTP 500 from the API."""
    error_response = httpx.Response(
        500,
        json={"error": "internal server error"},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )

    with patch("src.llm_client.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=error_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        client = LLMClient(provider="openai", api_key="k", model="m")
        result = await client.complete_with_fallback("sys", "usr", "safe fallback")

    assert result == "safe fallback"


@pytest.mark.asyncio
async def test_fallback_on_rate_limit():
    """complete_with_fallback returns fallback on HTTP 429 (rate limit)."""
    rate_limit_response = httpx.Response(
        429,
        json={"error": "rate limit exceeded"},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )

    with patch("src.llm_client.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=rate_limit_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        client = LLMClient(provider="openai", api_key="k", model="m")
        result = await client.complete_with_fallback("sys", "usr", "rate limited fallback")

    assert result == "rate limited fallback"


@pytest.mark.asyncio
async def test_fallback_on_connection_error():
    """complete_with_fallback returns fallback when the network is unreachable."""
    with patch("src.llm_client.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        client = LLMClient(provider="openai", api_key="k", model="m")
        result = await client.complete_with_fallback("sys", "usr", "network fallback")

    assert result == "network fallback"


# ---------------------------------------------------------------------------
# complete_with_fallback — success path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_returns_llm_response_on_success():
    """complete_with_fallback returns the actual LLM response when everything works."""
    fake_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "real answer"}}]},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )

    with patch("src.llm_client.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=fake_response)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        client = LLMClient(provider="openai", api_key="k", model="m")
        result = await client.complete_with_fallback("sys", "usr", "fallback")

    assert result == "real answer"


# ---------------------------------------------------------------------------
# Init defaults
# ---------------------------------------------------------------------------


def test_default_timeout():
    client = LLMClient(provider="openai", api_key="k", model="m")
    assert client._timeout == 8.0


def test_custom_timeout():
    client = LLMClient(provider="openai", api_key="k", model="m", timeout=5.0)
    assert client._timeout == 5.0
