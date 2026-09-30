"""Async LLM client with timeout management and fallback.

Supports OpenAI and Anthropic providers via httpx async HTTP calls.
Designed for graceful degradation — the bot must never crash due to LLM issues.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class LLMClient:
    """Async LLM client with timeout management and fallback.

    Attributes:
        _provider: LLM provider name ("openai" or "anthropic").
        _api_key: API key for the provider.
        _model: Model identifier (e.g. "gpt-4o-mini", "claude-3-haiku-20240307").
        _timeout: Per-call timeout in seconds (default 8.0).
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        timeout: float = 8.0,
    ) -> None:
        self._provider = provider.lower()
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM and return the completion text.

        Raises:
            httpx.TimeoutException: If the request exceeds the configured timeout.
            httpx.HTTPStatusError: If the API returns a non-2xx status.
            ValueError: If the provider is unsupported or the response cannot be parsed.
        """
        if self._provider == "openai":
            return await self._complete_openai(system_prompt, user_prompt)
        elif self._provider == "anthropic":
            return await self._complete_anthropic(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self._provider}")

    async def complete_with_fallback(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_response: str,
    ) -> str:
        """Try LLM completion with retry; on persistent error return *fallback_response*.

        Retries once on timeout before falling back.
        """
        for attempt in range(2):
            try:
                return await self.complete(system_prompt, user_prompt)
            except Exception:
                if attempt == 0:
                    logger.warning("LLM attempt 1 failed, retrying...")
                    continue
                logger.exception("LLM completion failed after retry, returning fallback")
                return fallback_response
        return fallback_response

    # ------------------------------------------------------------------
    # Provider-specific implementations
    # ------------------------------------------------------------------

    async def _complete_openai(self, system_prompt: str, user_prompt: str) -> str:
        """POST to the OpenAI chat completions endpoint."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": 2000,
        }

        timeout = httpx.Timeout(self._timeout, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def _complete_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """POST to the Anthropic messages endpoint."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model,
            "max_tokens": 2000,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }

        timeout = httpx.Timeout(self._timeout, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]
