"""Claude Code Provider — Anthropic via Claude CLI OAuth token."""

from __future__ import annotations

import asyncio
import shutil
from typing import Any

from loguru import logger

from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.providers.base import LLMResponse


async def _get_claude_token() -> str:
    """Fetch a fresh OAuth token from the Claude Code CLI."""
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RuntimeError(
            "Claude Code CLI (claude) not found in PATH. "
            "Install it first: https://docs.anthropic.com/en/docs/claude-code"
        )

    proc = await asyncio.create_subprocess_exec(
        claude_bin, "auth", "token",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        err = stderr.decode().strip() or stdout.decode().strip()
        raise RuntimeError(
            f"claude auth token failed (exit {proc.returncode}): {err}\n"
            "Run 'claude auth login' to authenticate first."
        )

    token = stdout.decode().strip()
    if not token:
        raise RuntimeError(
            "claude auth token returned empty output. "
            "Run 'claude auth login' to authenticate first."
        )
    return token


def _strip_claude_code_prefix(model: str) -> str:
    """Remove the claude-code/ or claude_code/ prefix."""
    if model.startswith("claude-code/") or model.startswith("claude_code/"):
        return model.split("/", 1)[1]
    return model


class ClaudeCodeProvider(LiteLLMProvider):
    """Anthropic via Claude Code CLI OAuth token.

    Fetches a fresh token before each chat() call and passes it
    to LiteLLM as the Anthropic API key.
    """

    def __init__(self, default_model: str = "claude-code/claude-sonnet-4-6"):
        super().__init__(
            api_key=None,
            api_base=None,
            default_model=_strip_claude_code_prefix(default_model),
            provider_name="anthropic",
        )
        self._original_model = default_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        try:
            token = await _get_claude_token()
        except RuntimeError as e:
            logger.error("Claude Code token fetch failed: {}", e)
            return LLMResponse(content=str(e), finish_reason="error")

        # Note: mutating self.api_key is not concurrency-safe, but nanobot
        # uses a single provider instance per session so this is acceptable.
        self.api_key = token

        resolved_model = _strip_claude_code_prefix(model) if model else None

        return await super().chat(
            messages=messages,
            tools=tools,
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

    def get_default_model(self) -> str:
        return self._original_model
