"""Claude Code Provider — Anthropic via Claude CLI OAuth token."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.providers.base import LLMResponse

# ---------------------------------------------------------------------------
# Credentials dataclass & helpers
# ---------------------------------------------------------------------------

_EXPIRY_BUFFER_MS = 5 * 60 * 1000  # 5 minutes in milliseconds


@dataclass
class ClaudeCredentials:
    """Holds an OAuth credential from any source."""

    access_token: str
    refresh_token: str | None = None
    expires_at: int | None = None
    source: str = "cli"


def _is_expired(creds: ClaudeCredentials) -> bool:
    """Return True if *creds* will expire within the 5-minute buffer."""
    if creds.expires_at is None:
        return False
    now_ms = int(time.time() * 1000)
    return creds.expires_at - now_ms < _EXPIRY_BUFFER_MS


# ---------------------------------------------------------------------------
# Tier 1: macOS Keychain
# ---------------------------------------------------------------------------

_KEYCHAIN_SERVICE = "Claude Code-credentials"


async def _read_keychain() -> ClaudeCredentials | None:
    """Read credentials from the macOS Keychain. Returns None on non-mac or failure."""
    if sys.platform != "darwin":
        return None

    try:
        proc = await asyncio.create_subprocess_exec(
            "security", "find-generic-password",
            "-s", _KEYCHAIN_SERVICE, "-w",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.debug("Keychain lookup failed (exit {})", proc.returncode)
            return None

        data = json.loads(stdout.decode().strip())
        oauth = data["claudeAiOauth"]

        return ClaudeCredentials(
            access_token=oauth["accessToken"],
            refresh_token=oauth.get("refreshToken"),
            expires_at=oauth.get("expiresAt"),
            source="keychain",
        )
    except Exception as exc:
        logger.debug("Keychain credential read failed: {}", exc)
        return None


# ---------------------------------------------------------------------------
# Tier 2: File-based credentials
# ---------------------------------------------------------------------------


def _get_credentials_file_path() -> Path:
    """Return the path to the Claude credentials file."""
    return Path.home() / ".claude" / ".credentials.json"


async def _read_credentials_file() -> ClaudeCredentials | None:
    """Read credentials from the on-disk credentials file."""
    try:
        path = _get_credentials_file_path()
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)

        # Support both {claudeAiOauth: ...} and {data: {claudeAiOauth: ...}}
        if "claudeAiOauth" in data:
            oauth = data["claudeAiOauth"]
        elif "data" in data and "claudeAiOauth" in data["data"]:
            oauth = data["data"]["claudeAiOauth"]
        else:
            logger.debug("Credentials file missing claudeAiOauth key")
            return None

        return ClaudeCredentials(
            access_token=oauth["accessToken"],
            refresh_token=oauth.get("refreshToken"),
            expires_at=oauth.get("expiresAt"),
            source="file",
        )
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.debug("Credentials file read failed: {}", exc)
        return None


# ---------------------------------------------------------------------------
# Tier 3: CLI fallback
# ---------------------------------------------------------------------------


async def _read_cli_token() -> ClaudeCredentials | None:
    """Fetch a fresh OAuth token from the Claude Code CLI."""
    claude_bin = shutil.which("claude")
    if not claude_bin:
        logger.debug("Claude Code CLI (claude) not found in PATH")
        return None

    try:
        proc = await asyncio.create_subprocess_exec(
            claude_bin, "auth", "token",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = stderr.decode().strip() or stdout.decode().strip()
            logger.debug("claude auth token failed (exit {}): {}", proc.returncode, err)
            return None

        token = stdout.decode().strip()
        if not token:
            logger.debug("claude auth token returned empty output")
            return None

        return ClaudeCredentials(access_token=token, source="cli")
    except Exception as exc:
        logger.debug("CLI token fetch failed: {}", exc)
        return None


# ---------------------------------------------------------------------------
# Unified credential reader (three-tier fallback)
# ---------------------------------------------------------------------------


async def _read_credentials() -> ClaudeCredentials | None:
    """Try Keychain → file → CLI, returning the first successful result."""
    creds = await _read_keychain()
    if creds is not None:
        return creds

    creds = await _read_credentials_file()
    if creds is not None:
        return creds

    creds = await _read_cli_token()
    if creds is not None:
        return creds

    return None


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


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
            token = await _read_cli_token()
            if token is None:
                raise RuntimeError(
                    "Claude Code CLI (claude) not found or auth failed. "
                    "Run 'claude auth login' to authenticate first."
                )
            token_str = token.access_token
        except RuntimeError as e:
            logger.error("Claude Code token fetch failed: {}", e)
            return LLMResponse(content=str(e), finish_reason="error")

        # Note: mutating self.api_key is not concurrency-safe, but nanobot
        # uses a single provider instance per session so this is acceptable.
        self.api_key = token_str

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
