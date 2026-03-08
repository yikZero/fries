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

import httpx
from loguru import logger

from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.providers.base import LLMResponse

# ---------------------------------------------------------------------------
# Credentials dataclass & helpers
# ---------------------------------------------------------------------------

_EXPIRY_BUFFER_MS = 5 * 60 * 1000  # 5 minutes in milliseconds

_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_REFRESH_HEADERS = {
    "User-Agent": "claude-cli/2.1.2",
    "Referer": "https://claude.ai/",
    "Origin": "https://claude.ai",
    "Content-Type": "application/json",
}

_OAUTH_BETA = "oauth-2025-04-20"


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
# Token refresh & write-back
# ---------------------------------------------------------------------------


async def _refresh_token(creds: ClaudeCredentials) -> ClaudeCredentials | None:
    """Refresh an expired OAuth token. Returns new credentials or None on failure."""
    if not creds.refresh_token:
        return None

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _TOKEN_URL,
                headers=_REFRESH_HEADERS,
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": creds.refresh_token,
                    "client_id": _CLIENT_ID,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        expires_at = int(time.time() * 1000) + data["expires_in"] * 1000
        new_creds = ClaudeCredentials(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", creds.refresh_token),
            expires_at=expires_at,
            source=creds.source,
        )
        await _write_back_credentials(new_creds)
        return new_creds
    except Exception as exc:
        logger.warning("OAuth token refresh failed: {}", exc)
        return None


async def _write_back_credentials(creds: ClaudeCredentials) -> None:
    """Persist refreshed credentials back to their original source."""
    oauth_data = {
        "accessToken": creds.access_token,
        "refreshToken": creds.refresh_token,
        "expiresAt": creds.expires_at,
    }

    if creds.source == "keychain" and sys.platform == "darwin":
        await _write_keychain(oauth_data)
    elif creds.source == "file":
        _write_credentials_file(oauth_data)


async def _write_keychain(oauth_data: dict) -> None:
    """Write updated OAuth data back to the macOS Keychain."""
    try:
        # Read existing keychain data to preserve other fields
        proc = await asyncio.create_subprocess_exec(
            "security", "find-generic-password",
            "-s", _KEYCHAIN_SERVICE, "-w",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()

        if proc.returncode == 0:
            data = json.loads(stdout.decode().strip())
        else:
            data = {}

        data["claudeAiOauth"] = oauth_data
        payload = json.dumps(data)

        proc = await asyncio.create_subprocess_exec(
            "security", "add-generic-password",
            "-U", "-s", _KEYCHAIN_SERVICE,
            "-w", payload, "-a", "claude-code",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
    except Exception as exc:
        logger.debug("Keychain write-back failed: {}", exc)


def _write_credentials_file(oauth_data: dict) -> None:
    """Write updated OAuth data back to the credentials file."""
    try:
        path = _get_credentials_file_path()
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)

        # Detect wrapper format
        if "data" in data and "claudeAiOauth" in data["data"]:
            data["data"]["claudeAiOauth"] = oauth_data
        else:
            data["claudeAiOauth"] = oauth_data

        path.write_text(json.dumps(data), encoding="utf-8")
    except Exception as exc:
        logger.debug("Credentials file write-back failed: {}", exc)


# ---------------------------------------------------------------------------
# Cached credential access (full flow)
# ---------------------------------------------------------------------------

_cached_credentials: ClaudeCredentials | None = None


async def _get_claude_token() -> ClaudeCredentials:
    """Return valid credentials, using cache / refresh / fresh-read as needed."""
    global _cached_credentials

    # 1. Cache hit
    if _cached_credentials and not _is_expired(_cached_credentials):
        return _cached_credentials

    # 2. Cache expired → try refresh
    if _cached_credentials and _is_expired(_cached_credentials):
        if _cached_credentials.refresh_token:
            refreshed = await _refresh_token(_cached_credentials)
            if refreshed:
                _cached_credentials = refreshed
                return _cached_credentials
        _cached_credentials = None

    # 3. Read fresh
    creds = await _read_credentials()
    if not creds:
        raise RuntimeError(
            "No Claude credentials found. "
            "Install Claude Code CLI and run: claude auth login"
        )

    # 4. Freshly read but expired → try refresh
    if _is_expired(creds) and creds.refresh_token:
        refreshed = await _refresh_token(creds)
        if refreshed:
            creds = refreshed

    _cached_credentials = creds
    return _cached_credentials


def _clear_cached_credentials() -> None:
    """Reset the cached credentials (useful for testing)."""
    global _cached_credentials
    _cached_credentials = None


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
        self.extra_headers["anthropic-beta"] = _OAUTH_BETA

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
            creds = await _get_claude_token()
        except RuntimeError as e:
            logger.error("Claude Code token fetch failed: {}", e)
            return LLMResponse(content=str(e), finish_reason="error")

        # Note: mutating self.api_key is not concurrency-safe, but nanobot
        # uses a single provider instance per session so this is acceptable.
        self.api_key = creds.access_token

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
