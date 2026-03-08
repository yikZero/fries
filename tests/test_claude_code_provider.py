import json
import time
import pytest


async def _async_return(val):
    return val


# Task 1 tests
def test_credentials_not_expired_when_far_future():
    from nanobot.providers.claude_code_provider import ClaudeCredentials, _is_expired
    creds = ClaudeCredentials(
        access_token="tok", refresh_token="ref",
        expires_at=int(time.time() * 1000) + 3600_000, source="keychain",
    )
    assert _is_expired(creds) is False


def test_credentials_expired_when_within_5_minutes():
    from nanobot.providers.claude_code_provider import ClaudeCredentials, _is_expired
    creds = ClaudeCredentials(
        access_token="tok", refresh_token="ref",
        expires_at=int(time.time() * 1000) + 4 * 60 * 1000, source="keychain",
    )
    assert _is_expired(creds) is True


def test_credentials_not_expired_when_no_expires_at():
    from nanobot.providers.claude_code_provider import ClaudeCredentials, _is_expired
    creds = ClaudeCredentials(access_token="tok", source="cli")
    assert _is_expired(creds) is False


# Task 2 tests
@pytest.mark.asyncio
async def test_read_keychain_parses_credentials(monkeypatch):
    from nanobot.providers.claude_code_provider import _read_keychain

    keychain_data = json.dumps({
        "claudeAiOauth": {
            "accessToken": "sk-ant-oat-test-access",
            "refreshToken": "sk-ant-oat-test-refresh",
            "expiresAt": 9999999999999,
        }
    })

    async def mock_subprocess(*args, **kwargs):
        class Result:
            returncode = 0
            async def communicate(self_):
                return keychain_data.encode(), b""
        return Result()

    monkeypatch.setattr("nanobot.providers.claude_code_provider.asyncio.create_subprocess_exec", mock_subprocess)
    monkeypatch.setattr("nanobot.providers.claude_code_provider.sys.platform", "darwin")

    creds = await _read_keychain()
    assert creds is not None
    assert creds.access_token == "sk-ant-oat-test-access"
    assert creds.refresh_token == "sk-ant-oat-test-refresh"
    assert creds.expires_at == 9999999999999
    assert creds.source == "keychain"


@pytest.mark.asyncio
async def test_read_keychain_returns_none_on_linux(monkeypatch):
    from nanobot.providers.claude_code_provider import _read_keychain
    monkeypatch.setattr("nanobot.providers.claude_code_provider.sys.platform", "linux")
    creds = await _read_keychain()
    assert creds is None


# Task 3 tests
@pytest.mark.asyncio
async def test_read_credentials_file(tmp_path, monkeypatch):
    from nanobot.providers.claude_code_provider import _read_credentials_file

    cred_file = tmp_path / ".credentials.json"
    cred_file.write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": "sk-ant-oat-file-access",
            "refreshToken": "sk-ant-oat-file-refresh",
            "expiresAt": 9999999999999,
        }
    }))
    monkeypatch.setattr("nanobot.providers.claude_code_provider._get_credentials_file_path", lambda: cred_file)

    creds = await _read_credentials_file()
    assert creds is not None
    assert creds.access_token == "sk-ant-oat-file-access"
    assert creds.source == "file"


@pytest.mark.asyncio
async def test_read_credentials_file_with_data_wrapper(tmp_path, monkeypatch):
    from nanobot.providers.claude_code_provider import _read_credentials_file

    cred_file = tmp_path / ".credentials.json"
    cred_file.write_text(json.dumps({
        "data": {
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat-wrapped",
                "refreshToken": "sk-ant-oat-wrapped-ref",
                "expiresAt": 9999999999999,
            }
        }
    }))
    monkeypatch.setattr("nanobot.providers.claude_code_provider._get_credentials_file_path", lambda: cred_file)

    creds = await _read_credentials_file()
    assert creds is not None
    assert creds.access_token == "sk-ant-oat-wrapped"


@pytest.mark.asyncio
async def test_read_credentials_file_returns_none_when_missing(tmp_path, monkeypatch):
    from nanobot.providers.claude_code_provider import _read_credentials_file
    monkeypatch.setattr("nanobot.providers.claude_code_provider._get_credentials_file_path", lambda: tmp_path / "nonexistent.json")

    creds = await _read_credentials_file()
    assert creds is None


# Task 4 tests
@pytest.mark.asyncio
async def test_read_credentials_fallback_keychain_to_file(monkeypatch):
    from nanobot.providers.claude_code_provider import _read_credentials, ClaudeCredentials

    monkeypatch.setattr("nanobot.providers.claude_code_provider._read_keychain", lambda: _async_return(None))
    file_creds = ClaudeCredentials(access_token="from-file", source="file")
    monkeypatch.setattr("nanobot.providers.claude_code_provider._read_credentials_file", lambda: _async_return(file_creds))

    creds = await _read_credentials()
    assert creds is not None
    assert creds.access_token == "from-file"


@pytest.mark.asyncio
async def test_read_credentials_fallback_to_cli(monkeypatch):
    from nanobot.providers.claude_code_provider import _read_credentials, ClaudeCredentials

    monkeypatch.setattr("nanobot.providers.claude_code_provider._read_keychain", lambda: _async_return(None))
    monkeypatch.setattr("nanobot.providers.claude_code_provider._read_credentials_file", lambda: _async_return(None))
    monkeypatch.setattr("nanobot.providers.claude_code_provider._read_cli_token", lambda: _async_return(ClaudeCredentials(access_token="from-cli", source="cli")))

    creds = await _read_credentials()
    assert creds is not None
    assert creds.source == "cli"
