import json
from pathlib import Path

import pytest

from claude_thermos.credentials import (
    StoredLogin,
    credentials_of,
    oauth_token,
    stored_login,
    with_credentials,
)

_OLD = "sk-ant-oat01-old"
_NEW = "sk-ant-oat01-new"


def _bearer(token: str) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


def _write_login(path: Path, token: str) -> None:
    path.write_text(json.dumps({"claudeAiOauth": {"accessToken": token, "refreshToken": "r"}}))


def _login(tmp_path: Path, token: str) -> tuple[StoredLogin, Path]:
    path = tmp_path / ".credentials.json"
    _write_login(path, token)
    return StoredLogin(path), path


def test_credentials_of_picks_the_auth_headers_whatever_their_case() -> None:
    headers = {"Authorization": "Bearer abc", "X-Api-Key": "key", "anthropic-beta": "x"}

    assert credentials_of(headers) == {"authorization": "Bearer abc", "x-api-key": "key"}


def test_with_credentials_replaces_only_the_credentials() -> None:
    headers = {"Authorization": "Bearer old", "anthropic-beta": "x"}

    swapped = with_credentials(headers, _bearer("new"))

    assert swapped == {"authorization": "Bearer new", "anthropic-beta": "x"}
    assert headers == {"Authorization": "Bearer old", "anthropic-beta": "x"}


def test_with_no_credentials_the_headers_keep_their_own() -> None:
    headers = {"Authorization": "Bearer old", "anthropic-beta": "x"}

    assert with_credentials(headers, {}) == headers


def test_oauth_token_is_read_from_a_bearer_header() -> None:
    assert oauth_token(_bearer(_OLD)) == _OLD


@pytest.mark.parametrize(
    "credentials",
    [
        {"x-api-key": "sk-ant-api03-key"},
        {"authorization": "Bearer gateway-token"},
        {},
    ],
)
def test_oauth_token_ignores_other_ways_of_authenticating(credentials: dict[str, str]) -> None:
    assert oauth_token(credentials) is None


def test_read_returns_the_stored_access_token(tmp_path: Path) -> None:
    login, _ = _login(tmp_path, _OLD)

    assert login.read() == _OLD


@pytest.mark.parametrize(
    "contents",
    [
        "not json",
        "[]",
        json.dumps({"claudeAiOauth": None}),
        # What Claude Code leaves behind once the login has ended.
        json.dumps({"claudeAiOauth": {"accessToken": "", "refreshToken": "", "expiresAt": 0}}),
    ],
)
def test_read_finds_no_token_in_a_file_without_one(tmp_path: Path, contents: str) -> None:
    path = tmp_path / ".credentials.json"
    path.write_text(contents)

    assert StoredLogin(path).read() is None


def test_read_finds_no_token_without_a_file(tmp_path: Path) -> None:
    assert StoredLogin(tmp_path / ".credentials.json").read() is None


def test_token_seen_in_the_file_is_replaced_by_the_one_that_follows_it(tmp_path: Path) -> None:
    login, path = _login(tmp_path, _OLD)
    login.observe(_bearer(_OLD))
    _write_login(path, _NEW)

    assert login.successor(_bearer(_OLD)) == _bearer(_NEW)


def test_replacement_keeps_the_other_credential_headers(tmp_path: Path) -> None:
    login, path = _login(tmp_path, _OLD)
    credentials = {**_bearer(_OLD), "x-api-key": "key"}
    login.observe(credentials)
    _write_login(path, _NEW)

    assert login.successor(credentials) == {**_bearer(_NEW), "x-api-key": "key"}


def test_token_never_seen_in_the_file_is_not_replaced(tmp_path: Path) -> None:
    # The file is some other account's login: the session's token came from
    # another config directory, or from the environment.
    login, _ = _login(tmp_path, "sk-ant-oat01-another-account")
    login.observe(_bearer(_OLD))

    assert login.successor(_bearer(_OLD)) is None


def test_token_the_login_was_never_shown_is_not_replaced(tmp_path: Path) -> None:
    login, _ = _login(tmp_path, _NEW)

    assert login.successor(_bearer(_OLD)) is None


def test_nothing_replaces_a_token_the_file_still_holds(tmp_path: Path) -> None:
    login, _ = _login(tmp_path, _OLD)
    login.observe(_bearer(_OLD))

    assert login.successor(_bearer(_OLD)) is None


def test_nothing_replaces_a_token_once_the_login_has_ended(tmp_path: Path) -> None:
    login, path = _login(tmp_path, _OLD)
    login.observe(_bearer(_OLD))
    _write_login(path, "")

    assert login.successor(_bearer(_OLD)) is None


def test_replacement_can_be_replaced_in_turn(tmp_path: Path) -> None:
    login, path = _login(tmp_path, _OLD)
    login.observe(_bearer(_OLD))
    _write_login(path, _NEW)
    login.successor(_bearer(_OLD))
    _write_login(path, "sk-ant-oat01-newer")

    assert login.successor(_bearer(_NEW)) == _bearer("sk-ant-oat01-newer")


def test_api_key_is_never_replaced(tmp_path: Path) -> None:
    login, _ = _login(tmp_path, _NEW)
    login.observe({"x-api-key": "sk-ant-api03-key"})

    assert login.successor({"x-api-key": "sk-ant-api03-key"}) is None


def test_stored_login_follows_the_config_dir(tmp_path: Path) -> None:
    _write_login(tmp_path / ".credentials.json", _OLD)

    assert stored_login({"CLAUDE_CONFIG_DIR": str(tmp_path)}).read() == _OLD


def test_stored_login_defaults_to_the_home_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".claude").mkdir()
    _write_login(tmp_path / ".claude" / ".credentials.json", _OLD)

    assert stored_login({}).read() == _OLD
