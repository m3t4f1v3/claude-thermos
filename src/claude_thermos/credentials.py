import json
from collections.abc import Mapping
from pathlib import Path

CREDENTIALS_FILE = ".credentials.json"

_AUTHORIZATION = "authorization"
# The request headers that say who is asking, lowercased.
_CREDENTIAL_HEADERS = frozenset({_AUTHORIZATION, "x-api-key"})
_BEARER = "Bearer "
# How every Claude OAuth access token starts. An API key does not start this
# way, and neither does a token for a gateway.
_OAUTH_TOKEN_PREFIX = "sk-ant-oat"


def credentials_of(headers: Mapping[str, str]) -> dict[str, str]:
    """Return the headers that authenticate a request, names lowercased.

    Args:
        headers: A request's headers.

    Returns:
        The credential headers among them, or {} if there are none.
    """
    return {
        name.lower(): value
        for name, value in headers.items()
        if name.lower() in _CREDENTIAL_HEADERS
    }


def with_credentials(headers: Mapping[str, str], credentials: Mapping[str, str]) -> dict[str, str]:
    """Return `headers` authenticating with `credentials` in place of their own.

    Args:
        headers: A request's headers. Not mutated.
        credentials: The credential headers to send, names lowercased. With
            none to swap in, the headers keep their own.

    Returns:
        A new dict of headers.
    """
    if not credentials:
        return dict(headers)
    kept = {
        name: value for name, value in headers.items() if name.lower() not in _CREDENTIAL_HEADERS
    }
    return {**kept, **credentials}


def oauth_token(credentials: Mapping[str, str]) -> str | None:
    """Return the Claude OAuth access token `credentials` carry.

    Args:
        credentials: Credential headers, names lowercased.

    Returns:
        The token, or None if they authenticate some other way.
    """
    value = credentials.get(_AUTHORIZATION, "")
    if not value.startswith(_BEARER + _OAUTH_TOKEN_PREFIX):
        return None
    return value.removeprefix(_BEARER)


class StoredLogin:
    """The OAuth login Claude Code keeps in its config directory, read for a
    token to replace one the API has stopped accepting.

    Claude Code swaps its access token for a new one every few hours and
    writes the new one here; the old one is revoked on the spot. A session
    sitting idle sends nothing, so the proxy never sees the new token go
    past, and a warm that replays the old one is refused.

    This file is not necessarily the login a session uses: a wrapper can
    point `claude` at another config directory, and a token can come from
    the environment instead. Sending another account's token would rewrite
    the whole prefix under that account, so a token is only ever replaced if
    it was itself in this file while a session was using it (see `observe`).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        # Tokens seen in a session's credentials, each with whether the file
        # held it when it was first seen.
        self._seen: dict[str, bool] = {}

    def read(self) -> str | None:
        """Return the access token the file holds now.

        Returns:
            The token, or None if there is none to read: no such file (macOS
            keeps the login in the Keychain), one that does not parse, or one
            Claude Code emptied when the login ended.
        """
        try:
            stored = json.loads(self._path.read_text())
        except (OSError, ValueError):
            return None
        oauth = stored.get("claudeAiOauth") if isinstance(stored, dict) else None
        token = oauth.get("accessToken") if isinstance(oauth, dict) else None
        return token if isinstance(token, str) and token else None

    def observe(self, credentials: Mapping[str, str]) -> None:
        """Note the credentials a session is sending.

        If their token is the one the file holds, the file is that session's
        login, and its later tokens may stand in for this one. Each token is
        checked against the file once, the first time it is seen.

        Args:
            credentials: The session's credential headers, names lowercased.
        """
        token = oauth_token(credentials)
        if token is not None and token not in self._seen:
            self._seen[token] = token == self.read()

    def successor(self, credentials: Mapping[str, str]) -> dict[str, str] | None:
        """Return credentials carrying the token that replaced a refused one.

        Args:
            credentials: The credential headers the API refused, names
                lowercased.

        Returns:
            The same credentials with the file's current token, or None if
            there is nothing to swap in: their token never came from this
            file, or the file still holds it, or holds none.
        """
        token = oauth_token(credentials)
        if token is None or not self._seen.get(token, False):
            return None
        current = self.read()
        if current is None or current == token:
            return None
        self._seen[current] = True
        return {**credentials, _AUTHORIZATION: _BEARER + current}


def stored_login(environ: Mapping[str, str]) -> StoredLogin:
    """Return the login Claude Code stores when run under `environ`.

    Args:
        environ: The environment `claude` runs in.

    Returns:
        The StoredLogin for CLAUDE_CONFIG_DIR, or for ~/.claude without it.
    """
    custom = environ.get("CLAUDE_CONFIG_DIR")
    config_dir = Path(custom) if custom else Path.home() / ".claude"
    return StoredLogin(config_dir / CREDENTIALS_FILE)
