import hashlib
import json
from collections.abc import Iterator

_HASH_LENGTH = 8

# Claude Code prepends this attribution block to the system prompt. It carries
# a per-conversation fingerprint that changes when the conversation is
# compacted, and api.anthropic.com strips it before the prompt cache sees it,
# so it is not part of the cacheable prefix and must not split a lineage.
_ATTRIBUTION_PREFIX = "x-anthropic-billing-header:"

_TTL_SEC = {"5m": 300, "1h": 3600}
_DEFAULT_TTL_SEC = 300


def extract_session_id(request_body: dict) -> str | None:
    """Return the session_id embedded in metadata.user_id, or None.

    request_body['metadata']['user_id'] is a JSON string containing
    {"session_id": ...}. Returns None if it is absent or unparseable.

    Args:
        request_body: The decoded request body.

    Returns:
        The embedded session_id, or None if absent or unparseable.
    """
    metadata = request_body.get("metadata")
    if not isinstance(metadata, dict):
        return None
    user_id = metadata.get("user_id")
    if not isinstance(user_id, str):
        return None
    try:
        decoded = json.loads(user_id)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(decoded, dict):
        return None
    session_id = decoded.get("session_id")
    if not isinstance(session_id, str):
        return None
    return session_id


def _system_text(system: str | list[dict] | None) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    return "".join(
        text
        for block in system
        if not (text := block.get("text", "")).startswith(_ATTRIBUTION_PREFIX)
    )


def _iter_cache_ttls(node: object) -> Iterator[int]:
    if isinstance(node, dict):
        cache_control = node.get("cache_control")
        if isinstance(cache_control, dict):
            yield _TTL_SEC.get(cache_control.get("ttl", ""), _DEFAULT_TTL_SEC)
        for value in node.values():
            yield from _iter_cache_ttls(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_cache_ttls(value)


def cache_ttl_sec(request_body: dict) -> int:
    """Return the lifetime, in seconds, of the request's cached prefix.

    Reads the `ttl` of every cache_control breakpoint in tools, system, and
    messages and returns the shortest, since the prefix is only as durable
    as its first tier to expire. A breakpoint without a `ttl` uses the API
    default of five minutes, as does a request with no breakpoints at all.

    Args:
        request_body: The decoded request body.

    Returns:
        300 for a five-minute prefix, 3600 for a one-hour prefix.
    """
    ttls = [
        ttl
        for key in ("tools", "system", "messages")
        for ttl in _iter_cache_ttls(request_body.get(key))
    ]
    return min(ttls, default=_DEFAULT_TTL_SEC)


class LineageId(str):
    """Identifier for a cacheable prefix lineage.

    A distinct type (a ``str`` subclass) so it type-checks apart from
    arbitrary strings, while remaining usable as a dict key and serializing
    straight to JSON. The value is a short hex digest of the prefix identity:
    model + sorted tool names + concatenated system text, leaving out Claude
    Code's attribution block."""

    __slots__ = ()

    @classmethod
    def from_request_body(cls, request_body: dict) -> "LineageId":
        """Derive the lineage id from a request body.

        Two requests with the same model, tool-name set, and system text
        produce equal ids; a different model, tool set, or system text
        produces a different one. Tool ordering does not affect the result.

        Args:
            request_body: The decoded request body.

        Returns:
            The lineage id for the request's cacheable prefix.
        """
        model = request_body.get("model", "")
        tool_names = sorted(tool.get("name", "") for tool in request_body.get("tools", []))
        system_text = _system_text(request_body.get("system"))
        identity = json.dumps([model, tool_names, system_text])
        digest = hashlib.sha256(identity.encode()).hexdigest()[:_HASH_LENGTH]
        return cls(digest)
