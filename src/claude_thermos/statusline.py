import json
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from claude_thermos.eventlog import EVENTS_FILE, EventType, default_log_root

_SECONDS_PER_MINUTE = 60


def _warm_expiry(events: Iterable[Mapping[str, Any]]) -> float | None:
    """Return when the last successful warm's TTL runs out, or None.

    A compaction replaces the prefix a warm kept alive, so only warms after
    the most recent compaction count."""
    expires_at: float | None = None
    for event in events:
        name = event.get("event")
        if name == EventType.COMPACTED:
            expires_at = None
        elif name == EventType.WARM_RESULT and event.get("ttl_refreshed"):
            ts, ttl = event.get("ts"), event.get("ttl_sec")
            if isinstance(ts, int | float) and isinstance(ttl, int | float):
                expires_at = ts + ttl
    return expires_at


def _paused_until(events: Iterable[Mapping[str, Any]]) -> float | None:
    """Return when the last pause is due to end, or None if none was left
    standing."""
    resumes_at: float | None = None
    for event in events:
        name = event.get("event")
        if name == EventType.PAUSED:
            value = event.get("resumes_at")
            resumes_at = value if isinstance(value, int | float) else None
        elif name == EventType.PAUSE_LIFTED:
            resumes_at = None
    return resumes_at


def read_events(session_id: str, root: Path | None = None) -> list[dict[str, Any]]:
    """Read a session's event log, skipping lines that do not parse.

    Returns an empty list when the session has no log, or when `session_id`
    is not a plain name and so could point outside the log directory."""
    if session_id in ("", ".", "..") or Path(session_id).name != session_id:
        return []
    path = (root if root is not None else default_log_root()) / session_id / EVENTS_FILE
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def cache_status(status: Mapping[str, Any], events: Sequence[Mapping[str, Any]], now: float) -> str:
    """Describe the main conversation's prompt cache for the status line.

    Claude Code's own `prompt_cache.expires_at` counts only the requests it
    sent, so it goes cold an hour after the last turn even when a warm has
    kept the cache alive. The real expiry is the later of that and the last
    warm's. While thermos holds the conversation paused for the session
    limit, the line says until when, since Claude Code reports that pause as
    an ordinary usage limit.

    Args:
        status: The JSON Claude Code passes to a status line command.
        events: The session's thermos event log.
        now: The current Unix time.

    Returns:
        The status text, or an empty string before the conversation has
        reported any cache use.
    """
    prompt_cache = status.get("prompt_cache")
    if not isinstance(prompt_cache, Mapping) or not prompt_cache.get("caching_observed"):
        return ""
    reported = prompt_cache.get("expires_at")
    own_expiry = float(reported) if isinstance(reported, int | float) else None
    warm_expiry = _warm_expiry(events)
    kept_by_warm = warm_expiry is not None and (own_expiry is None or warm_expiry > own_expiry)
    expires_at = warm_expiry if kept_by_warm else own_expiry
    if expires_at is None or expires_at <= now:
        cache = "cache cold"
    else:
        minutes = int((expires_at - now) // _SECONDS_PER_MINUTE)
        left = f"{minutes}m left" if minutes else "<1m left"
        cache = f"cache warm · {left} · kept by thermos" if kept_by_warm else f"cache warm · {left}"
    resumes_at = _paused_until(events)
    if resumes_at is None or resumes_at <= now:
        return cache
    return f"paused by thermos until {clock_time(resumes_at)} · {cache}"


def clock_time(ts: float) -> str:
    """Format a Unix time as local hours and minutes."""
    return time.strftime("%H:%M", time.localtime(ts))


def render(stdin_text: str, root: Path | None = None, now: float | None = None) -> str:
    """Build the status line from the JSON Claude Code sends on stdin.

    Never raises: a status line command that fails leaves the line blank,
    so input that does not parse renders as an empty string."""
    try:
        status = json.loads(stdin_text)
    except ValueError:
        return ""
    if not isinstance(status, dict):
        return ""
    session_id = status.get("session_id")
    events = read_events(session_id, root) if isinstance(session_id, str) else []
    return cache_status(status, events, time.time() if now is None else now)


def main() -> None:
    """Entry point for `claude-thermos-statusline`, set as Claude Code's
    statusLine command.

    It is a separate script from `claude-thermos` so each refresh imports
    only this module, not the proxy."""
    print(render(sys.stdin.read()))
