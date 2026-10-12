import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

_MAX_CYCLES_ERROR = 'max-cycles must be a non-negative integer or "auto"'
_KEEPALIVE_INTERVAL_ERROR = 'keepalive-interval must be a positive integer or "auto"'
_PERCENT_ERROR = "must be a percentage between 0 and 100"

# How long before a one-hour prefix expires the keepalive refreshes it.
KEEPALIVE_MARGIN_SEC = 300

_FIVE_MINUTES_SEC = 300

# The real Anthropic API endpoint, used as the default upstream for the
# detached proxy so it never depends on (and can never inherit) a client's
# ANTHROPIC_BASE_URL that points back at the proxy itself.
DEFAULT_UPSTREAM = "https://api.anthropic.com"

# Upstream Anthropic API the launcher's proxy reverse-proxies to and warm
# requests are sent to. Resolved once from the environment so the reverse
# proxy and the warmer always agree on the same endpoint.
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_UPSTREAM)


def is_loopback_url(url: str) -> bool:
    """Report whether `url`'s host is a loopback address.

    Used to reject a detached-proxy upstream that points back at the local
    proxy, which would make the proxy reverse-proxy to itself. Matches the
    whole loopback range (127.0.0.0/8 and ::1), not just 127.0.0.1, plus the
    literal hostname "localhost".

    Args:
        url: The URL to inspect.

    Returns:
        True iff the URL's host is a loopback address.
    """
    host = urlparse(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class Config:
    idle_threshold_sec: int = 270
    warm_interval_sec: int = 270
    warm_max_cycles: int | None = 2
    subagent_active_window_sec: int = 540
    disabled: bool = False
    # How long a session may sit idle before a detached proxy evicts it.
    # Unused by the launcher (its process dies with claude); only the daemon
    # runs long enough to accumulate stale sessions.
    session_ttl_sec: int = 3600
    # Keep warming after the turn ends, not only while a subagent runs.
    keepalive: bool = False
    # Seconds between keepalive warms on a one-hour prefix; None derives it
    # from the TTL. A five-minute prefix always uses warm_interval_sec.
    keepalive_interval_sec: int | None = None
    # Share of the subscription session limit, as a fraction, that one idle
    # episode may spend on warms.
    ping_budget: float = 0.10
    # Share of every subscription limit, as a fraction, that warms must leave
    # unused, so a compaction and the next real message still fit.
    session_reserve: float = 0.10
    # Compact the conversation once an ended turn can no longer be kept warm.
    compact: bool = False
    # Smallest prompt, in tokens, worth compacting. Below it the summary
    # costs about as much as the rewrite it would save.
    compact_min_tokens: int = 100_000
    # Share of the subscription session limit, as a fraction, at which a
    # kept-alive conversation is paused until the limit resets, so what is
    # left keeps its cache warm. None never pauses.
    pause_at: float | None = None

    def warm_interval_for(self, ttl_sec: int) -> int:
        """Return the seconds between warms for a prefix with this TTL.

        Args:
            ttl_sec: The cached prefix's lifetime, in seconds.

        Returns:
            warm_interval_sec for a five-minute prefix; for a longer one,
            keepalive_interval_sec, or the TTL less a safety margin.
        """
        if ttl_sec <= _FIVE_MINUTES_SEC:
            return self.warm_interval_sec
        if self.keepalive_interval_sec is not None:
            return self.keepalive_interval_sec
        return ttl_sec - KEEPALIVE_MARGIN_SEC


def _parse_max_cycles(value: int | str) -> int | None:
    if value == "auto":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(_MAX_CYCLES_ERROR) from None
    if parsed < 0:
        raise ValueError(_MAX_CYCLES_ERROR)
    return parsed


def _parse_keepalive_interval(value: int | str) -> int | None:
    if value == "auto":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(_KEEPALIVE_INTERVAL_ERROR) from None
    if parsed <= 0:
        raise ValueError(_KEEPALIVE_INTERVAL_ERROR)
    return parsed


def _parse_percent(name: str, value: float) -> float:
    if not 0 <= value <= 100:
        raise ValueError(f"{name} {_PERCENT_ERROR}")
    return value / 100


def _parse_pause_at(value: float) -> float | None:
    # At 100% the API refuses requests itself, so there is nothing to pause.
    fraction = _parse_percent("pause-at", value)
    return None if fraction >= 1 else fraction


def _is_disabled(environ: Mapping[str, str]) -> bool:
    raw = environ.get("CLAUDE_THERMOS_DISABLE", "")
    if not raw:
        return False
    return raw.strip().lower() not in ("0", "false")


def build_config(
    idle_threshold_sec: int,
    warm_interval_sec: int,
    max_cycles_raw: str,
    subagent_active_window_sec: int,
    environ: Mapping[str, str],
    keepalive: bool = False,
    keepalive_interval_raw: str = "auto",
    ping_budget_pct: float = 10.0,
    session_reserve_pct: float = 10.0,
    compact: bool = False,
    compact_min_tokens: int = 100_000,
    pause_at_pct: float = 100.0,
) -> Config:
    """Build a Config from already-resolved option values.

    Flag/env/default precedence is handled by the CLI layer; this parses
    max_cycles, the keepalive interval, and the percentages, and reads the
    disable env var.

    Args:
        idle_threshold_sec: Idle threshold, in seconds, before warming.
        warm_interval_sec: Interval, in seconds, between warming cycles.
        max_cycles_raw: Maximum warming cycles as a string, or "auto".
        subagent_active_window_sec: Subagent active window, in seconds.
        environ: Environment mapping read for the disable flag.
        keepalive: Whether to keep warming after the turn ends.
        keepalive_interval_raw: Seconds between keepalive warms on a
            one-hour prefix as a string, or "auto".
        ping_budget_pct: Percent of the session limit one idle episode may
            spend on warms.
        session_reserve_pct: Percent of every limit that warms leave unused.
        compact: Whether to compact once an ended turn cannot be kept warm.
        compact_min_tokens: Smallest prompt, in tokens, worth compacting.
        pause_at_pct: Percent of the session limit at which a kept-alive
            conversation pauses until the limit resets; 100 never pauses.

    Returns:
        A Config with the resolved and parsed values.
    """
    return Config(
        idle_threshold_sec=idle_threshold_sec,
        warm_interval_sec=warm_interval_sec,
        warm_max_cycles=_parse_max_cycles(max_cycles_raw),
        subagent_active_window_sec=subagent_active_window_sec,
        disabled=_is_disabled(environ),
        keepalive=keepalive,
        keepalive_interval_sec=_parse_keepalive_interval(keepalive_interval_raw),
        ping_budget=_parse_percent("ping-budget", ping_budget_pct),
        session_reserve=_parse_percent("session-reserve", session_reserve_pct),
        compact=compact,
        compact_min_tokens=compact_min_tokens,
        pause_at=_parse_pause_at(pause_at_pct),
    )
