import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

_HEADER_PREFIX = "anthropic-ratelimit-unified-"
_UTILIZATION_HEADER = re.compile(rf"^{_HEADER_PREFIX}(?P<window>.+)-utilization$")
_OVERAGE_STATUS_HEADER = f"{_HEADER_PREFIX}overage-status"

# The rolling window Claude subscriptions call the "session" limit, and the
# name the API gives it when it is the limit that refused a request.
SESSION_WINDOW = "5h"
_SESSION_CLAIM = "five_hour"

_REJECTED = "rejected"
# Overage statuses under which paid extra usage carries requests past a limit.
_OVERAGE_ALLOWED = frozenset({"allowed", "allowed_warning"})


@dataclass(frozen=True)
class LimitWindow:
    """One subscription rate-limit window as the API last reported it.

    `utilization` is the fraction of the window's allowance already used,
    from 0.0 to 1.0. `resets_at` is the Unix time the window rolls over, or
    None if the API did not say."""

    utilization: float
    resets_at: float | None = None
    status: str | None = None

    def utilization_at(self, now: float) -> float:
        """Return the utilization at `now`, which is zero once the window
        has rolled over since this reading was taken."""
        if self.resets_at is not None and now >= self.resets_at:
            return 0.0
        return self.utilization

    def allowed_at(self, now: float) -> bool:
        if self.resets_at is not None and now >= self.resets_at:
            return True
        return self.status != _REJECTED


@dataclass(frozen=True)
class RateLimits:
    """The subscription usage limits reported on one API response.

    Claude subscriptions return an `anthropic-ratelimit-unified-*` header
    family on every /v1/messages response: per window (`5h` for the session
    limit, `7d` for the weekly one) a utilization fraction, a reset time,
    and a status. API-key traffic carries none of them, so `windows` is
    empty there and every budget check that needs a reading is skipped."""

    windows: Mapping[str, LimitWindow]
    overage_status: str | None = None

    def overflow_covered(self) -> bool:
        """Report whether paid extra usage carries requests past a limit, so
        reaching one does not stop the session."""
        return self.overage_status in _OVERAGE_ALLOWED

    def session_utilization(self, now: float) -> float | None:
        """Return the session (5-hour) utilization at `now`, or None when
        the API reported no session window."""
        window = self.windows.get(SESSION_WINDOW)
        return None if window is None else window.utilization_at(now)

    def session_resets_at(self) -> float | None:
        window = self.windows.get(SESSION_WINDOW)
        return None if window is None else window.resets_at

    def headroom(self, now: float) -> float:
        """Return the fraction of allowance left in the tightest window.

        Every window gates a request, so the room for one more is whatever
        the fullest window has left."""
        used = max((window.utilization_at(now) for window in self.windows.values()), default=0.0)
        return max(0.0, 1.0 - used)

    def allowed(self, now: float) -> bool:
        """Report whether every window still accepts requests at `now`."""
        return all(window.allowed_at(now) for window in self.windows.values())


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_rate_limits(headers: Mapping[str, str]) -> RateLimits | None:
    """Parse the subscription rate-limit headers from an API response.

    Args:
        headers: The response headers, matched case-insensitively.

    Returns:
        The reported limits, or None when the response carries no
        utilization header (API-key traffic, or a non-messages response).
    """
    lowered = {name.lower(): value for name, value in headers.items()}
    windows: dict[str, LimitWindow] = {}
    for name, value in lowered.items():
        match = _UTILIZATION_HEADER.match(name)
        utilization = _as_float(value)
        if match is None or utilization is None:
            continue
        window = match["window"]
        windows[window] = LimitWindow(
            utilization=utilization,
            resets_at=_as_float(lowered.get(f"{_HEADER_PREFIX}{window}-reset")),
            status=lowered.get(f"{_HEADER_PREFIX}{window}-status"),
        )
    if not windows:
        return None
    return RateLimits(windows, overage_status=lowered.get(_OVERAGE_STATUS_HEADER))


def rejection_headers(limits: RateLimits, resets_at: float, now: float) -> dict[str, str]:
    """Return response headers reporting the session limit reached until
    `resets_at`.

    They take the shape of the API's own refusal at the five-hour limit,
    which Claude Code answers by waiting in the open session and continuing
    the turn after the reset. Every other window keeps its last reading.

    Args:
        limits: The session's latest limits.
        resets_at: When the session limit resets, as Unix time.
        now: The current Unix time.

    Returns:
        The rate-limit headers, plus retry-after and x-should-retry.
    """
    headers: dict[str, str] = {}
    for name, window in limits.windows.items():
        headers[f"{_HEADER_PREFIX}{name}-utilization"] = str(window.utilization)
        if window.resets_at is not None:
            headers[f"{_HEADER_PREFIX}{name}-reset"] = str(int(window.resets_at))
        if window.status is not None:
            headers[f"{_HEADER_PREFIX}{name}-status"] = window.status
    reset = str(int(resets_at))
    headers.update(
        {
            f"{_HEADER_PREFIX}{SESSION_WINDOW}-status": _REJECTED,
            f"{_HEADER_PREFIX}{SESSION_WINDOW}-reset": reset,
            f"{_HEADER_PREFIX}status": _REJECTED,
            f"{_HEADER_PREFIX}reset": reset,
            f"{_HEADER_PREFIX}representative-claim": _SESSION_CLAIM,
            _OVERAGE_STATUS_HEADER: _REJECTED,
            "retry-after": str(max(0, math.ceil(resets_at - now))),
            "x-should-retry": "false",
        }
    )
    return headers
