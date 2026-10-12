import asyncio
import copy
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypeGuard

import httpx

from claude_thermos.config import ANTHROPIC_BASE_URL, Config
from claude_thermos.credentials import StoredLogin, credentials_of, with_credentials
from claude_thermos.eventlog import EventLog, EventType
from claude_thermos.limits import RateLimits, parse_rate_limits
from claude_thermos.lineage import LineageId
from claude_thermos.state import LineageState, SessionState
from claude_thermos.usage import Usage, parse_usage_json

_DEFAULT_POLL_INTERVAL_SEC = 5.0
_WARM_MAX_TOKENS = 1
_WARM_TIMEOUT_SEC = 60.0
# Pause before retrying a warm that failed, while the prefix is still alive.
_RETRY_INTERVAL_SEC = 60.0
# Slack for comparing utilization fractions the API reports to two decimals.
_EPSILON = 1e-9

# Captured request headers that must not be replayed on the warm request: the
# host header points at the local proxy and content-length describes the
# original body, so both are wrong for a direct call with a rewritten body.
# httpx recomputes them for us once they are removed.
_STRIPPED_HEADERS = frozenset({"host", "content-length"})


def build_warm_request(body: dict) -> dict:
    """Turn a stored real request body into a minimal-output warm request.

    The entire cacheable prefix is preserved exactly — model, system,
    tools, messages, thinking, tool_choice, and context_management,
    including every cache_control breakpoint — so the warm request reads
    and refreshes every cache tier the real request uses, message history
    included. The prompt cache keys the message tier on thinking state and
    tool_choice, so stripping either would leave the real lineage's message
    cache to expire; only tools and system survive that. Keeping them
    intact is what makes the warm preserve the whole prefix.

    Only generation is neutralized: max_tokens is capped at 1, streaming is
    disabled, and output_config is removed. max_tokens is 1 rather than 0
    because a zero-token request is rejected when thinking is enabled, and
    thinking must stay enabled to keep the message tier warm. The single
    generated token is discarded — the cache is written during prefill,
    before generation. output_config is not part of the cache key, and
    dropping it keeps a structured-output format from constraining the
    one-token reply.

    Args:
        body: The stored last real main request body. Not mutated.

    Returns:
        A new request body suitable for cache warming.
    """
    warm = copy.deepcopy(body)
    warm["max_tokens"] = _WARM_MAX_TOKENS
    warm["stream"] = False
    warm.pop("output_config", None)
    return warm


# Why an idle episode stopped warming. The value is the `reason` field of the
# cap_reached event.
STOP_MAX_CYCLES = "max_cycles"
STOP_PING_BUDGET = "ping_budget_spent"
STOP_SESSION_LOW = "session_limit_low"
STOP_LIMIT_REACHED = "limit_reached"
STOP_CACHE_EXPIRED = "cache_expired"
STOP_CACHE_MISS = "cache_miss"
STOP_COMPACTED = "compacted"
STOP_UNAUTHORIZED = "unauthorized"

# Stops that leave a live cached prefix behind, which a compaction can still
# read cheaply before it expires.
_COMPACTABLE_STOPS = frozenset({STOP_MAX_CYCLES, STOP_PING_BUDGET, STOP_SESSION_LOW})

# Why a compaction was not requested. The value is the `reason` field of the
# compact_skipped event.
SKIP_TURN_IN_PROGRESS = "turn_in_progress"
SKIP_CONTEXT_SMALL = "context_small"

# A compactor asks Claude Code to compact a session. It returns None once the
# request is made, or a short reason it could not be.
Compactor = Callable[[str], str | None]

# Why a pause ended. The value is the `reason` field of the pause_lifted event.
LIFT_RESET = "reset"
LIFT_PROMPT = "prompt"


def update_pause(state: SessionState, eventlog: EventLog, config: Config, now: float) -> bool:
    """Start or end a session's pause, logging the change, and report whether
    it is paused.

    Pausing exists to spend what is left of the session limit on keepalive
    warms, so it applies only when keepalive is on.

    Args:
        state: The session's state.
        eventlog: The session's event log.
        config: The warmer configuration.
        now: Current time, in seconds.

    Returns:
        True iff the session's conversation is paused at `now`.
    """
    if not config.keepalive:
        return False
    was_paused = state.paused_until is not None
    paused = state.update_pause(now, config.pause_at)
    lineage_id = state.main_lineage_id() or LineageId("")
    if paused and not was_paused:
        eventlog.emit(
            EventType.PAUSED,
            lineage_id=lineage_id,
            resumes_at=state.paused_until,
            session_utilization=_session_utilization(state, now),
        )
    elif was_paused and not paused:
        eventlog.emit(EventType.PAUSE_LIFTED, lineage_id=lineage_id, reason=LIFT_RESET)
    return paused


@dataclass
class _Episode:
    """Per-session warming state for a single idle episode: the main lineage
    waiting on a subagent, or on the user once the turn has ended. Reset when
    the main lineage resumes or warming no longer applies."""

    warm_count: int = 0
    last_warm_sent: float = float("-inf")
    last_attempt: float = float("-inf")
    in_progress: bool = False
    cap_emitted: bool = False
    main_request_count: int | None = None
    # Fraction of the subscription session limit this episode's warms used.
    spent: float = 0.0
    # The credentials the API last refused. Nothing more is sent until the
    # session's are different ones.
    rejected: dict[str, str] | None = None


@dataclass(frozen=True)
class _Failure:
    """A warm that brought back no usage: what went wrong, and the HTTP
    status if the API answered at all."""

    error: str
    status: int | None = None


class Warmer:
    """Periodic driver that warms the main lineage's cache prefix while it
    sits idle.

    `tick` makes the warm decision for one session at an injected `now`;
    `run` polls all active sessions on an interval. Per-session episode
    state enforces the interval, the spending caps, one-warm-at-a-time, and
    reset on resume.

    An episode ends once it has sent its cap of warms, or sooner, when
    the API reports subscription limits, once the warms have used up the
    ping budget or would cut into the session reserve. If the turn has
    ended and a compactor is configured, a large enough conversation is
    then compacted while its prefix is still cached, so the eventual resume
    re-reads a summary rather than rewriting the whole history.

    Warm requests are sent directly to the Anthropic API through the owned
    httpx client (injectable for testing), never through the proxy, so they
    can never touch Claude Code's traffic.

    A warm authenticates with the newest credentials any request in the
    session carried. If the API refuses them, the warm is sent again with
    the token that replaced them in Claude Code's stored login, when one is
    configured and has one; otherwise the episode stops until the session
    sends different credentials.
    """

    def __init__(
        self,
        config: Config,
        base_url: str = ANTHROPIC_BASE_URL,
        client: httpx.AsyncClient | None = None,
        compactor: Compactor | None = None,
        login: StoredLogin | None = None,
    ) -> None:
        self._config = config
        self._base_url = base_url.rstrip("/")
        self._client = (
            client
            if client is not None
            else httpx.AsyncClient(timeout=httpx.Timeout(_WARM_TIMEOUT_SEC))
        )
        self._compactor = compactor
        self._login = login
        self._episodes: dict[str, _Episode] = {}

    async def tick(self, state: SessionState, eventlog: EventLog, now: float) -> None:
        """Evaluate one session and warm the main lineage if warranted.

        Warms only when the main lineage is idle, a subagent is active or
        keepalive covers the session, the prefix's warm interval has passed since the
        last warm-or-real main request, and the episode may still spend.
        Emits warm_fired / warm_result / warm_error on a warm, cap_reached
        when the episode stops, compact_requested / compact_skipped when a
        stop calls for a compaction, and resume_detected when the main
        lineage sends a new real request. An episode stopped because its
        credentials were refused starts again if the session's change.

        Args:
            state: The session's state.
            eventlog: The session's event log.
            now: Current time, in seconds.
        """
        main = state.main_lineage()
        if main is None:
            return
        if self._login is not None:
            self._login.observe(state.credentials)
        episode = self._episodes.setdefault(state.session_id, _Episode())
        was_paused = state.paused_until is not None
        paused = update_pause(state, eventlog, self._config, now)
        if was_paused and not paused:
            # The session limit has reset, so the rest of this idle spell
            # draws on a new window: fresh caps, same cached prefix.
            episode.warm_count = 0
            episode.spent = 0.0
            episode.cap_emitted = False

        if (
            episode.main_request_count is not None
            and main.request_count > episode.main_request_count
        ):
            eventlog.emit(EventType.RESUME_DETECTED, lineage_id=main.lineage_id)
            self._reset(episode)
            return

        idle = state.is_main_idle(now, self._config.idle_threshold_sec)
        active = (self._config.keepalive and state.live) or state.subagent_active(
            now, self._config.subagent_active_window_sec
        )
        if not (idle and active):
            if not active:
                self._reset(episode)
            return

        if episode.main_request_count is None:
            episode.main_request_count = main.request_count

        if episode.rejected is not None:
            if state.credentials == episode.rejected:
                return
            # Another lineage has sent different credentials since the API
            # refused the last ones, so the episode has something new to try.
            episode.rejected = None
            episode.cap_emitted = False
            episode.last_attempt = float("-inf")

        if episode.cap_emitted or episode.in_progress or main.last_request_body is None:
            return

        last_refresh = max(main.last_request_sent, episode.last_warm_sent)
        if now - last_refresh < self._config.warm_interval_for(main.cache_ttl_sec):
            return
        if now - episode.last_attempt < _RETRY_INTERVAL_SEC:
            return

        stop = self._stop_reason(state, main, episode, now, last_refresh, paused)
        if stop is None:
            stop = await self._fire(state, episode, main, main.last_request_body, eventlog, now)
        if stop is not None:
            self._stop(state, main, episode, eventlog, stop, now)

    def _stop_reason(
        self,
        state: SessionState,
        main: LineageState,
        episode: _Episode,
        now: float,
        last_refresh: float,
        paused: bool = False,
    ) -> str | None:
        """Decide whether the episode may send another warm.

        While the conversation is paused, what is left of the session limit
        was set aside for exactly these warms, so only the limit itself
        stops them: not the cycle cap, the ping budget, or the reserve.

        Args:
            state: The session's state.
            main: The main lineage's state.
            episode: The session's episode.
            now: Current time, in seconds.
            last_refresh: When the prefix was last written or read.
            paused: Whether the conversation is paused.

        Returns:
            None to warm, or the reason the episode must stop.
        """
        if state.main_compacted():
            return STOP_COMPACTED
        if now - last_refresh >= main.cache_ttl_sec:
            # The prefix is gone; a "warm" now would pay to rewrite all of it.
            return STOP_CACHE_EXPIRED

        max_cycles = self._config.warm_max_cycles
        if not paused and max_cycles is not None and episode.warm_count >= max_cycles:
            return STOP_MAX_CYCLES

        limits = state.limits
        used = None if limits is None else limits.session_utilization(now)
        if limits is None or used is None:
            return None
        if not limits.allowed(now):
            return STOP_LIMIT_REACHED
        if paused:
            return None
        next_warm = episode.spent / episode.warm_count if episode.warm_count else 0.0
        budget = self._config.ping_budget
        if budget <= 0 or episode.spent + next_warm > budget + _EPSILON:
            return STOP_PING_BUDGET
        if limits.headroom(now) - next_warm < self._config.session_reserve - _EPSILON:
            return STOP_SESSION_LOW
        return None

    def _stop(
        self,
        state: SessionState,
        main: LineageState,
        episode: _Episode,
        eventlog: EventLog,
        reason: str,
        now: float,
    ) -> None:
        """End the episode and, where it helps, compact the conversation.

        Compaction is requested only when the stop left a live prefix
        behind, the session is open in a terminal, its turn has ended, its
        prompt is large enough to be worth summarizing, and a compactor is
        configured. The
        session is marked compacted as soon as the request is made, so
        nothing warms a prefix that is about to be summarized away.

        Args:
            state: The session's state.
            main: The main lineage's state.
            episode: The session's episode.
            eventlog: The session's event log.
            reason: Why the episode stopped.
            now: Current time, in seconds.
        """
        episode.cap_emitted = True
        eventlog.emit(
            EventType.CAP_REACHED,
            lineage_id=main.lineage_id,
            cycle=episode.warm_count,
            reason=reason,
            **_limit_fields(state, episode, now),
        )
        if reason not in _COMPACTABLE_STOPS:
            return
        if not self._config.compact or self._compactor is None or not state.live:
            return
        skipped = None
        if not state.main_turn_ended():
            skipped = SKIP_TURN_IN_PROGRESS
        elif main.last_input_tokens < self._config.compact_min_tokens:
            skipped = SKIP_CONTEXT_SMALL
        else:
            skipped = self._compactor(state.session_id)
        if skipped is not None:
            eventlog.emit(EventType.COMPACT_SKIPPED, lineage_id=main.lineage_id, reason=skipped)
            return
        state.on_compacted(now)
        eventlog.emit(EventType.COMPACT_REQUESTED, lineage_id=main.lineage_id, reason=reason)

    async def _fire(
        self,
        state: SessionState,
        episode: _Episode,
        main: LineageState,
        body: dict,
        eventlog: EventLog,
        now: float,
    ) -> str | None:
        """Send one warm and account for it.

        A warm that read the prefix counts as a cycle and restarts the
        interval. One that failed is retried after a short pause, until the
        prefix expires. One that wrote more than it read found the prefix
        already gone, so the episode stops rather than pay for more.

        A warm the API refuses as unauthorized is sent again at once with
        the token that replaced its own in Claude Code's stored login.
        Without one the episode stops: the same credentials would only be
        refused again.

        Args:
            state: The session's state.
            episode: The session's episode.
            main: The main lineage's state.
            body: The stored real request body to warm from.
            eventlog: The session's event log.
            now: Current time, in seconds.

        Returns:
            None normally, or the reason the episode must stop.
        """
        episode.in_progress = True
        try:
            cycle = episode.warm_count + 1
            eventlog.emit(
                EventType.WARM_FIRED,
                lineage_id=main.lineage_id,
                cycle=cycle,
                idle_for_sec=round(now - main.last_response_end, 3),
                subagent_active=state.subagent_active(now, self._config.subagent_active_window_sec),
            )
            used_before = None if state.limits is None else state.limits.session_utilization(now)
            episode.last_attempt = now
            headers = with_credentials(_replay_headers(main.auth_headers), state.credentials)
            result, limits = await self._send(body, headers)
            if _unauthorized(result) and self._login is not None:
                replaced = self._login.successor(credentials_of(headers))
                if replaced is not None:
                    eventlog.emit(
                        EventType.WARM_ERROR,
                        lineage_id=main.lineage_id,
                        cycle=cycle,
                        error=result.error,
                        retried=True,
                    )
                    state.credentials = replaced
                    headers = with_credentials(headers, replaced)
                    result, limits = await self._send(body, headers)
            if limits is not None:
                state.limits = limits
                used_after = limits.session_utilization(now)
                if used_before is not None and used_after is not None:
                    episode.spent += max(0.0, used_after - used_before)
            if isinstance(result, _Failure):
                eventlog.emit(
                    EventType.WARM_ERROR,
                    lineage_id=main.lineage_id,
                    cycle=cycle,
                    error=result.error,
                )
                if _unauthorized(result):
                    episode.rejected = credentials_of(headers)
                    return STOP_UNAUTHORIZED
                return None
            refreshed = result.cache_read >= result.cache_creation and result.cache_read > 0
            eventlog.emit(
                EventType.WARM_RESULT,
                lineage_id=main.lineage_id,
                cycle=cycle,
                usage=_usage_dict(result),
                ttl_refreshed=refreshed,
                ttl_sec=main.cache_ttl_sec,
                **_limit_fields(state, episode, now),
            )
            episode.warm_count += 1
            episode.last_warm_sent = now
            episode.last_attempt = float("-inf")
            return None if refreshed else STOP_CACHE_MISS
        finally:
            episode.in_progress = False

    async def _send(self, body: dict, headers: dict) -> tuple[Usage | _Failure, RateLimits | None]:
        """POST a warm request built from `body`, failing quiet.

        Any failure — a non-2xx status, a network error, a timeout, or an
        unexpected error while building or parsing — is swallowed and
        surfaced as a _Failure, never raised, so a failed warm can
        neither disturb real traffic nor kill the warming loop. Cancellation
        propagates so the task can still be stopped cleanly.

        Args:
            body: The stored real request body to warm from.
            headers: The headers to send.

        Returns:
            The response Usage on a 2xx response, or a short description of
            the error on any failure; and the subscription limits the
            response reported, if it reported any.
        """
        warm_body = build_warm_request(body)
        limits = None
        try:
            response = await self._client.post(
                f"{self._base_url}/v1/messages",
                json=warm_body,
                headers=headers,
            )
            limits = parse_rate_limits(response.headers)
            response.raise_for_status()
            return parse_usage_json(response.json()), limits
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            return _Failure(f"HTTP {status}: {_api_error_message(exc.response)}", status), limits
        except Exception as exc:
            return _Failure(f"{type(exc).__name__}: {exc}"), limits

    def _reset(self, episode: _Episode) -> None:
        episode.warm_count = 0
        episode.last_warm_sent = float("-inf")
        episode.last_attempt = float("-inf")
        episode.cap_emitted = False
        episode.main_request_count = None
        episode.spent = 0.0
        episode.rejected = None

    def forget(self, session_id: str) -> None:
        """Drop any per-session episode state for `session_id`.

        Called when a session is evicted so its episode entry does not leak
        for the life of a long-running daemon.

        Args:
            session_id: The session to forget.
        """
        self._episodes.pop(session_id, None)

    def is_warming(self, session_id: str) -> bool:
        """Report whether a warm request is currently in flight for a session.

        The reaper consults this so it never closes a session's event log
        while `_fire` is mid-warm (awaiting the HTTP call, then emitting).

        Args:
            session_id: The session to check.

        Returns:
            True iff the session has an episode with a warm in progress.
        """
        episode = self._episodes.get(session_id)
        return episode is not None and episode.in_progress

    async def run(
        self,
        sessions_provider: Callable[[], Iterable[tuple[SessionState, EventLog]]],
        clock: Callable[[], float] = time.time,
        poll_interval_sec: float = _DEFAULT_POLL_INTERVAL_SEC,
    ) -> None:
        """Poll all active sessions on an interval, warming as warranted.

        Stops cleanly when cancelled.

        Args:
            sessions_provider: Returns the currently active
                (SessionState, EventLog) pairs.
            clock: Returns the current time, in seconds; injectable.
            poll_interval_sec: Seconds to sleep between polls.
        """
        while True:
            now = clock()
            for state, eventlog in sessions_provider():
                await self.tick(state, eventlog, now)
            await asyncio.sleep(poll_interval_sec)


def _limit_fields(state: SessionState, episode: _Episode, now: float) -> dict:
    """Return the event fields describing subscription usage, or none when
    the API has not reported any.

    Args:
        state: The session's state.
        episode: The session's episode.
        now: Current time, in seconds.

    Returns:
        session_utilization and ping_spent, both as fractions, or {}.
    """
    used = _session_utilization(state, now)
    if used is None:
        return {}
    return {"session_utilization": used, "ping_spent": round(episode.spent, 4)}


def _session_utilization(state: SessionState, now: float) -> float | None:
    used = None if state.limits is None else state.limits.session_utilization(now)
    return None if used is None else round(used, 4)


def _unauthorized(result: Usage | _Failure) -> TypeGuard[_Failure]:
    """Report whether the API refused a warm's credentials."""
    return isinstance(result, _Failure) and result.status == httpx.codes.UNAUTHORIZED


def _replay_headers(headers: dict) -> dict:
    """Drop headers that must not be replayed on the direct warm request.

    Args:
        headers: The captured request headers.

    Returns:
        A new dict without the stripped headers, matched case-insensitively.
    """
    return {k: v for k, v in headers.items() if k.lower() not in _STRIPPED_HEADERS}


def _api_error_message(response: httpx.Response) -> str:
    """Extract the Anthropic API error message from a failed response.

    The API returns errors as {"error": {"message": ...}}. Falls back to
    the raw response text when the body is not the expected shape.

    Args:
        response: The failed HTTP response.

    Returns:
        The API's error message, or the raw response text.
    """
    try:
        error = response.json().get("error", {})
        message = error.get("message")
        if isinstance(message, str):
            return message
    except (ValueError, AttributeError):
        pass
    return response.text


def _usage_dict(usage: Usage) -> dict:
    return {
        "uncached_input": usage.uncached_input,
        "cache_read": usage.cache_read,
        "cache_creation": usage.cache_creation,
        "output": usage.output,
    }
