from collections.abc import Mapping
from dataclasses import dataclass, field

from claude_thermos.credentials import credentials_of
from claude_thermos.limits import SESSION_WINDOW, LimitWindow, RateLimits
from claude_thermos.lineage import LineageId, cache_ttl_sec

END_TURN = "end_turn"

# Time Claude Code may take to send its first request once a usage limit
# resets. A stop that ends sooner than the cache TTL less this leaves the
# cache alive without any help.
_RESUME_LEEWAY_SEC = 300
# Slack for comparing utilization fractions the API reports to two decimals.
_EPSILON = 1e-9


@dataclass
class LineageState:
    lineage_id: LineageId
    first_seen: float
    last_request_sent: float
    last_response_end: float = float("-inf")
    in_flight: int = 0
    request_count: int = 0
    auth_headers: dict[str, str] = field(default_factory=dict)
    last_request_body: dict | None = None
    has_tools: bool = False
    cache_ttl_sec: int = 0
    last_stop_reason: str | None = None
    # Input tokens of the last response, cached or not: the size of the prefix.
    last_input_tokens: int = 0


class SessionState:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._lineages: dict[LineageId, LineageState] = {}
        self._main_id: LineageId | None = None
        # Latest subscription limits seen on any of this session's responses,
        # or None until one reports them (never, on API-key traffic).
        self.limits: RateLimits | None = None
        # The credential headers of the newest request on any lineage, side
        # requests included. Claude Code replaces its OAuth token every few
        # hours and the old one stops working, so the copy the main lineage
        # sent with its last request can be dead while this one is current.
        self.credentials: dict[str, str] = {}
        # When Claude Code last reported compacting this session, or -inf.
        self.compacted_at: float = float("-inf")
        # Whether a hook reported this session open in a terminal. Only a
        # live session is kept warm after its turn ends: anything else that
        # passes through the proxy (a `claude -p` child, a session since
        # cleared) has no one coming back to it.
        self.live: bool = False
        # While set, the conversation is paused until this Unix time (when
        # the session limit resets) so the rest of the limit keeps its cache
        # warm. See `update_pause`.
        self.paused_until: float | None = None
        # A prompt sent above the pause threshold lets the conversation run
        # until the session limit resets, at this Unix time.
        self._pause_waived_until: float = float("-inf")

    def _session_window(self) -> LimitWindow | None:
        return None if self.limits is None else self.limits.windows.get(SESSION_WINDOW)

    def _pause_deadline(self, now: float, pause_at: float) -> float | None:
        """Return when the session limit resets, if the conversation should
        pause until then, or None.

        It pauses once the session limit is used past `pause_at`, unless
        paid extra usage would carry it past the limit, a prompt waived the
        pause, or the limit resets soon enough that the cache outlives the
        stop anyway. Only a session open in a terminal pauses: anything else
        is not kept warm, so stopping it would buy nothing.
        """
        window = self._session_window()
        main = self.main_lineage()
        if not self.live or window is None or window.resets_at is None or main is None:
            return None
        if self.limits is not None and self.limits.overflow_covered():
            return None
        if now < self._pause_waived_until or window.utilization_at(now) < pause_at - _EPSILON:
            return None
        if window.resets_at - now <= main.cache_ttl_sec - _RESUME_LEEWAY_SEC:
            return None
        return window.resets_at

    def update_pause(self, now: float, pause_at: float | None) -> bool:
        """Start or end the pause, and report whether it holds at `now`.

        A pause holds until the session limit resets, even if the reset
        later comes close enough that a new one would not start: Claude
        Code is by then waiting out the limit, and typing into that wait
        would cancel it.

        Args:
            now: Current time, in seconds.
            pause_at: Share of the session limit, as a fraction, past which
                the conversation pauses; None never pauses.

        Returns:
            True iff the conversation is paused.
        """
        if self.paused_until is not None and now >= self.paused_until:
            self.paused_until = None
        if self.paused_until is None and pause_at is not None:
            self.paused_until = self._pause_deadline(now, pause_at)
        return self.paused_until is not None

    def waive_pause(self, now: float, pause_at: float | None) -> bool:
        """Let the conversation run past the pause threshold until the
        session limit resets, because the user sent a prompt.

        A prompt sent below the threshold waives nothing: the turn it starts
        still pauses if it crosses the threshold while the user is away.

        Args:
            now: Current time, in seconds.
            pause_at: The pause threshold, as for `update_pause`.

        Returns:
            True iff a pause had already started and is now lifted.
        """
        was_paused = self.paused_until is not None and now < self.paused_until
        window = self._session_window()
        if not self.update_pause(now, pause_at) or window is None or window.resets_at is None:
            return False
        self._pause_waived_until = window.resets_at
        self.paused_until = None
        return was_paused

    def is_side_request(self, lineage: LineageId, body: dict, now: float) -> bool:
        """Report whether a request only borrows its lineage's prefix.

        Claude Code sends requests that reuse a conversation's exact prefix
        without continuing it: compaction, prompt suggestions, forked
        subagents. On a subscription those carry a five-minute cache TTL
        while the conversation itself carries one hour, so a request whose
        TTL is shorter than its lineage's is one of them. Replaying such a
        body would warm a prefix the conversation never sends again.

        The shorter TTL only marks a side request while the lineage's own
        prefix can still be alive. Past that, the conversation itself has
        been downgraded to the shorter TTL (Claude Code does this once a
        subscription starts drawing on usage credits) and the request is
        the conversation again.

        Args:
            lineage: The request's lineage id.
            body: The decoded request body.
            now: Current time, in seconds.

        Returns:
            True iff the request should leave the lineage's state alone.
        """
        state = self._lineages.get(lineage)
        if state is None:
            return False
        if cache_ttl_sec(body) >= state.cache_ttl_sec:
            return False
        return now - state.last_request_sent < state.cache_ttl_sec

    def note_credentials(self, headers: Mapping[str, str]) -> None:
        """Record the credentials a request carried as the session's newest.

        A request that carries none leaves the last ones in place.

        Args:
            headers: The request's headers.
        """
        credentials = credentials_of(headers)
        if credentials:
            self.credentials = credentials

    def on_request(self, lineage: LineageId, body: dict, headers: dict, now: float) -> None:
        """Create or update the lineage's state for a new request.

        Bumps in_flight and request_count, sets last_request_sent, stores
        the latest auth headers and last_request_body, records
        has_tools = len(tools) > 0, and records the prefix's cache TTL and
        the session's newest credentials.
        Callers skip side requests (see `is_side_request`).

        Args:
            lineage: The request's lineage id.
            body: The decoded request body.
            headers: The request's headers.
            now: Current time, in seconds.
        """
        state = self._lineages.get(lineage)
        if state is None:
            state = LineageState(lineage_id=lineage, first_seen=now, last_request_sent=now)
            self._lineages[lineage] = state
        state.in_flight += 1
        state.request_count += 1
        state.last_request_sent = now
        state.auth_headers = dict(headers)
        self.note_credentials(headers)
        state.last_request_body = body
        state.has_tools = len(body.get("tools", [])) > 0
        state.cache_ttl_sec = cache_ttl_sec(body)
        if state.has_tools and lineage != self._main_id and self._succeeds_main(state):
            self._main_id = lineage

    def _succeeds_main(self, candidate: LineageState) -> bool:
        """Report whether a lineage that just sent a request takes over as main.

        The first lineage with tools becomes main. After a compaction the
        conversation can come back under a new lineage, because Claude Code
        rebuilds the request's tools and system prompt, so the first lineage
        to continue it takes over: one with tools, a cache TTL at least as
        long as the old main's, and arriving before the old main sends again.
        The TTL test keeps a subagent started in that gap from taking over on
        a subscription, where subagents carry five minutes against the
        conversation's hour.
        """
        main = self.main_lineage()
        if main is None:
            return True
        return (
            self.compacted_at >= main.last_request_sent
            and candidate.cache_ttl_sec >= main.cache_ttl_sec
        )

    def on_response(
        self,
        lineage: LineageId,
        now: float,
        stop_reason: str | None = None,
        input_tokens: int | None = None,
    ) -> None:
        """Record a completed response for the lineage.

        Decrements in_flight (floored at 0), sets last_response_end, and
        records the response's stop reason and prompt size.

        Args:
            lineage: The response's lineage id.
            now: Current time, in seconds.
            stop_reason: The response's stop_reason, or None if unknown.
            input_tokens: The response's total input tokens, or None if
                unknown, which leaves the last known size in place.
        """
        state = self._lineages.get(lineage)
        if state is None:
            return
        state.in_flight = max(0, state.in_flight - 1)
        state.last_response_end = now
        state.last_stop_reason = stop_reason
        if input_tokens is not None:
            state.last_input_tokens = input_tokens

    def on_compacted(self, now: float) -> None:
        """Record that Claude Code compacted this session's conversation.

        Args:
            now: Current time, in seconds.
        """
        self.compacted_at = now

    def main_lineage_id(self) -> LineageId | None:
        """Return the id of the lineage carrying the conversation.

        That is the first lineage with tools, or the one that continued the
        conversation after a compaction (see `_succeeds_main`).

        Returns:
            The main lineage id, or None if none exists yet.
        """
        return self._main_id

    def main_lineage(self) -> LineageState | None:
        """Return the main lineage's state, or None if none exists yet.

        Returns:
            The LineageState of the main lineage, or None.
        """
        if self._main_id is None:
            return None
        return self._lineages[self._main_id]

    def is_main_idle(self, now: float, idle_threshold_sec: int) -> bool:
        """Report whether the main lineage is idle.

        Args:
            now: Current time, in seconds.
            idle_threshold_sec: Seconds of inactivity that count as idle.

        Returns:
            True iff a main lineage exists, its in_flight is 0, and
            now - last_response_end >= idle_threshold_sec.
        """
        main_id = self.main_lineage_id()
        if main_id is None:
            return False
        main = self._lineages[main_id]
        return main.in_flight == 0 and now - main.last_response_end >= idle_threshold_sec

    def main_turn_ended(self) -> bool:
        """Report whether the main lineage is waiting on the user.

        Returns:
            True iff a main lineage exists, nothing is in flight on it, and
            its last response ended the turn rather than calling a tool.
        """
        main = self.main_lineage()
        return main is not None and main.in_flight == 0 and main.last_stop_reason == END_TURN

    def main_compacted(self) -> bool:
        """Report whether the conversation was compacted after the main
        lineage's last request, leaving that request's prefix obsolete.

        Returns:
            True iff a compaction was reported since the main lineage last
            sent a request.
        """
        main = self.main_lineage()
        return main is not None and self.compacted_at >= main.last_request_sent

    def subagent_active(self, now: float, window_sec: int) -> bool:
        """Report whether any subagent (non-main) lineage is active.

        Args:
            now: Current time, in seconds.
            window_sec: Recency window, in seconds, for recent activity.

        Returns:
            True iff any non-main lineage has in_flight > 0 or
            now - max(last_request_sent, last_response_end) < window_sec.
        """
        main_id = self.main_lineage_id()
        for lineage, state in self._lineages.items():
            if lineage == main_id:
                continue
            if state.in_flight > 0:
                return True
            if now - max(state.last_request_sent, state.last_response_end) < window_sec:
                return True
        return False

    def last_activity(self) -> float:
        """Return the most recent request-or-response time across all lineages.

        Used to evict long-idle sessions in a detached (daemon) proxy, where
        sessions would otherwise accumulate for the life of the process.

        Returns:
            The max of last_request_sent and last_response_end over every
            lineage, or float("-inf") if no lineage has been seen.
        """
        activity = float("-inf")
        for state in self._lineages.values():
            activity = max(activity, state.last_request_sent, state.last_response_end)
        return activity
