import asyncio
import json
import socket
import time
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Protocol

import httpx
from mitmproxy import http
from mitmproxy.options import Options
from mitmproxy.tools.dump import DumpMaster

from claude_thermos.config import ANTHROPIC_BASE_URL, Config
from claude_thermos.credentials import StoredLogin
from claude_thermos.eventlog import EventLog, EventType, EventWriter
from claude_thermos.limits import parse_rate_limits, rejection_headers
from claude_thermos.lineage import LineageId, extract_session_id
from claude_thermos.state import SessionState
from claude_thermos.usage import Usage, parse_stop_reason_sse, parse_usage_json, parse_usage_sse
from claude_thermos.warmer import LIFT_PROMPT, Compactor, Warmer, update_pause

MESSAGES_PATH = "/v1/messages"

# The body of the refusal sent while a conversation is paused, in the API's
# error shape. Claude Code shows its own usage-limit notice rather than this
# message, but it says what happened to anyone reading the raw response.
_PAUSED_ERROR = {
    "type": "error",
    "error": {
        "type": "rate_limit_error",
        "message": (
            "claude-thermos paused this session near your session limit to keep its prompt "
            "cache warm until the limit resets."
        ),
    },
}

# Local-only endpoint Claude Code's hooks post to (see `hook_settings`). The
# proxy answers it itself; it is never forwarded upstream.
HOOK_PATH = "/claude-thermos/hook"

# The hook events the proxy listens for: a prompt submitted in a session marks
# it as open in a terminal, SessionEnd closes it, and PostCompact reports that
# its conversation was summarized.
_HOOK_PROMPT = "UserPromptSubmit"
_HOOK_SESSION_END = "SessionEnd"
_HOOK_COMPACTED = "PostCompact"

_SSE_CONTENT_TYPE = "text/event-stream"

# mitmproxy's own type for `Response.stream`: False buffers the body, a callable
# is handed each chunk as it arrives and returns what to forward.
StreamSetting = Callable[[bytes], Iterable[bytes] | bytes] | bool

# How often the detached-proxy session reaper scans for idle sessions to evict.
_REAP_INTERVAL_SEC = 60.0


def find_free_port() -> int:
    """Find a free TCP port on the loopback interface.

    Binds a socket to 127.0.0.1:0, reads the assigned port, and closes it.

    Returns:
        The assigned port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def build_master(port: int, addon: object | None, upstream: str = ANTHROPIC_BASE_URL) -> DumpMaster:
    """Construct a DumpMaster in reverse-proxy mode. Does not start it.

    Configured in reverse mode to `upstream` (the real Anthropic API by
    default) on 127.0.0.1:<port>, with quiet/no-terminal options.

    Args:
        port: Loopback port the proxy listens on.
        addon: Addon to register, or None to register nothing.
        upstream: Upstream base URL to reverse-proxy to.

    Returns:
        The configured, unstarted DumpMaster.
    """
    options = Options(
        mode=[f"reverse:{upstream}"],
        listen_host="127.0.0.1",
        listen_port=port,
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()

    master = DumpMaster(
        options,
        loop=loop,
        with_termlog=False,
        with_dumper=False,
    )

    if addon is not None:
        master.addons.add(addon)

    return master


class _RequestLike(Protocol):
    """The subset of mitmproxy's Request that WarmerAddon's hooks read (and,
    for `headers`, mutate in place via `__setitem__`). Kept as a structural
    protocol (rather than importing mitmproxy's concrete type) so a
    lightweight duck-typed fake satisfies it in tests. Members are declared
    as read-only properties so a plain attribute on a fake class still
    satisfies the protocol."""

    @property
    def path(self) -> str: ...
    @property
    def headers(self) -> MutableMapping[str, str]: ...
    def get_text(self) -> str | None: ...


class _ResponseLike(Protocol):
    """The subset of mitmproxy's Response that WarmerAddon's hooks read, plus
    `stream`, which the responseheaders hook sets to pass a body through."""

    stream: StreamSetting

    @property
    def content(self) -> bytes | None: ...
    @property
    def headers(self) -> Mapping[str, str]: ...


class FlowLike(Protocol):
    """The subset of mitmproxy's HTTPFlow that WarmerAddon's hooks use.
    `response` is settable because the proxy answers its own hook endpoint."""

    @property
    def request(self) -> _RequestLike: ...
    @property
    def response(self) -> _ResponseLike | None: ...
    @response.setter
    def response(self, value: http.Response) -> None: ...


def _handle_request(
    state: SessionState,
    eventlog: EventLog,
    body: dict,
    headers: dict,
    now: float,
    seen_lineages: set[LineageId],
) -> LineageId:
    """Record a request on state and emit session/lineage events.

    Emits session_start on the first request ever seen for the session and
    lineage_seen on the first sight of a lineage. A side request (see
    `SessionState.is_side_request`) is logged and otherwise ignored, apart
    from its credentials, which are as current as any other request's.

    Args:
        state: The session's state.
        eventlog: The session's event log.
        body: The decoded request body.
        headers: The request's headers.
        now: Current time, in seconds.
        seen_lineages: Lineage ids already seen this session; mutated in place.

    Returns:
        The resolved lineage id.
    """
    lineage = LineageId.from_request_body(body)
    if state.is_side_request(lineage, body, now):
        state.note_credentials(headers)
        eventlog.emit(EventType.SIDE_REQUEST, lineage_id=lineage)
        return lineage

    is_new_session = not seen_lineages
    is_new_lineage = lineage not in seen_lineages

    state.on_request(lineage, body, headers, now)

    if is_new_session:
        eventlog.emit(EventType.SESSION_START, lineage_id=lineage, session_id=state.session_id)
    if is_new_lineage:
        eventlog.emit(
            EventType.LINEAGE_SEEN, lineage_id=lineage, has_tools=len(body.get("tools", [])) > 0
        )
        seen_lineages.add(lineage)

    return lineage


def _is_sse(raw_response: bytes) -> bool:
    stripped = raw_response.lstrip()
    return stripped.startswith(b"event:") or stripped.startswith(b"data:")


def _parse_response(raw_response: bytes) -> tuple[Usage, str | None]:
    """Parse usage and the stop reason from a /v1/messages response.

    A body that is neither SSE nor a JSON object (an interrupted stream, an
    upstream error page) yields zero usage and no stop reason, so the
    response is still recorded on its lineage.

    Args:
        raw_response: The raw response bytes.

    Returns:
        The response's usage and its stop_reason, or None if it has none.
    """
    try:
        if _is_sse(raw_response):
            return parse_usage_sse([raw_response]), parse_stop_reason_sse([raw_response])
        body = json.loads(raw_response)
        stop_reason = body.get("stop_reason")
        return parse_usage_json(body), stop_reason if isinstance(stop_reason, str) else None
    except (ValueError, AttributeError):
        return Usage(0, 0, 0, 0), None


def _handle_response(
    state: SessionState,
    eventlog: EventLog,
    lineage: LineageId,
    raw_response: bytes,
    now: float,
    headers: Mapping[str, str] | None = None,
    side: bool = False,
) -> None:
    """Record a response on state and emit a usage event.

    Parses usage and the stop reason from `raw_response`, records the
    response on `state`, and emits a usage event. Subscription limits in
    `headers` replace the session's last reading. A side request's response
    updates the limits and the log but not its lineage.

    Args:
        state: The session's state.
        eventlog: The session's event log.
        lineage: The response's lineage id.
        raw_response: The raw response bytes.
        now: Current time, in seconds.
        headers: The response headers, if available.
        side: Whether the response answers a side request.
    """
    limits = parse_rate_limits(headers) if headers is not None else None
    if limits is not None:
        state.limits = limits
    usage, stop_reason = _parse_response(raw_response)
    if not side:
        input_tokens = usage.uncached_input + usage.cache_read + usage.cache_creation
        state.on_response(lineage, now, stop_reason, input_tokens)
    eventlog.emit(
        EventType.USAGE,
        lineage_id=lineage,
        usage={
            "uncached_input": usage.uncached_input,
            "cache_read": usage.cache_read,
            "cache_creation": usage.cache_creation,
            "output": usage.output,
        },
    )


def hook_settings(port: int) -> dict:
    """Return Claude Code settings that report session events to the proxy.

    Registers HTTP hooks pointing at the proxy's own HOOK_PATH, so the
    warmer knows which session a terminal has open, when that session ends,
    and when its conversation was compacted. Keepalive only ever applies to
    a session reported this way.

    Args:
        port: Loopback port the proxy listens on.

    Returns:
        A settings object suitable for `claude --settings`.
    """
    handler = {"hooks": [{"type": "http", "url": f"http://127.0.0.1:{port}{HOOK_PATH}"}]}
    return {
        "hooks": {event: [handler] for event in (_HOOK_PROMPT, _HOOK_SESSION_END, _HOOK_COMPACTED)}
    }


@dataclass
class _Session:
    state: SessionState
    eventlog: EventLog
    seen_lineages: set[LineageId]


class WarmerAddon:
    """mitmproxy addon that observes /v1/messages traffic into per-session
    SessionState + EventLog, emits structured events, and drives a Warmer
    that refreshes the main lineage's cache prefix while it sits idle.
    Streamed responses are passed through as they arrive rather than
    buffered, so Claude Code still sees tokens (and its stream watchdogs
    still see bytes) while the addon keeps its own copy to parse.

    Unless warming is disabled, the addon owns a Warmer and a background
    task started from mitmproxy's `running` hook (on the proxy event loop)
    that polls the active sessions. On shutdown the task is cancelled, each
    session's summary is written, and its event log is closed.

    All sessions share a single EventWriter thread so N concurrent sessions
    cost one writer thread rather than N. When a custom eventlog_factory is
    supplied (in tests) it owns its logs' writers instead, and the addon
    creates no shared writer."""

    def __init__(
        self,
        config: Config | None = None,
        eventlog_factory: Callable[[str], EventLog] | None = None,
        client: httpx.AsyncClient | None = None,
        upstream: str | None = None,
        reap_sessions: bool = False,
        compactor: Compactor | None = None,
        single_session: bool = False,
        login: StoredLogin | None = None,
    ) -> None:
        self._config = config if config is not None else Config()
        if eventlog_factory is None:
            self._event_writer: EventWriter | None = EventWriter()
            eventlog_factory = lambda sid: EventLog(sid, writer=self._event_writer)  # noqa: E731
        else:
            self._event_writer = None
        self._eventlog_factory = eventlog_factory
        self._sessions: dict[str, _Session] = {}
        self._warmer: Warmer | None = None
        if not self._config.disabled:
            base_url = upstream if upstream is not None else ANTHROPIC_BASE_URL
            self._warmer = Warmer(
                self._config,
                base_url=base_url,
                client=client,
                compactor=compactor,
                login=login,
            )
        # Keyed by id(flow), for the life of one exchange: the chunks of a
        # response being streamed through, and the flows that are side requests.
        self._streamed: dict[int, list[bytes]] = {}
        self._side_flows: set[int] = set()
        # Requests answered with the pause refusal, which never reached the API.
        self._refused: set[int] = set()
        # Sessions a hook reported open. The launcher fronts one terminal, so
        # there a newly reported session replaces the last; the daemon serves
        # many clients and waits for each session's own SessionEnd.
        self._live: set[str] = set()
        self._single_session = single_session
        # Session eviction is only for the long-lived daemon; the launcher
        # process dies with claude, so its default behavior stays unchanged.
        self._reap_sessions = reap_sessions
        self._warmer_task: asyncio.Task | None = None
        self._reaper_task: asyncio.Task | None = None

    def active_sessions(self) -> list[tuple[SessionState, EventLog]]:
        """Return the active (SessionState, EventLog) pairs.

        Returns:
            One pair per session seen so far.
        """
        return [(session.state, session.eventlog) for session in self._sessions.values()]

    def running(self) -> None:
        """Handle mitmproxy's startup hook on the proxy event loop.

        Starts the background warming task (unless warming is disabled). In
        daemon mode it also starts the session reaper that evicts long-idle
        sessions; the launcher never reaps, keeping its behavior unchanged.
        """
        if self._warmer is not None and self._warmer_task is None:
            self._warmer_task = asyncio.create_task(self._warmer.run(self.active_sessions))
        if self._reap_sessions and self._reaper_task is None:
            self._reaper_task = asyncio.create_task(self._reap())

    def request(self, flow: FlowLike) -> None:
        """Handle a mitmproxy request hook.

        On a /v1/messages request, decodes the JSON body, resolves
        session/lineage, updates state, and forces accept-encoding:
        identity so the response usage parses. A hook post is answered
        locally, and so is a model request from a paused session.

        Args:
            flow: The mitmproxy flow for the request.
        """
        if flow.request.path.startswith(HOOK_PATH):
            self._handle_hook(flow)
            return
        if not flow.request.path.startswith(MESSAGES_PATH):
            return
        flow.request.headers["accept-encoding"] = "identity"

        resolved = self._resolve_session_and_body(flow)
        if resolved is None:
            return
        session, body = resolved
        now = time.time()
        if self._refuse_if_paused(flow, session, now):
            return
        if session.state.is_side_request(LineageId.from_request_body(body), body, now):
            self._side_flows.add(id(flow))
        _handle_request(
            session.state,
            session.eventlog,
            body,
            dict(flow.request.headers),
            now,
            session.seen_lineages,
        )

    def responseheaders(self, flow: FlowLike) -> None:
        """Handle a mitmproxy responseheaders hook.

        Switches a streamed /v1/messages response to pass-through, teeing
        each chunk into a buffer the response hook parses. Without this
        mitmproxy holds the whole response until it completes.

        Args:
            flow: The mitmproxy flow whose response headers arrived.
        """
        response = flow.response
        if response is None or not flow.request.path.startswith(MESSAGES_PATH):
            return
        if not response.headers.get("content-type", "").startswith(_SSE_CONTENT_TYPE):
            return
        chunks: list[bytes] = []
        self._streamed[id(flow)] = chunks

        def tee(chunk: bytes) -> bytes:
            chunks.append(chunk)
            return chunk

        response.stream = tee

    def response(self, flow: FlowLike) -> None:
        """Handle a mitmproxy response hook.

        Parses usage, the stop reason, and subscription limits from the
        response, updates state, and emits a usage event. Never modifies
        the response body.

        Args:
            flow: The mitmproxy flow for the response.
        """
        chunks = self._streamed.pop(id(flow), None)
        side = id(flow) in self._side_flows
        self._side_flows.discard(id(flow))
        if id(flow) in self._refused:
            self._refused.discard(id(flow))
            return
        if flow.response is None:
            return
        resolved = self._resolve_session_and_body(flow)
        if resolved is None:
            return
        session, body = resolved
        lineage = LineageId.from_request_body(body)
        _handle_response(
            session.state,
            session.eventlog,
            lineage,
            b"".join(chunks) if chunks is not None else flow.response.content or b"",
            time.time(),
            headers=flow.response.headers,
            side=side,
        )

    def error(self, flow: FlowLike) -> None:
        """Handle a mitmproxy error hook.

        A request that ends without a complete response (the user
        interrupted it, or the connection dropped) never reaches the
        response hook, so its lineage is released here. Otherwise it would
        count as in flight, and never idle, for the rest of the session.

        Args:
            flow: The mitmproxy flow that failed.
        """
        self._streamed.pop(id(flow), None)
        side = id(flow) in self._side_flows
        self._side_flows.discard(id(flow))
        if id(flow) in self._refused:
            self._refused.discard(id(flow))
            return
        resolved = self._resolve_session_and_body(flow)
        if resolved is None or side:
            return
        session, body = resolved
        session.state.on_response(LineageId.from_request_body(body), time.time())

    def _refuse_if_paused(self, flow: FlowLike, session: _Session, now: float) -> bool:
        """Answer a model request with a usage-limit refusal if its session
        is paused, and report whether it did.

        The refusal has the shape of the API's own at the session limit, so
        Claude Code stops at this request boundary, waits for the reset in
        the open session, and then sends the request again. Token counting
        is free, so it is never refused.

        Args:
            flow: The mitmproxy flow for the request.
            session: The request's session.
            now: Current time, in seconds.

        Returns:
            True iff the request was answered here and must not go upstream.
        """
        if flow.request.path.split("?", 1)[0] != MESSAGES_PATH:
            return False
        state = session.state
        if not update_pause(state, session.eventlog, self._config, now):
            return False
        if state.limits is None or state.paused_until is None:
            return False
        headers = rejection_headers(state.limits, state.paused_until, now)
        headers["content-type"] = "application/json"
        flow.response = http.Response.make(429, json.dumps(_PAUSED_ERROR), headers)
        self._refused.add(id(flow))
        return True

    def _handle_hook(self, flow: FlowLike) -> None:
        """Answer a Claude Code hook post and record what it reports.

        The hook posts its JSON input, which names the event and the
        session. The request is answered here and never forwarded upstream.

        Args:
            flow: The mitmproxy flow carrying the hook's POST.
        """
        try:
            payload = json.loads(flow.request.get_text() or "")
        except (json.JSONDecodeError, ValueError):
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("session_id"), str):
            self._on_hook(payload["session_id"], payload)
        # An empty 2xx body is a successful hook with no output.
        flow.response = http.Response.make(200)

    def _on_hook(self, session_id: str, payload: dict) -> None:
        event = payload.get("hook_event_name")
        if event == _HOOK_PROMPT:
            if self._single_session:
                self._live.clear()
            self._live.add(session_id)
        elif event == _HOOK_SESSION_END:
            self._live.discard(session_id)
        for known_id, known in self._sessions.items():
            known.state.live = known_id in self._live
        session = self._sessions.get(session_id)
        if event == _HOOK_PROMPT and session is not None and self._config.keepalive:
            # A prompt sent past the pause threshold is the user choosing to
            # spend what is left of the session limit.
            if session.state.waive_pause(time.time(), self._config.pause_at):
                session.eventlog.emit(
                    EventType.PAUSE_LIFTED,
                    lineage_id=session.state.main_lineage_id() or LineageId(""),
                    reason=LIFT_PROMPT,
                )
        if event == _HOOK_COMPACTED and session is not None:
            session.state.on_compacted(time.time())
            session.eventlog.emit(
                EventType.COMPACTED,
                lineage_id=session.state.main_lineage_id() or LineageId(""),
                trigger=payload.get("trigger"),
            )

    def done(self) -> None:
        """Handle mitmproxy's addon shutdown hook.

        Cancels the warming and reaper tasks, then for each remaining session
        emits a session_end event and closes the event log, which flushes its
        records to disk and writes summary.json with the rollup totals.
        Finally shuts down the shared writer thread, if the addon owns one.
        """
        if self._warmer_task is not None:
            self._warmer_task.cancel()
            self._warmer_task = None
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            self._reaper_task = None
        for session_id in list(self._sessions):
            self._evict_session(session_id)
        if self._event_writer is not None:
            self._event_writer.close()

    def _evict_session(self, session_id: str) -> None:
        """Tear down a single session: emit session_end, close its log, and
        forget its warmer episode. Idempotent; a no-op if already gone.

        Args:
            session_id: The session to evict.
        """
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        main_id = session.state.main_lineage_id() or LineageId("")
        session.eventlog.emit(EventType.SESSION_END, lineage_id=main_id)
        session.eventlog.close()
        if self._warmer is not None:
            self._warmer.forget(session_id)

    def _evict_stale(self, now: float) -> None:
        """Evict every session idle for at least session_ttl_sec.

        A session with a warm request currently in flight is left alone, so
        the reaper never closes an event log out from under `Warmer._fire`.

        Args:
            now: Current time, in seconds.
        """
        ttl = self._config.session_ttl_sec
        stale = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.state.last_activity() >= ttl
            and not (self._warmer is not None and self._warmer.is_warming(session_id))
        ]
        for session_id in stale:
            self._evict_session(session_id)

    async def _reap(self, poll_interval_sec: float = _REAP_INTERVAL_SEC) -> None:
        """Poll on an interval, evicting long-idle sessions. Stops on cancel.

        Args:
            poll_interval_sec: Seconds to sleep between reaping scans.
        """
        while True:
            await asyncio.sleep(poll_interval_sec)
            try:
                self._evict_stale(time.time())
            except Exception:
                # Best-effort eviction: keep the reaper alive even if one session teardown fails.
                continue

    def _get_or_create_session(self, session_id: str) -> _Session:
        session = self._sessions.get(session_id)
        if session is None:
            state = SessionState(session_id)
            state.live = session_id in self._live
            session = _Session(
                state=state,
                eventlog=self._eventlog_factory(session_id),
                seen_lineages=set(),
            )
            self._sessions[session_id] = session
        return session

    def _resolve_session_and_body(self, flow: FlowLike) -> tuple[_Session, dict] | None:
        if not flow.request.path.startswith(MESSAGES_PATH):
            return None
        try:
            body = json.loads(flow.request.get_text() or "")
        except (json.JSONDecodeError, ValueError):
            return None
        session_id = extract_session_id(body)
        if session_id is None:
            return None
        return self._get_or_create_session(session_id), body
