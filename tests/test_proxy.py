import json
import socket
import time
from collections.abc import MutableMapping
from pathlib import Path

from mitmproxy import http

from claude_thermos.config import Config
from claude_thermos.eventlog import EventLog
from claude_thermos.lineage import LineageId
from claude_thermos.proxy import (
    HOOK_PATH,
    StreamSetting,
    WarmerAddon,
    build_master,
    find_free_port,
    hook_settings,
    _handle_request,
    _handle_response,
)
from claude_thermos.state import SessionState


def test_find_free_port_bindable() -> None:
    port = find_free_port()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))


def test_build_master_reverse_options() -> None:
    port = find_free_port()

    master = build_master(port, None)

    assert master.options.mode == ["reverse:https://api.anthropic.com"]
    assert master.options.listen_host == "127.0.0.1"
    assert master.options.listen_port == port


def test_build_master_custom_upstream() -> None:
    port = find_free_port()

    master = build_master(port, None, upstream="https://example.test")

    assert master.options.mode == ["reverse:https://example.test"]


def test_build_master_registers_addon() -> None:
    port = find_free_port()
    sentinel = object()

    master = build_master(port, sentinel)

    assert any(a is sentinel for a in master.addons.chain)


def test_warmer_addon_forwards_upstream_to_warmer() -> None:
    addon = WarmerAddon(config=Config(), upstream="https://example.test")

    assert addon._warmer is not None
    assert addon._warmer._base_url == "https://example.test"


_HEADERS = {"authorization": "Bearer abc"}


def _body(model: str, tool_count: int, system: str = "system prompt") -> dict:
    return {
        "model": model,
        "system": system,
        "tools": [{"name": f"tool-{i}"} for i in range(tool_count)],
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"user_id": json.dumps({"session_id": "sess-1"})},
    }


def _sse_event(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _events(tmp_path: Path, session_id: str = "sess-1") -> list[dict]:
    lines = (tmp_path / session_id / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_handle_request_emits_session_and_lineage(tmp_path: Path) -> None:
    state = SessionState("sess-1")
    eventlog = EventLog("sess-1", root=tmp_path)
    seen_lineages: set[LineageId] = set()
    main = _body("claude-opus-4-8", 12)

    lineage = _handle_request(state, eventlog, main, _HEADERS, now=0, seen_lineages=seen_lineages)

    assert state.main_lineage_id() == lineage

    # A second request on the same lineage must not re-emit either event.
    _handle_request(state, eventlog, main, _HEADERS, now=1, seen_lineages=seen_lineages)
    eventlog.close()

    events = _events(tmp_path)
    session_start = [e for e in events if e["event"] == "session_start"]
    lineage_seen = [e for e in events if e["event"] == "lineage_seen"]
    assert len(session_start) == 1
    assert session_start[0]["lineage_id"] == lineage
    assert len(lineage_seen) == 1
    assert lineage_seen[0]["lineage_id"] == lineage


def test_handle_response_emits_usage(tmp_path: Path) -> None:
    state = SessionState("sess-1")
    eventlog = EventLog("sess-1", root=tmp_path)
    seen_lineages: set[LineageId] = set()
    main = _body("claude-opus-4-8", 12)

    lineage = _handle_request(state, eventlog, main, _HEADERS, now=0, seen_lineages=seen_lineages)
    assert state.is_main_idle(now=0, idle_threshold_sec=0) is False  # request still in flight

    raw_response = _sse_event(
        "message_start",
        {
            "message": {
                "usage": {
                    "input_tokens": 2,
                    "cache_read_input_tokens": 157547,
                    "cache_creation_input_tokens": 4520,
                }
            }
        },
    ) + _sse_event("message_delta", {"usage": {"output_tokens": 1286}})

    _handle_response(state, eventlog, lineage, raw_response, now=0)

    assert state.is_main_idle(now=0, idle_threshold_sec=0) is True  # in_flight decremented to 0
    eventlog.close()

    events = _events(tmp_path)
    usage_events = [e for e in events if e["event"] == "usage"]
    assert len(usage_events) == 1
    assert usage_events[0]["lineage_id"] == lineage
    assert usage_events[0]["usage"] == {
        "uncached_input": 2,
        "cache_read": 157547,
        "cache_creation": 4520,
        "output": 1286,
    }


def test_no_session_id_excluded_from_main(tmp_path: Path) -> None:
    state = SessionState("sess-1")
    eventlog = EventLog("sess-1", root=tmp_path)
    seen_lineages: set[LineageId] = set()

    # A title/quota ping: no tools, and no session metadata at all.
    ping = {
        "model": "claude-opus-4-8",
        "tools": [],
        "messages": [{"role": "user", "content": "hi"}],
    }
    main = _body("claude-opus-4-8", 12)

    _handle_request(state, eventlog, ping, _HEADERS, now=0, seen_lineages=seen_lineages)
    assert state.main_lineage_id() is None

    _handle_request(state, eventlog, main, _HEADERS, now=1, seen_lineages=seen_lineages)
    assert state.main_lineage_id() == LineageId.from_request_body(main)

    eventlog.close()


class _FakeRequest:
    """Duck-typed stand-in for mitmproxy's Request: exposes only the
    attributes WarmerAddon's hooks read/write (`path`, `headers`,
    `get_text()`)."""

    def __init__(self, path: str, text: str, headers: dict[str, str] | None = None) -> None:
        self.path = path
        self.headers: MutableMapping[str, str] = dict(headers) if headers else {}
        self._text = text

    def get_text(self) -> str:
        return self._text


class _FakeResponse:
    """Duck-typed stand-in for mitmproxy's Response: exposes only `content`,
    `headers`, and the `stream` attribute the addon sets."""

    def __init__(self, content: bytes, headers: dict[str, str] | None = None) -> None:
        self.content = content
        self.headers: dict[str, str] = dict(headers) if headers else {}
        self.stream: StreamSetting = False


class _FakeFlow:
    """Duck-typed stand-in for mitmproxy's HTTPFlow: exposes only `request`
    and `response`."""

    def __init__(self, request: _FakeRequest, response: _FakeResponse | None = None) -> None:
        self.request = request
        self.response: _FakeResponse | http.Response | None = response


def _addon(tmp_path: Path) -> WarmerAddon:
    return WarmerAddon(eventlog_factory=lambda session_id: EventLog(session_id, root=tmp_path))


def test_request_forces_accept_encoding_identity(tmp_path: Path) -> None:
    addon = _addon(tmp_path)
    flow = _FakeFlow(
        request=_FakeRequest(
            path="/v1/messages",
            text=json.dumps(_body("claude-opus-4-8", 12)),
            headers={"accept-encoding": "gzip"},
        )
    )

    addon.request(flow)

    assert flow.request.headers["accept-encoding"] == "identity"


def test_request_with_no_session_id_passes_through(tmp_path: Path) -> None:
    addon = _addon(tmp_path)
    ping = {
        "model": "claude-opus-4-8",
        "tools": [],
        "messages": [{"role": "user", "content": "hi"}],
    }
    flow = _FakeFlow(request=_FakeRequest(path="/v1/messages", text=json.dumps(ping)))

    addon.request(flow)  # must not raise, and must not create any session state

    assert list(tmp_path.iterdir()) == []


def test_request_with_invalid_json_body_does_not_raise(tmp_path: Path) -> None:
    addon = _addon(tmp_path)
    flow = _FakeFlow(request=_FakeRequest(path="/v1/messages", text="not json"))

    addon.request(flow)  # must not raise

    assert list(tmp_path.iterdir()) == []


def test_request_response_round_trip_emits_events(tmp_path: Path) -> None:
    addon = _addon(tmp_path)
    main = _body("claude-opus-4-8", 12)
    flow = _FakeFlow(request=_FakeRequest(path="/v1/messages", text=json.dumps(main)))

    addon.request(flow)
    flow.response = _FakeResponse(
        content=_sse_event(
            "message_start",
            {
                "message": {
                    "usage": {
                        "input_tokens": 2,
                        "cache_read_input_tokens": 3,
                        "cache_creation_input_tokens": 4,
                    }
                }
            },
        )
        + _sse_event("message_delta", {"usage": {"output_tokens": 5}})
    )
    addon.response(flow)
    addon.done()

    events = _events(tmp_path)
    event_types = [e["event"] for e in events]
    assert "session_start" in event_types
    assert "lineage_seen" in event_types
    assert "usage" in event_types


_LIMIT_HEADERS = {
    "anthropic-ratelimit-unified-5h-utilization": "0.4",
    "anthropic-ratelimit-unified-5h-reset": "1000",
    "anthropic-ratelimit-unified-5h-status": "allowed",
}


def _turn(stop_reason: str, input_tokens: int = 2) -> bytes:
    return _sse_event(
        "message_start", {"message": {"usage": {"input_tokens": input_tokens}}}
    ) + _sse_event(
        "message_delta", {"delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 5}}
    )


def _with_ttl(body: dict, ttl: str, text: str = "hi") -> dict:
    marked = {"type": "text", "text": text, "cache_control": {"type": "ephemeral", "ttl": ttl}}
    return {**body, "messages": [{"role": "user", "content": [marked]}]}


def _messages_flow(body: dict) -> _FakeFlow:
    return _FakeFlow(request=_FakeRequest(path="/v1/messages", text=json.dumps(body)))


def _hook_flow(event: str, session_id: str = "sess-1", **fields: str) -> _FakeFlow:
    payload = {"hook_event_name": event, "session_id": session_id, **fields}
    return _FakeFlow(request=_FakeRequest(path=HOOK_PATH, text=json.dumps(payload)))


def _state(addon: WarmerAddon, session_id: str = "sess-1") -> SessionState:
    return addon._sessions[session_id].state


def test_handle_response_records_stop_reason_prompt_size_and_limits(tmp_path: Path) -> None:
    state = SessionState("sess-1")
    eventlog = EventLog("sess-1", root=tmp_path)
    main = _body("claude-opus-4-8", 12)
    lineage = _handle_request(state, eventlog, main, _HEADERS, now=0, seen_lineages=set())

    raw_response = _sse_event(
        "message_start",
        {
            "message": {
                "usage": {
                    "input_tokens": 2,
                    "cache_read_input_tokens": 157547,
                    "cache_creation_input_tokens": 4520,
                }
            }
        },
    ) + _sse_event(
        "message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 9}}
    )
    _handle_response(state, eventlog, lineage, raw_response, now=1, headers=_LIMIT_HEADERS)
    eventlog.close()

    main_state = state.main_lineage()
    assert main_state is not None
    assert main_state.last_stop_reason == "end_turn"
    assert main_state.last_input_tokens == 2 + 157547 + 4520
    assert state.main_turn_ended() is True
    assert state.limits is not None
    assert state.limits.session_utilization(now=1) == 0.4


def test_handle_response_tolerates_a_body_it_cannot_parse(tmp_path: Path) -> None:
    state = SessionState("sess-1")
    eventlog = EventLog("sess-1", root=tmp_path)
    main = _body("claude-opus-4-8", 12)
    lineage = _handle_request(state, eventlog, main, _HEADERS, now=0, seen_lineages=set())

    _handle_response(state, eventlog, lineage, b"<html>bad gateway</html>", now=1)
    eventlog.close()

    assert state.is_main_idle(now=1, idle_threshold_sec=0) is True
    assert state.main_turn_ended() is False


def test_side_request_leaves_the_conversation_untouched(tmp_path: Path) -> None:
    addon = _addon(tmp_path)
    main = _with_ttl(_body("claude-opus-4-8", 12), "1h")
    compaction = _with_ttl(_body("claude-opus-4-8", 12), "5m", text="summarize the conversation")

    flow = _messages_flow(main)
    addon.request(flow)
    flow.response = _FakeResponse(content=_turn("end_turn"))
    addon.response(flow)

    side = _messages_flow(compaction)
    addon.request(side)
    main_state = _state(addon).main_lineage()
    assert main_state is not None
    assert main_state.request_count == 1
    assert main_state.in_flight == 0
    assert main_state.last_request_body == main

    side.response = _FakeResponse(content=_turn("tool_use"), headers=_LIMIT_HEADERS)
    addon.response(side)
    addon.done()

    assert main_state.last_stop_reason == "end_turn"
    names = [e["event"] for e in _events(tmp_path)]
    assert names.count("side_request") == 1
    assert names.count("usage") == 2


def test_side_request_still_carries_the_newest_credentials(tmp_path: Path) -> None:
    addon = _addon(tmp_path)
    main = _with_ttl(_body("claude-opus-4-8", 12), "1h")
    flow = _FakeFlow(
        _FakeRequest("/v1/messages", json.dumps(main), {"authorization": "Bearer old"})
    )
    addon.request(flow)

    side_body = _with_ttl(_body("claude-opus-4-8", 12), "5m")
    side = _FakeFlow(
        _FakeRequest("/v1/messages", json.dumps(side_body), {"authorization": "Bearer new"})
    )
    addon.request(side)

    state = _state(addon)
    main_state = state.main_lineage()
    addon.done()
    assert main_state is not None
    assert main_state.request_count == 1
    assert main_state.auth_headers["authorization"] == "Bearer old"
    assert state.credentials == {"authorization": "Bearer new"}


def test_side_response_updates_limits(tmp_path: Path) -> None:
    addon = _addon(tmp_path)
    main = _with_ttl(_body("claude-opus-4-8", 12), "1h")
    flow = _messages_flow(main)
    addon.request(flow)
    flow.response = _FakeResponse(content=_turn("end_turn"))
    addon.response(flow)

    side = _messages_flow(_with_ttl(_body("claude-opus-4-8", 12), "5m"))
    addon.request(side)
    side.response = _FakeResponse(content=_turn("end_turn"), headers=_LIMIT_HEADERS)
    addon.response(side)

    limits = _state(addon).limits
    addon.done()
    assert limits is not None
    assert limits.session_utilization(now=0) == 0.4


def test_streamed_response_is_passed_through_and_still_parsed(tmp_path: Path) -> None:
    addon = _addon(tmp_path)
    flow = _messages_flow(_body("claude-opus-4-8", 12))
    addon.request(flow)
    # A streamed response reaches the response hook with no content of its own.
    response = _FakeResponse(content=b"", headers={"content-type": "text/event-stream"})
    flow.response = response

    addon.responseheaders(flow)
    assert callable(response.stream)
    body = _turn("end_turn", input_tokens=7)
    chunks = [body[:20], body[20:], b""]
    forwarded = [response.stream(chunk) for chunk in chunks]
    addon.response(flow)
    addon.done()

    assert forwarded == chunks
    usage = next(e for e in _events(tmp_path) if e["event"] == "usage")
    assert usage["usage"]["uncached_input"] == 7
    assert addon._streamed == {}


def test_non_streamed_response_is_left_buffered(tmp_path: Path) -> None:
    addon = _addon(tmp_path)
    flow = _messages_flow(_body("claude-opus-4-8", 12))
    addon.request(flow)
    response = _FakeResponse(content=b"{}", headers={"content-type": "application/json"})
    flow.response = response

    addon.responseheaders(flow)
    addon.done()

    assert response.stream is False


def test_interrupted_request_releases_its_lineage(tmp_path: Path) -> None:
    addon = _addon(tmp_path)
    flow = _messages_flow(_body("claude-opus-4-8", 12))
    addon.request(flow)
    state = _state(addon)
    assert state.is_main_idle(now=10**12, idle_threshold_sec=0) is False

    addon.error(flow)
    addon.done()

    assert state.is_main_idle(now=10**12, idle_threshold_sec=0) is True
    assert state.main_turn_ended() is False


def test_hook_settings_point_every_hook_at_the_proxy() -> None:
    settings = hook_settings(8123)

    assert set(settings["hooks"]) == {"UserPromptSubmit", "SessionEnd", "PostCompact"}
    for handlers in settings["hooks"].values():
        assert handlers == [
            {"hooks": [{"type": "http", "url": f"http://127.0.0.1:8123{HOOK_PATH}"}]}
        ]


def test_hook_post_is_answered_locally(tmp_path: Path) -> None:
    addon = _addon(tmp_path)
    flow = _hook_flow("UserPromptSubmit")

    addon.request(flow)

    assert isinstance(flow.response, http.Response)
    assert flow.response.status_code == 200
    assert "accept-encoding" not in flow.request.headers


def test_malformed_hook_post_is_still_answered(tmp_path: Path) -> None:
    addon = _addon(tmp_path)
    flow = _FakeFlow(request=_FakeRequest(path=HOOK_PATH, text="not json"))

    addon.request(flow)

    assert isinstance(flow.response, http.Response)
    assert addon._live == set()


def test_prompt_hook_marks_the_session_live_before_and_after_it_is_seen(tmp_path: Path) -> None:
    addon = _addon(tmp_path)

    # The hook fires before the prompt's first request reaches the proxy.
    addon.request(_hook_flow("UserPromptSubmit", "sess-1"))
    addon.request(_messages_flow(_body("claude-opus-4-8", 12)))
    state = _state(addon)
    assert state.live is True

    addon.request(_hook_flow("SessionEnd", "sess-1", reason="clear"))
    addon.done()
    assert state.live is False


def test_session_never_reported_by_a_hook_is_not_live(tmp_path: Path) -> None:
    addon = _addon(tmp_path)

    addon.request(_messages_flow(_body("claude-opus-4-8", 12)))
    state = _state(addon)
    addon.done()

    assert state.live is False


def _session_body(session_id: str) -> dict:
    return {
        **_body("claude-opus-4-8", 12),
        "metadata": {"user_id": json.dumps({"session_id": session_id})},
    }


def _live_after_two_prompts(single_session: bool, tmp_path: Path) -> dict[str, bool]:
    """Submit a prompt in sess-1, then in sess-2, and report which are live."""
    addon = WarmerAddon(
        eventlog_factory=lambda session_id: EventLog(session_id, root=tmp_path),
        single_session=single_session,
    )
    for session_id in ("sess-1", "sess-2"):
        addon.request(_hook_flow("UserPromptSubmit", session_id))
        addon.request(_messages_flow(_session_body(session_id)))
    live = {session_id: _state(addon, session_id).live for session_id in ("sess-1", "sess-2")}
    addon.done()
    return live


def test_launcher_has_one_live_session_at_a_time(tmp_path: Path) -> None:
    live = _live_after_two_prompts(single_session=True, tmp_path=tmp_path)

    assert live == {"sess-1": False, "sess-2": True}


def test_daemon_keeps_every_reported_session_live(tmp_path: Path) -> None:
    live = _live_after_two_prompts(single_session=False, tmp_path=tmp_path)

    assert live == {"sess-1": True, "sess-2": True}


def test_compaction_hook_marks_the_conversation_compacted(tmp_path: Path) -> None:
    addon = _addon(tmp_path)
    addon.request(_messages_flow(_body("claude-opus-4-8", 12)))

    addon.request(_hook_flow("PostCompact", trigger="manual"))
    state = _state(addon)
    addon.done()

    assert state.main_compacted() is True
    compacted = next(e for e in _events(tmp_path) if e["event"] == "compacted")
    assert compacted["trigger"] == "manual"


def test_compaction_hook_for_an_unknown_session_is_ignored(tmp_path: Path) -> None:
    addon = _addon(tmp_path)

    addon.request(_hook_flow("PostCompact", "sess-unknown"))

    assert list(tmp_path.iterdir()) == []


def _pausing_addon(tmp_path: Path) -> WarmerAddon:
    config = Config(keepalive=True, pause_at=0.95)
    return WarmerAddon(config, eventlog_factory=lambda sid: EventLog(sid, root=tmp_path))


def _limit_headers(used: float, resets_at: float) -> dict[str, str]:
    return {
        "anthropic-ratelimit-unified-5h-utilization": str(used),
        "anthropic-ratelimit-unified-5h-reset": str(int(resets_at)),
        "anthropic-ratelimit-unified-5h-status": "allowed_warning",
        "anthropic-ratelimit-unified-overage-status": "rejected",
    }


def _turn_at(addon: WarmerAddon, used: float, resets_at: float) -> None:
    """Run one live one-hour main turn whose response reports `used`."""
    flow = _messages_flow(_with_ttl(_body("claude-opus-4-8", 12), "1h"))
    addon.request(flow)
    flow.response = _FakeResponse(_turn("tool_use"), headers=_limit_headers(used, resets_at))
    addon.response(flow)


def test_request_past_the_pause_threshold_is_refused_until_the_reset(tmp_path: Path) -> None:
    addon = _pausing_addon(tmp_path)
    addon.request(_hook_flow("UserPromptSubmit"))
    resets_at = time.time() + 4 * 3600
    _turn_at(addon, used=0.96, resets_at=resets_at)
    state = _state(addon)
    main = state.main_lineage()
    assert main is not None
    sent = main.request_count

    flow = _messages_flow(_with_ttl(_body("claude-opus-4-8", 12), "1h", text="tool result"))
    addon.request(flow)
    refusal = flow.response
    addon.response(flow)
    addon.done()

    assert isinstance(refusal, http.Response)
    assert refusal.status_code == 429
    assert refusal.headers["anthropic-ratelimit-unified-status"] == "rejected"
    assert refusal.headers["anthropic-ratelimit-unified-reset"] == str(int(resets_at))
    assert refusal.headers["anthropic-ratelimit-unified-representative-claim"] == "five_hour"
    # The refusal never reached the API, so it changes nothing the warmer reads.
    assert main.request_count == sent
    assert state.limits is not None and state.limits.allowed(time.time())
    assert [e["event"] for e in _events(tmp_path)].count("usage") == 1
    paused = [e for e in _events(tmp_path) if e["event"] == "paused"]
    assert len(paused) == 1
    assert paused[0]["resumes_at"] == int(resets_at)


def test_prompt_past_the_threshold_lifts_the_pause(tmp_path: Path) -> None:
    addon = _pausing_addon(tmp_path)
    addon.request(_hook_flow("UserPromptSubmit"))
    _turn_at(addon, used=0.96, resets_at=time.time() + 4 * 3600)
    refused = _messages_flow(_with_ttl(_body("claude-opus-4-8", 12), "1h"))
    addon.request(refused)
    assert refused.response is not None

    addon.request(_hook_flow("UserPromptSubmit"))
    allowed = _messages_flow(_with_ttl(_body("claude-opus-4-8", 12), "1h"))
    addon.request(allowed)
    addon.done()

    assert allowed.response is None
    lifted = [e for e in _events(tmp_path) if e["event"] == "pause_lifted"]
    assert [e["reason"] for e in lifted] == ["prompt"]


def test_token_counting_is_never_refused(tmp_path: Path) -> None:
    addon = _pausing_addon(tmp_path)
    addon.request(_hook_flow("UserPromptSubmit"))
    _turn_at(addon, used=0.96, resets_at=time.time() + 4 * 3600)

    flow = _FakeFlow(
        request=_FakeRequest(
            path="/v1/messages/count_tokens",
            text=json.dumps(_with_ttl(_body("claude-opus-4-8", 12), "1h")),
        )
    )
    addon.request(flow)
    addon.done()

    assert flow.response is None


def test_session_no_terminal_has_open_is_not_paused(tmp_path: Path) -> None:
    addon = _pausing_addon(tmp_path)
    _turn_at(addon, used=0.96, resets_at=time.time() + 4 * 3600)

    flow = _messages_flow(_with_ttl(_body("claude-opus-4-8", 12), "1h"))
    addon.request(flow)
    addon.done()

    assert flow.response is None
