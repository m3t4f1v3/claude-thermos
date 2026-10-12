import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import httpx

from claude_thermos.config import Config
from claude_thermos.credentials import StoredLogin
from claude_thermos.eventlog import EventLog
from claude_thermos.limits import LimitWindow, RateLimits
from claude_thermos.lineage import LineageId
from claude_thermos.state import SessionState
from claude_thermos.warmer import Warmer, build_warm_request


def _body(model: str, tool_count: int, system: str = "system prompt") -> dict:
    return {
        "model": model,
        "system": system,
        "tools": [{"name": f"tool-{i}"} for i in range(tool_count)],
        "messages": [{"role": "user", "content": "hi"}],
    }


_HEADERS = {"authorization": "Bearer abc"}
_MAIN = _body("claude-opus-4-8", 31)
_SUBAGENT = _body("claude-sonnet-5", 25, system="subagent system prompt")
_MAIN_ID = LineageId.from_request_body(_MAIN)
_SUBAGENT_ID = LineageId.from_request_body(_SUBAGENT)

_CONFIG = Config(
    idle_threshold_sec=270,
    warm_interval_sec=270,
    warm_max_cycles=2,
    subagent_active_window_sec=540,
)

_USAGE_RESPONSE = {
    "input_tokens": 0,
    "cache_read_input_tokens": 272360,
    "cache_creation_input_tokens": 12,
    "output_tokens": 0,
}


def _recording_client(requests: list[httpx.Request]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"usage": _USAGE_RESPONSE})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _client_from(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _idle_session(headers: dict = _HEADERS) -> SessionState:
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN, headers, now=0)
    state.on_response(_MAIN_ID, now=0)
    state.on_request(_SUBAGENT_ID, _SUBAGENT, headers, now=100)
    return state


def _events(tmp_path: Path) -> EventLog:
    return EventLog("sess-1", root=tmp_path)


def _read_events(tmp_path: Path) -> list[dict]:
    lines = (tmp_path / "sess-1" / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _event_names(tmp_path: Path) -> list[str]:
    return [e["event"] for e in _read_events(tmp_path)]


async def test_warms_when_idle_and_subagent_active(tmp_path: Path) -> None:
    state = _idle_session()
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    warmer = Warmer(_CONFIG, client=_recording_client(requests))

    await warmer.tick(state, log, now=280)
    log.close()

    assert len(requests) == 1
    assert requests[0].url.path == "/v1/messages"
    assert requests[0].headers["authorization"] == "Bearer abc"
    assert json.loads(requests[0].content)["max_tokens"] == 1

    events = _read_events(tmp_path)
    names = [e["event"] for e in events]
    assert "warm_fired" in names
    result = next(e for e in events if e["event"] == "warm_result")
    assert result["usage"]["cache_read"] == 272360


async def test_respects_interval(tmp_path: Path) -> None:
    state = _idle_session()
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    warmer = Warmer(_CONFIG, client=_recording_client(requests))

    await warmer.tick(state, log, now=280)
    await warmer.tick(state, log, now=500)
    log.close()

    assert len(requests) == 1


async def test_cap_stops_warming(tmp_path: Path) -> None:
    state = _idle_session()
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    warmer = Warmer(_CONFIG, client=_recording_client(requests))

    await warmer.tick(state, log, now=280)
    await warmer.tick(state, log, now=560)
    await warmer.tick(state, log, now=840)
    await warmer.tick(state, log, now=1120)
    log.close()

    assert len(requests) == 2
    assert _event_names(tmp_path).count("cap_reached") == 1


async def test_auto_ignores_cap(tmp_path: Path) -> None:
    state = _idle_session()
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    warmer = Warmer(Config(warm_max_cycles=None), client=_recording_client(requests))

    await warmer.tick(state, log, now=280)
    await warmer.tick(state, log, now=560)
    await warmer.tick(state, log, now=840)
    log.close()

    assert len(requests) == 3


async def test_resume_resets_and_stops(tmp_path: Path) -> None:
    state = _idle_session()
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    warmer = Warmer(_CONFIG, client=_recording_client(requests))

    await warmer.tick(state, log, now=280)
    state.on_request(_MAIN_ID, _MAIN, _HEADERS, now=410)
    await warmer.tick(state, log, now=500)
    await warmer.tick(state, log, now=800)
    log.close()

    assert len(requests) == 1
    assert "resume_detected" in _event_names(tmp_path)


async def test_no_warm_without_subagent(tmp_path: Path) -> None:
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN, _HEADERS, now=0)
    state.on_response(_MAIN_ID, now=0)
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    warmer = Warmer(_CONFIG, client=_recording_client(requests))

    await warmer.tick(state, log, now=280)
    log.close()

    assert len(requests) == 0


async def test_warm_error_on_http_error(tmp_path: Path) -> None:
    state = _idle_session()
    log = _events(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
        )

    warmer = Warmer(_CONFIG, client=_client_from(handler))

    await warmer.tick(state, log, now=280)
    log.close()

    events = _read_events(tmp_path)
    names = [e["event"] for e in events]
    assert "warm_fired" in names
    assert "warm_result" not in names
    error = next(e for e in events if e["event"] == "warm_error")
    assert error["error"] == "HTTP 429: slow down"


async def test_warm_error_on_network_error(tmp_path: Path) -> None:
    state = _idle_session()
    log = _events(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    warmer = Warmer(_CONFIG, client=_client_from(handler))

    await warmer.tick(state, log, now=280)
    log.close()

    events = _read_events(tmp_path)
    names = [e["event"] for e in events]
    assert "warm_result" not in names
    error = next(e for e in events if e["event"] == "warm_error")
    assert "boom" in error["error"]


async def test_warm_error_on_unexpected_exception(tmp_path: Path) -> None:
    state = _idle_session()
    log = _events(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("kaboom")

    warmer = Warmer(_CONFIG, client=_client_from(handler))

    await warmer.tick(state, log, now=280)
    log.close()

    events = _read_events(tmp_path)
    names = [e["event"] for e in events]
    assert "warm_result" not in names
    error = next(e for e in events if e["event"] == "warm_error")
    assert "kaboom" in error["error"]


async def test_warm_strips_replay_hostile_headers(tmp_path: Path) -> None:
    headers = {"authorization": "Bearer abc", "host": "127.0.0.1:8080", "content-length": "999999"}
    state = _idle_session(headers)
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    warmer = Warmer(_CONFIG, client=_recording_client(requests))

    await warmer.tick(state, log, now=280)
    log.close()

    assert len(requests) == 1
    sent = requests[0]
    assert sent.headers["authorization"] == "Bearer abc"
    assert sent.headers["host"] != "127.0.0.1:8080"
    assert sent.headers["content-length"] == str(len(sent.content))


def _stored_body() -> dict:
    return {
        "model": "claude-opus-4-8",
        "system": [
            {"type": "text", "text": "you are helpful"},
            {"type": "text", "text": "long prefix", "cache_control": {"type": "ephemeral"}},
        ],
        "tools": [{"name": "Read"}, {"name": "Edit"}],
        "messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "abc",
                        "content": "done",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ],
        "max_tokens": 4096,
        "thinking": {"type": "adaptive"},
        "stream": True,
        "output_config": {"foo": "bar"},
        "context_management": {"edits": []},
    }


def test_build_warm_request_preserves_prefix() -> None:
    body = _stored_body()
    warm = build_warm_request(body)
    assert warm["system"] == body["system"]
    assert warm["tools"] == body["tools"]
    assert warm["messages"] == body["messages"]


def test_build_warm_request_neutralizes_generation_params() -> None:
    warm = build_warm_request(_stored_body())
    assert warm["max_tokens"] == 1
    assert warm["stream"] is False
    assert "output_config" not in warm


def test_build_warm_request_preserves_message_cache_keys() -> None:
    body = _stored_body()
    body["tool_choice"] = {"type": "any"}
    warm = build_warm_request(body)
    assert warm["thinking"] == body["thinking"]
    assert warm["tool_choice"] == body["tool_choice"]
    assert warm["context_management"] == body["context_management"]


def test_build_warm_request_does_not_mutate_input() -> None:
    body = _stored_body()
    original = _stored_body()
    build_warm_request(body)
    assert body == original


def _with_ttl(body: dict, ttl: str) -> dict:
    marked = {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": ttl}}
    return {**body, "messages": [{"role": "user", "content": [marked]}]}


_MAIN_1H = _with_ttl(_MAIN, "1h")

_KEEPALIVE = Config(
    idle_threshold_sec=270,
    warm_interval_sec=270,
    warm_max_cycles=None,
    keepalive=True,
    ping_budget=0.10,
    session_reserve=0.10,
    compact=True,
    compact_min_tokens=100_000,
)


def _limits(session: float, weekly: float = 0.2, status: str = "allowed") -> RateLimits:
    return RateLimits(
        {
            "5h": LimitWindow(utilization=session, resets_at=18_000, status=status),
            "7d": LimitWindow(utilization=weekly, resets_at=600_000, status="allowed"),
        }
    )


def _ended_session(
    limits: RateLimits | None = None,
    live: bool = True,
    stop_reason: str = "end_turn",
    input_tokens: int = 250_000,
    headers: dict = _HEADERS,
) -> SessionState:
    """A session whose one-hour main turn finished at t=0, with no subagent."""
    state = SessionState("sess-1")
    state.live = live
    state.limits = limits
    state.on_request(_MAIN_ID, _MAIN_1H, headers, now=0)
    state.on_response(_MAIN_ID, now=0, stop_reason=stop_reason, input_tokens=input_tokens)
    return state


def _metered_client(requests: list[httpx.Request], utilizations: list[float]) -> httpx.AsyncClient:
    """A client whose Nth response reports the Nth session utilization."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        used = utilizations[min(len(requests), len(utilizations)) - 1]
        headers = {
            "anthropic-ratelimit-unified-5h-utilization": str(used),
            "anthropic-ratelimit-unified-5h-reset": "18000",
            "anthropic-ratelimit-unified-5h-status": "allowed",
        }
        return httpx.Response(200, json={"usage": _USAGE_RESPONSE}, headers=headers)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class _Compactor:
    """Records compaction requests and answers with a fixed outcome."""

    def __init__(self, outcome: str | None = None) -> None:
        self.outcome = outcome
        self.session_ids: list[str] = []

    def __call__(self, session_id: str) -> str | None:
        self.session_ids.append(session_id)
        return self.outcome


def _stop_reasons(tmp_path: Path) -> list[str]:
    return [e["reason"] for e in _read_events(tmp_path) if e["event"] == "cap_reached"]


async def test_keepalive_warms_an_ended_turn_before_its_hour_is_up(tmp_path: Path) -> None:
    state = _ended_session()
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    warmer = Warmer(_KEEPALIVE, client=_recording_client(requests))

    await warmer.tick(state, log, now=3000)
    assert len(requests) == 0

    await warmer.tick(state, log, now=3300)
    log.close()

    assert len(requests) == 1
    result = next(e for e in _read_events(tmp_path) if e["event"] == "warm_result")
    assert result["ttl_sec"] == 3600
    assert result["ttl_refreshed"] is True


async def test_keepalive_repeats_every_interval(tmp_path: Path) -> None:
    state = _ended_session()
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    warmer = Warmer(_KEEPALIVE, client=_recording_client(requests))

    for now in (3300, 5000, 6600, 9900):
        await warmer.tick(state, log, now=now)
    log.close()

    assert len(requests) == 3


async def test_keepalive_skips_a_session_no_terminal_has_open(tmp_path: Path) -> None:
    state = _ended_session(live=False)
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    warmer = Warmer(_KEEPALIVE, client=_recording_client(requests))

    await warmer.tick(state, log, now=3300)
    log.close()

    assert len(requests) == 0


async def test_subagent_wait_on_a_one_hour_prefix_uses_the_long_interval(tmp_path: Path) -> None:
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN_1H, _HEADERS, now=0)
    state.on_response(_MAIN_ID, now=0)
    state.on_request(_SUBAGENT_ID, _SUBAGENT, _HEADERS, now=100)
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    warmer = Warmer(_CONFIG, client=_recording_client(requests))

    await warmer.tick(state, log, now=280)
    assert len(requests) == 0

    await warmer.tick(state, log, now=3300)
    log.close()

    assert len(requests) == 1


async def test_expired_prefix_is_not_rewarmed_or_compacted(tmp_path: Path) -> None:
    state = _ended_session()
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    compactor = _Compactor()
    warmer = Warmer(_KEEPALIVE, client=_recording_client(requests), compactor=compactor)

    # The machine slept through the warm that was due at 3300.
    await warmer.tick(state, log, now=3700)
    await warmer.tick(state, log, now=3705)
    log.close()

    assert len(requests) == 0
    assert _stop_reasons(tmp_path) == ["cache_expired"]
    assert compactor.session_ids == []


async def test_ping_budget_ends_the_episode_with_a_compaction(tmp_path: Path) -> None:
    state = _ended_session(_limits(0.40))
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    compactor = _Compactor()
    warmer = Warmer(_KEEPALIVE, client=_metered_client(requests, [0.44, 0.48]), compactor=compactor)

    # Each warm costs 4% of the session; a third would overrun the 10% budget.
    for now in (3300, 6600, 9900, 9905):
        await warmer.tick(state, log, now=now)
    log.close()

    assert len(requests) == 2
    assert _stop_reasons(tmp_path) == ["ping_budget_spent"]
    assert compactor.session_ids == ["sess-1"]
    events = _read_events(tmp_path)
    last_warm = [e for e in events if e["event"] == "warm_result"][-1]
    assert last_warm["session_utilization"] == 0.48
    assert last_warm["ping_spent"] == 0.08
    requested = next(e for e in events if e["event"] == "compact_requested")
    assert requested["reason"] == "ping_budget_spent"
    assert state.main_compacted() is True


async def test_zero_ping_budget_compacts_without_warming(tmp_path: Path) -> None:
    state = _ended_session(_limits(0.10))
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    compactor = _Compactor()
    config = replace(_KEEPALIVE, ping_budget=0.0)
    warmer = Warmer(config, client=_recording_client(requests), compactor=compactor)

    await warmer.tick(state, log, now=3300)
    log.close()

    assert len(requests) == 0
    assert _stop_reasons(tmp_path) == ["ping_budget_spent"]
    assert compactor.session_ids == ["sess-1"]


async def test_session_reserve_compacts_instead_of_warming(tmp_path: Path) -> None:
    state = _ended_session(_limits(0.95))
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    compactor = _Compactor()
    warmer = Warmer(_KEEPALIVE, client=_recording_client(requests), compactor=compactor)

    await warmer.tick(state, log, now=3300)
    log.close()

    assert len(requests) == 0
    assert _stop_reasons(tmp_path) == ["session_limit_low"]
    assert compactor.session_ids == ["sess-1"]


async def test_session_reserve_counts_the_next_warm(tmp_path: Path) -> None:
    state = _ended_session(_limits(0.80))
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    compactor = _Compactor()
    config = replace(_KEEPALIVE, ping_budget=0.50)
    warmer = Warmer(config, client=_metered_client(requests, [0.86]), compactor=compactor)

    # 14% is left after the first warm; another 6% would dip into the 10% reserve.
    await warmer.tick(state, log, now=3300)
    await warmer.tick(state, log, now=6600)
    log.close()

    assert len(requests) == 1
    assert _stop_reasons(tmp_path) == ["session_limit_low"]


async def test_weekly_limit_counts_against_the_reserve(tmp_path: Path) -> None:
    state = _ended_session(_limits(0.10, weekly=0.95))
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    warmer = Warmer(_KEEPALIVE, client=_recording_client(requests))

    await warmer.tick(state, log, now=3300)
    log.close()

    assert len(requests) == 0
    assert _stop_reasons(tmp_path) == ["session_limit_low"]


async def test_exhausted_limit_stops_without_compacting(tmp_path: Path) -> None:
    state = _ended_session(_limits(1.0, status="rejected"))
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    compactor = _Compactor()
    warmer = Warmer(_KEEPALIVE, client=_recording_client(requests), compactor=compactor)

    await warmer.tick(state, log, now=3300)
    log.close()

    assert len(requests) == 0
    assert _stop_reasons(tmp_path) == ["limit_reached"]
    assert compactor.session_ids == []


async def test_stale_reading_from_a_window_that_has_reset_does_not_block(tmp_path: Path) -> None:
    limits = RateLimits({"5h": LimitWindow(utilization=0.99, resets_at=3000, status="rejected")})
    state = _ended_session(limits)
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    warmer = Warmer(_KEEPALIVE, client=_metered_client(requests, [0.02]))

    await warmer.tick(state, log, now=3300)
    log.close()

    assert len(requests) == 1
    result = next(e for e in _read_events(tmp_path) if e["event"] == "warm_result")
    assert result["ping_spent"] == 0.02


async def test_cycle_cap_applies_alongside_the_budget(tmp_path: Path) -> None:
    state = _ended_session(_limits(0.10))
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    compactor = _Compactor()
    config = replace(_KEEPALIVE, warm_max_cycles=1)
    warmer = Warmer(config, client=_metered_client(requests, [0.10]), compactor=compactor)

    await warmer.tick(state, log, now=3300)
    await warmer.tick(state, log, now=6600)
    log.close()

    assert len(requests) == 1
    assert _stop_reasons(tmp_path) == ["max_cycles"]
    assert compactor.session_ids == ["sess-1"]


async def test_unfinished_turn_is_not_compacted(tmp_path: Path) -> None:
    state = _ended_session(_limits(0.95), stop_reason="tool_use")
    log = _events(tmp_path)
    compactor = _Compactor()
    warmer = Warmer(_KEEPALIVE, client=_recording_client([]), compactor=compactor)

    await warmer.tick(state, log, now=3300)
    log.close()

    assert compactor.session_ids == []
    skipped = next(e for e in _read_events(tmp_path) if e["event"] == "compact_skipped")
    assert skipped["reason"] == "turn_in_progress"
    assert state.main_compacted() is False


async def test_small_conversation_is_not_compacted(tmp_path: Path) -> None:
    state = _ended_session(_limits(0.95), input_tokens=40_000)
    log = _events(tmp_path)
    compactor = _Compactor()
    warmer = Warmer(_KEEPALIVE, client=_recording_client([]), compactor=compactor)

    await warmer.tick(state, log, now=3300)
    log.close()

    assert compactor.session_ids == []
    skipped = next(e for e in _read_events(tmp_path) if e["event"] == "compact_skipped")
    assert skipped["reason"] == "context_small"


async def test_compactor_refusal_is_logged_and_not_retried(tmp_path: Path) -> None:
    state = _ended_session(_limits(0.95))
    log = _events(tmp_path)
    compactor = _Compactor(outcome="input_not_empty")
    warmer = Warmer(_KEEPALIVE, client=_recording_client([]), compactor=compactor)

    await warmer.tick(state, log, now=3300)
    await warmer.tick(state, log, now=3305)
    log.close()

    assert compactor.session_ids == ["sess-1"]
    names = _event_names(tmp_path)
    assert names.count("compact_skipped") == 1
    assert "compact_requested" not in names
    assert state.main_compacted() is False


async def test_no_compaction_without_the_option_or_a_compactor(tmp_path: Path) -> None:
    log = _events(tmp_path)
    compactor = _Compactor()
    disabled = Warmer(
        replace(_KEEPALIVE, compact=False), client=_recording_client([]), compactor=compactor
    )
    unequipped = Warmer(_KEEPALIVE, client=_recording_client([]))

    await disabled.tick(_ended_session(_limits(0.95)), log, now=3300)
    await unequipped.tick(_ended_session(_limits(0.95)), log, now=3300)
    log.close()

    assert compactor.session_ids == []
    assert "compact_requested" not in _event_names(tmp_path)


async def test_capped_subagent_wait_does_not_compact_a_session_that_is_not_live(
    tmp_path: Path,
) -> None:
    state = _idle_session()
    state.on_response(_MAIN_ID, now=0, stop_reason="end_turn", input_tokens=250_000)
    log = _events(tmp_path)
    compactor = _Compactor()
    config = replace(_CONFIG, warm_max_cycles=0, compact=True)
    warmer = Warmer(config, client=_recording_client([]), compactor=compactor)

    await warmer.tick(state, log, now=280)
    log.close()

    assert _stop_reasons(tmp_path) == ["max_cycles"]
    assert compactor.session_ids == []


async def test_compacted_conversation_is_not_warmed(tmp_path: Path) -> None:
    state = _ended_session(_limits(0.10))
    state.on_compacted(now=60)
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    compactor = _Compactor()
    warmer = Warmer(_KEEPALIVE, client=_recording_client(requests), compactor=compactor)

    await warmer.tick(state, log, now=3300)
    log.close()

    assert len(requests) == 0
    assert _stop_reasons(tmp_path) == ["compacted"]
    assert compactor.session_ids == []


async def test_warm_that_misses_the_cache_stops_the_episode(tmp_path: Path) -> None:
    state = _ended_session()
    log = _events(tmp_path)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        usage = {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 250_000}
        return httpx.Response(200, json={"usage": usage})

    compactor = _Compactor()
    warmer = Warmer(_KEEPALIVE, client=_client_from(handler), compactor=compactor)

    await warmer.tick(state, log, now=3300)
    await warmer.tick(state, log, now=6600)
    log.close()

    assert len(requests) == 1
    assert _stop_reasons(tmp_path) == ["cache_miss"]
    assert compactor.session_ids == []
    result = next(e for e in _read_events(tmp_path) if e["event"] == "warm_result")
    assert result["ttl_refreshed"] is False


async def test_failed_warm_is_retried_while_the_prefix_lives(tmp_path: Path) -> None:
    state = _ended_session()
    log = _events(tmp_path)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ConnectError("offline")
        return httpx.Response(200, json={"usage": _USAGE_RESPONSE})

    warmer = Warmer(replace(_KEEPALIVE, warm_max_cycles=1), client=_client_from(handler))

    await warmer.tick(state, log, now=3300)
    await warmer.tick(state, log, now=3310)
    assert len(requests) == 1

    await warmer.tick(state, log, now=3360)
    log.close()

    assert len(requests) == 2
    names = _event_names(tmp_path)
    assert names.count("warm_error") == 1
    assert names.count("warm_result") == 1
    # The failure did not use up the single allowed cycle.
    assert "cap_reached" not in names


async def test_failed_warm_gives_up_when_the_prefix_expires(tmp_path: Path) -> None:
    state = _ended_session()
    log = _events(tmp_path)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectError("offline")

    warmer = Warmer(_KEEPALIVE, client=_client_from(handler))

    for now in (3300, 3360, 3420, 3480, 3540, 3600, 3660):
        await warmer.tick(state, log, now=now)
    log.close()

    assert len(requests) == 5
    assert _stop_reasons(tmp_path) == ["cache_expired"]


async def test_resume_starts_a_fresh_budget(tmp_path: Path) -> None:
    state = _ended_session(_limits(0.40))
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    warmer = Warmer(_KEEPALIVE, client=_metered_client(requests, [0.48, 0.50]))

    await warmer.tick(state, log, now=3300)
    state.on_request(_MAIN_ID, _MAIN_1H, _HEADERS, now=4000)
    state.on_response(_MAIN_ID, now=4010, stop_reason="end_turn", input_tokens=250_000)
    await warmer.tick(state, log, now=4020)
    await warmer.tick(state, log, now=7300)
    log.close()

    assert len(requests) == 2
    results = [e for e in _read_events(tmp_path) if e["event"] == "warm_result"]
    assert [e["ping_spent"] for e in results] == [0.08, 0.02]


_PAUSING = replace(_KEEPALIVE, pause_at=0.95, warm_max_cycles=1)


def _events_named(tmp_path: Path, name: str) -> list[dict]:
    return [e for e in _read_events(tmp_path) if e["event"] == name]


async def test_paused_turn_is_warmed_past_the_caps_until_the_reset(tmp_path: Path) -> None:
    # 96% of the session limit is used, and it resets at t=18000, long after
    # the hour the cache lasts: the turn pauses, and the rest of the limit
    # keeps it warm, past the cycle cap and through the reserve.
    state = _ended_session(_limits(0.96), stop_reason="tool_use")
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    compactor = _Compactor()
    client = _metered_client(requests, [0.96, 0.97, 0.97, 0.98])
    warmer = Warmer(_PAUSING, client=client, compactor=compactor)

    for now in (3300, 6600, 9900, 13200, 16500):
        await warmer.tick(state, log, now=now)
    log.close()

    assert len(requests) == 5
    assert _stop_reasons(tmp_path) == []
    assert compactor.session_ids == []
    paused = _events_named(tmp_path, "paused")
    assert [e["resumes_at"] for e in paused] == [18_000]


async def test_ended_turn_past_the_threshold_is_also_kept_warm(tmp_path: Path) -> None:
    state = _ended_session(_limits(0.96))
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    compactor = _Compactor()
    warmer = Warmer(_PAUSING, client=_metered_client(requests, [0.96]), compactor=compactor)

    await warmer.tick(state, log, now=3300)
    await warmer.tick(state, log, now=6600)
    log.close()

    assert len(requests) == 2
    assert compactor.session_ids == []


async def test_pause_lifts_at_the_reset_and_starts_a_new_episode(tmp_path: Path) -> None:
    state = _ended_session(_limits(0.96), stop_reason="tool_use")
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    compactor = _Compactor()
    client = _metered_client(requests, [0.96, 0.97, 0.97, 0.98, 0.98, 0.02])
    warmer = Warmer(_PAUSING, client=client, compactor=compactor)

    for now in (3300, 6600, 9900, 13200, 16500, 19800):
        await warmer.tick(state, log, now=now)
    log.close()

    assert state.paused_until is None
    assert [e["reason"] for e in _events_named(tmp_path, "pause_lifted")] == ["reset"]
    # The first warm of the new window is within the fresh episode's cap.
    assert len(requests) == 6
    assert _stop_reasons(tmp_path) == []


async def test_reset_within_the_cache_lifetime_does_not_pause(tmp_path: Path) -> None:
    # Resetting 45 minutes out, the limit comes back before an hour-long
    # cache could expire, so stopping early would buy nothing.
    limits = RateLimits({"5h": LimitWindow(utilization=0.96, resets_at=6000, status="allowed")})
    state = _ended_session(limits)
    log = _events(tmp_path)
    warmer = Warmer(_PAUSING, client=_recording_client([]), compactor=_Compactor())

    await warmer.tick(state, log, now=3300)
    log.close()

    assert state.paused_until is None
    assert _events_named(tmp_path, "paused") == []
    assert _stop_reasons(tmp_path) == ["session_limit_low"]


async def test_paused_warms_stop_at_the_limit_itself(tmp_path: Path) -> None:
    state = _ended_session(_limits(0.96), stop_reason="tool_use")
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    warmer = Warmer(_PAUSING, client=_recording_client(requests))
    await warmer.tick(state, log, now=60)  # pauses
    state.limits = RateLimits(
        {"5h": LimitWindow(utilization=1.0, resets_at=18_000, status="rejected")}
    )

    await warmer.tick(state, log, now=3300)
    log.close()

    assert len(requests) == 0
    assert _stop_reasons(tmp_path) == ["limit_reached"]


_OLD_TOKEN = "sk-ant-oat01-old"
_NEW_TOKEN = "sk-ant-oat01-new"
_REVOKED = {
    "type": "error",
    "error": {"type": "authentication_error", "message": "OAuth access token has been revoked."},
}


def _bearer(token: str) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


def _client_accepting(requests: list[httpx.Request], token: str | None) -> httpx.AsyncClient:
    """A client that answers 401, as the API does to a revoked token, unless
    the request carries `token`."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if token is None or request.headers.get("authorization") != f"Bearer {token}":
            return httpx.Response(401, json=_REVOKED)
        return httpx.Response(200, json={"usage": _USAGE_RESPONSE})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _write_login(tmp_path: Path, token: str) -> StoredLogin:
    path = tmp_path / ".credentials.json"
    path.write_text(json.dumps({"claudeAiOauth": {"accessToken": token}}))
    return StoredLogin(path)


def _sent_tokens(requests: list[httpx.Request]) -> list[str]:
    return [request.headers["authorization"].removeprefix("Bearer ") for request in requests]


async def test_warm_uses_the_newest_credentials_in_the_session(tmp_path: Path) -> None:
    # Claude Code swapped its token while the main agent waited on a subagent:
    # the main lineage's last request carries the old one, the subagent's the new.
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN, {**_bearer(_OLD_TOKEN), "anthropic-beta": "main"}, now=0)
    state.on_response(_MAIN_ID, now=0)
    state.on_request(_SUBAGENT_ID, _SUBAGENT, {"Authorization": f"Bearer {_NEW_TOKEN}"}, now=100)
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    warmer = Warmer(_CONFIG, client=_client_accepting(requests, _NEW_TOKEN))

    await warmer.tick(state, log, now=280)
    log.close()

    assert _sent_tokens(requests) == [_NEW_TOKEN]
    assert requests[0].headers["anthropic-beta"] == "main"
    assert "warm_result" in _event_names(tmp_path)


async def test_revoked_token_is_replaced_from_the_stored_login(tmp_path: Path) -> None:
    state = _ended_session(headers=_bearer(_OLD_TOKEN))
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    login = _write_login(tmp_path, _OLD_TOKEN)
    warmer = Warmer(_KEEPALIVE, client=_client_accepting(requests, _NEW_TOKEN), login=login)

    await warmer.tick(state, log, now=60)
    # Claude Code swaps its token while the session sits idle, sending nothing.
    _write_login(tmp_path, _NEW_TOKEN)
    await warmer.tick(state, log, now=3300)
    await warmer.tick(state, log, now=6600)
    log.close()

    # The second warm goes straight out with the new token.
    assert _sent_tokens(requests) == [_OLD_TOKEN, _NEW_TOKEN, _NEW_TOKEN]
    assert state.credentials == _bearer(_NEW_TOKEN)
    events = _read_events(tmp_path)
    errors = [e for e in events if e["event"] == "warm_error"]
    assert [(e["cycle"], e["error"], e["retried"]) for e in errors] == [
        (1, "HTTP 401: OAuth access token has been revoked.", True)
    ]
    results = [e for e in events if e["event"] == "warm_result"]
    assert [(e["cycle"], e["ttl_refreshed"]) for e in results] == [(1, True), (2, True)]
    assert [e["event"] for e in events].count("warm_fired") == 2
    assert _stop_reasons(tmp_path) == []


async def test_unauthorized_warm_stops_instead_of_retrying(tmp_path: Path) -> None:
    state = _ended_session(_limits(0.10), headers=_bearer(_OLD_TOKEN))
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    compactor = _Compactor()
    warmer = Warmer(_KEEPALIVE, client=_client_accepting(requests, None), compactor=compactor)

    for now in (3300, 3360, 3420, 3480, 3540, 3600, 3660):
        await warmer.tick(state, log, now=now)
    log.close()

    assert len(requests) == 1
    assert _stop_reasons(tmp_path) == ["unauthorized"]
    assert compactor.session_ids == []
    errors = _events_named(tmp_path, "warm_error")
    assert len(errors) == 1
    assert "retried" not in errors[0]


async def test_login_of_another_account_is_not_sent(tmp_path: Path) -> None:
    # The stored login never held the session's token, so it is not this
    # session's: its token would rewrite the prefix under the wrong account.
    state = _ended_session(headers=_bearer(_OLD_TOKEN))
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    login = _write_login(tmp_path, _NEW_TOKEN)
    warmer = Warmer(_KEEPALIVE, client=_client_accepting(requests, _NEW_TOKEN), login=login)

    await warmer.tick(state, log, now=60)
    await warmer.tick(state, log, now=3300)
    log.close()

    assert _sent_tokens(requests) == [_OLD_TOKEN]
    assert _stop_reasons(tmp_path) == ["unauthorized"]


async def test_refused_replacement_stops_the_episode(tmp_path: Path) -> None:
    state = _ended_session(headers=_bearer(_OLD_TOKEN))
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    login = _write_login(tmp_path, _OLD_TOKEN)
    warmer = Warmer(_KEEPALIVE, client=_client_accepting(requests, None), login=login)

    await warmer.tick(state, log, now=60)
    _write_login(tmp_path, _NEW_TOKEN)
    await warmer.tick(state, log, now=3300)
    await warmer.tick(state, log, now=3360)
    log.close()

    assert _sent_tokens(requests) == [_OLD_TOKEN, _NEW_TOKEN]
    assert _stop_reasons(tmp_path) == ["unauthorized"]
    assert [e.get("retried", False) for e in _events_named(tmp_path, "warm_error")] == [True, False]


async def test_new_credentials_from_a_subagent_restart_an_unauthorized_episode(
    tmp_path: Path,
) -> None:
    state = _idle_session(_bearer(_OLD_TOKEN))
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    warmer = Warmer(_CONFIG, client=_client_accepting(requests, _NEW_TOKEN))

    await warmer.tick(state, log, now=280)
    await warmer.tick(state, log, now=285)
    assert _sent_tokens(requests) == [_OLD_TOKEN]

    state.on_request(_SUBAGENT_ID, _SUBAGENT, _bearer(_NEW_TOKEN), now=286)
    await warmer.tick(state, log, now=290)
    log.close()

    assert _sent_tokens(requests) == [_OLD_TOKEN, _NEW_TOKEN]
    assert _stop_reasons(tmp_path) == ["unauthorized"]
    assert _event_names(tmp_path).count("warm_result") == 1


async def test_resume_clears_an_unauthorized_stop(tmp_path: Path) -> None:
    state = _ended_session(headers=_bearer(_OLD_TOKEN))
    log = _events(tmp_path)
    requests: list[httpx.Request] = []
    warmer = Warmer(_KEEPALIVE, client=_client_accepting(requests, _NEW_TOKEN))

    await warmer.tick(state, log, now=3300)
    state.on_request(_MAIN_ID, _MAIN_1H, _bearer(_NEW_TOKEN), now=4000)
    state.on_response(_MAIN_ID, now=4010, stop_reason="end_turn", input_tokens=250_000)
    await warmer.tick(state, log, now=4020)
    await warmer.tick(state, log, now=7300)
    log.close()

    assert _sent_tokens(requests) == [_OLD_TOKEN, _NEW_TOKEN]
    assert _stop_reasons(tmp_path) == ["unauthorized"]
