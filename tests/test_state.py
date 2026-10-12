from claude_thermos.limits import LimitWindow, RateLimits
from claude_thermos.lineage import LineageId
from claude_thermos.state import SessionState


def _body(model: str, tool_count: int, system: str = "system prompt") -> dict:
    return {
        "model": model,
        "system": system,
        "tools": [{"name": f"tool-{i}"} for i in range(tool_count)],
        "messages": [{"role": "user", "content": "hi"}],
    }


_HEADERS = {"authorization": "Bearer abc"}

_PING = _body("claude-opus-4-8", 0)
_MAIN = _body("claude-opus-4-8", 31)
_SUBAGENT = _body("claude-sonnet-5", 25, system="subagent system prompt")

_PING_ID = LineageId.from_request_body(_PING)
_MAIN_ID = LineageId.from_request_body(_MAIN)
_SUBAGENT_ID = LineageId.from_request_body(_SUBAGENT)


def test_main_is_first_substantive_lineage() -> None:
    state = SessionState("sess-1")
    state.on_request(_PING_ID, _PING, _HEADERS, now=0)
    state.on_request(_MAIN_ID, _MAIN, _HEADERS, now=1)
    assert state.main_lineage_id() == _MAIN_ID


def test_main_idle_after_threshold() -> None:
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN, _HEADERS, now=0)
    state.on_response(_MAIN_ID, now=0)
    state.on_request(_SUBAGENT_ID, _SUBAGENT, _HEADERS, now=100)
    state.on_response(_SUBAGENT_ID, now=110)
    assert state.is_main_idle(now=200, idle_threshold_sec=270) is False
    assert state.is_main_idle(now=300, idle_threshold_sec=270) is True


def test_main_not_idle_while_in_flight() -> None:
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN, _HEADERS, now=0)
    assert state.is_main_idle(now=10_000, idle_threshold_sec=1) is False


def test_subagent_active_during_gap() -> None:
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN, _HEADERS, now=0)
    state.on_response(_MAIN_ID, now=0)
    state.on_request(_SUBAGENT_ID, _SUBAGENT, _HEADERS, now=100)
    state.on_response(_SUBAGENT_ID, now=110)
    assert state.subagent_active(now=110, window_sec=540) is True
    assert state.subagent_active(now=110 + 540, window_sec=540) is False


def test_subagent_active_ignores_main() -> None:
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN, _HEADERS, now=0)
    assert state.subagent_active(now=0, window_sec=540) is False


def _with_ttl(body: dict, ttl: str) -> dict:
    marked = {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": ttl}}
    return {**body, "messages": [{"role": "user", "content": [marked]}]}


_MAIN_1H = _with_ttl(_MAIN, "1h")
_MAIN_5M = _with_ttl(_MAIN, "5m")


def test_shorter_ttl_on_the_same_lineage_is_a_side_request() -> None:
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN_1H, _HEADERS, now=0)

    assert state.is_side_request(_MAIN_ID, _MAIN_5M, now=10) is True
    assert state.is_side_request(_MAIN_ID, _MAIN_1H, now=10) is False


def test_first_request_of_a_lineage_is_never_a_side_request() -> None:
    state = SessionState("sess-1")

    assert state.is_side_request(_MAIN_ID, _MAIN_5M, now=0) is False


def test_shorter_ttl_is_the_conversation_once_the_longer_prefix_has_expired() -> None:
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN_1H, _HEADERS, now=0)

    assert state.is_side_request(_MAIN_ID, _MAIN_5M, now=3599) is True
    assert state.is_side_request(_MAIN_ID, _MAIN_5M, now=3600) is False


def test_longer_ttl_is_adopted() -> None:
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN_5M, _HEADERS, now=0)

    assert state.is_side_request(_MAIN_ID, _MAIN_1H, now=10) is False
    state.on_request(_MAIN_ID, _MAIN_1H, _HEADERS, now=10)
    main = state.main_lineage()
    assert main is not None
    assert main.cache_ttl_sec == 3600


def test_main_turn_ended_follows_the_last_stop_reason() -> None:
    state = SessionState("sess-1")
    assert state.main_turn_ended() is False

    state.on_request(_MAIN_ID, _MAIN, _HEADERS, now=0)
    assert state.main_turn_ended() is False  # in flight

    state.on_response(_MAIN_ID, now=1, stop_reason="tool_use")
    assert state.main_turn_ended() is False

    state.on_request(_MAIN_ID, _MAIN, _HEADERS, now=2)
    state.on_response(_MAIN_ID, now=3, stop_reason="end_turn")
    assert state.main_turn_ended() is True


def test_response_records_prompt_size_and_keeps_it_when_unknown() -> None:
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN, _HEADERS, now=0)
    state.on_response(_MAIN_ID, now=1, stop_reason="end_turn", input_tokens=250_000)
    state.on_request(_MAIN_ID, _MAIN, _HEADERS, now=2)
    state.on_response(_MAIN_ID, now=3)

    main = state.main_lineage()
    assert main is not None
    assert main.last_input_tokens == 250_000


def test_compaction_obsoletes_the_prefix_until_the_next_request() -> None:
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN, _HEADERS, now=0)
    assert state.main_compacted() is False

    state.on_compacted(now=10)
    assert state.main_compacted() is True

    state.on_request(_MAIN_ID, _MAIN, _HEADERS, now=20)
    assert state.main_compacted() is False


_REBUILT_1H = _with_ttl(_body("claude-opus-4-8", 29), "1h")
_REBUILT_ID = LineageId.from_request_body(_REBUILT_1H)
_SUBAGENT_5M = _with_ttl(_SUBAGENT, "5m")


def test_lineage_continuing_a_compacted_conversation_becomes_main() -> None:
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN_1H, _HEADERS, now=0)
    state.on_compacted(now=10)

    state.on_request(_REBUILT_ID, _REBUILT_1H, _HEADERS, now=11)

    assert state.main_lineage_id() == _REBUILT_ID
    assert state.main_compacted() is False


def test_new_lineage_without_a_compaction_is_a_subagent() -> None:
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN_1H, _HEADERS, now=0)

    state.on_request(_REBUILT_ID, _REBUILT_1H, _HEADERS, now=11)

    assert state.main_lineage_id() == _MAIN_ID


def test_shorter_ttl_subagent_after_a_compaction_does_not_take_over() -> None:
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN_1H, _HEADERS, now=0)
    state.on_compacted(now=10)

    state.on_request(_SUBAGENT_ID, _SUBAGENT_5M, _HEADERS, now=11)
    assert state.main_lineage_id() == _MAIN_ID

    state.on_request(_REBUILT_ID, _REBUILT_1H, _HEADERS, now=12)
    assert state.main_lineage_id() == _REBUILT_ID


def test_main_that_resumes_after_a_compaction_keeps_its_place() -> None:
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN_1H, _HEADERS, now=0)
    state.on_compacted(now=10)
    state.on_request(_MAIN_ID, _MAIN_1H, _HEADERS, now=11)

    state.on_request(_REBUILT_ID, _REBUILT_1H, _HEADERS, now=12)

    assert state.main_lineage_id() == _MAIN_ID


def _near_limit(
    used: float = 0.96, resets_at: float = 18_000, overage: str | None = None
) -> RateLimits:
    window = LimitWindow(utilization=used, resets_at=resets_at, status="allowed_warning")
    return RateLimits({"5h": window}, overage_status=overage)


def _live_session(limits: RateLimits) -> SessionState:
    state = SessionState("sess-1")
    state.live = True
    state.limits = limits
    state.on_request(_MAIN_ID, _MAIN_1H, _HEADERS, now=0)
    return state


def test_pauses_past_the_threshold_until_the_session_limit_resets() -> None:
    state = _live_session(_near_limit())

    assert state.update_pause(now=100, pause_at=0.95) is True
    assert state.paused_until == 18_000


def test_below_the_threshold_does_not_pause() -> None:
    state = _live_session(_near_limit(used=0.90))

    assert state.update_pause(now=100, pause_at=0.95) is False


def test_paid_extra_usage_means_no_pause() -> None:
    state = _live_session(_near_limit(overage="allowed"))

    assert state.update_pause(now=100, pause_at=0.95) is False


def test_session_without_a_terminal_does_not_pause() -> None:
    state = _live_session(_near_limit())
    state.live = False

    assert state.update_pause(now=100, pause_at=0.95) is False


def test_pause_holds_until_the_reset_even_once_the_reset_is_near() -> None:
    state = _live_session(_near_limit())
    state.update_pause(now=100, pause_at=0.95)

    # Fifteen minutes before the reset a new pause would not start.
    assert state.update_pause(now=17_100, pause_at=0.95) is True
    assert state.update_pause(now=18_000, pause_at=0.95) is False
    assert state.paused_until is None


def test_prompt_past_the_threshold_waives_the_pause_until_the_reset() -> None:
    state = _live_session(_near_limit())
    state.update_pause(now=100, pause_at=0.95)

    assert state.waive_pause(now=200, pause_at=0.95) is True
    assert state.update_pause(now=300, pause_at=0.95) is False


def test_prompt_below_the_threshold_waives_nothing() -> None:
    state = _live_session(_near_limit(used=0.50))

    assert state.waive_pause(now=100, pause_at=0.95) is False
    state.limits = _near_limit(used=0.96)
    assert state.update_pause(now=200, pause_at=0.95) is True


def test_session_keeps_the_credentials_of_its_newest_request() -> None:
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN, {"Authorization": "Bearer old"}, now=0)
    state.on_request(_SUBAGENT_ID, _SUBAGENT, {"Authorization": "Bearer new"}, now=100)

    assert state.credentials == {"authorization": "Bearer new"}
    main = state.main_lineage()
    assert main is not None
    assert main.auth_headers == {"Authorization": "Bearer old"}


def test_request_without_credentials_keeps_the_last_ones() -> None:
    state = SessionState("sess-1")
    state.on_request(_MAIN_ID, _MAIN, _HEADERS, now=0)
    state.note_credentials({"anthropic-beta": "x"})

    assert state.credentials == _HEADERS
