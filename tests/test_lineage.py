import json

from claude_thermos.lineage import LineageId, cache_ttl_sec, extract_session_id


def _body(
    model: str,
    tool_names: list[str],
    system: str | list[dict[str, object]],
    session_id: str | None = "sess-main",
) -> dict:
    body: dict = {
        "model": model,
        "system": system,
        "tools": [{"name": name} for name in tool_names],
        "messages": [{"role": "user", "content": "hi"}],
    }
    if session_id is not None:
        body["metadata"] = {"user_id": json.dumps({"session_id": session_id, "org_id": "org-1"})}
    return body


def test_session_id_extracted() -> None:
    body = _body("claude-opus-4-8", ["Read"], "you are helpful", session_id="sess-main")
    assert extract_session_id(body) == "sess-main"


def test_session_id_missing_returns_none() -> None:
    body_no_metadata = _body("claude-opus-4-8", ["Read"], "you are helpful", session_id=None)
    assert extract_session_id(body_no_metadata) is None

    body_no_session_id = _body("claude-opus-4-8", ["Read"], "you are helpful", session_id=None)
    body_no_session_id["metadata"] = {"user_id": json.dumps({"org_id": "org-1"})}
    assert extract_session_id(body_no_session_id) is None


def test_from_request_body_returns_lineage_id() -> None:
    body = _body("claude-opus-4-8", ["Read"], "you are helpful")
    assert isinstance(LineageId.from_request_body(body), LineageId)


def test_main_vs_subagent_differ() -> None:
    main_body = _body("claude-opus-4-8", [f"tool-{i}" for i in range(31)], "main system prompt")
    subagent_body = _body(
        "claude-sonnet-5", [f"stool-{i}" for i in range(25)], "subagent system prompt"
    )
    assert LineageId.from_request_body(main_body) != LineageId.from_request_body(subagent_body)


def test_main_lineage_stable_across_turns() -> None:
    tool_names = ["Read", "Write", "Bash"]
    turn_one = _body("claude-opus-4-8", tool_names, "you are helpful")
    turn_one["messages"] = [{"role": "user", "content": "first turn"}]
    turn_two = _body("claude-opus-4-8", tool_names, "you are helpful")
    turn_two["messages"] = [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "response"},
        {"role": "user", "content": "second turn"},
    ]
    assert LineageId.from_request_body(turn_one) == LineageId.from_request_body(turn_two)


def test_tool_order_independent() -> None:
    tool_names = ["Read", "Write", "Bash"]
    body_a = _body("claude-opus-4-8", tool_names, "you are helpful")
    body_b = _body("claude-opus-4-8", list(reversed(tool_names)), "you are helpful")
    assert LineageId.from_request_body(body_a) == LineageId.from_request_body(body_b)


def test_system_as_str_or_blocks() -> None:
    tool_names = ["Read", "Write"]
    body_str = _body("claude-opus-4-8", tool_names, "you are helpful")
    body_blocks = _body(
        "claude-opus-4-8",
        tool_names,
        [
            {"type": "text", "text": "you are helpful", "cache_control": {"type": "ephemeral"}},
        ],
    )
    assert LineageId.from_request_body(body_str) == LineageId.from_request_body(body_blocks)


_ATTRIBUTION = "x-anthropic-billing-header: cc_version=2.1.283.{}; cc_entrypoint=cli;"


def _system(fingerprint: str) -> list[dict[str, object]]:
    return [
        {"type": "text", "text": _ATTRIBUTION.format(fingerprint)},
        {"type": "text", "text": "you are helpful", "cache_control": {"type": "ephemeral"}},
    ]


def test_attribution_block_does_not_split_a_lineage() -> None:
    # Claude Code changes the block's fingerprint when a conversation is compacted.
    before = _body("claude-opus-4-8", ["Read"], _system("edd"))
    after = _body("claude-opus-4-8", ["Read"], _system("aba"))

    assert LineageId.from_request_body(before) == LineageId.from_request_body(after)


def test_system_text_beyond_attribution_still_splits_a_lineage() -> None:
    main = _body("claude-opus-4-8", ["Read"], _system("edd"))
    other = _body("claude-opus-4-8", ["Read"], _system("edd"))
    other["system"][1]["text"] = "you are a subagent"

    assert LineageId.from_request_body(main) != LineageId.from_request_body(other)


def _marked(ttl: str | None) -> dict[str, object]:
    cache_control: dict[str, str] = {"type": "ephemeral"}
    if ttl is not None:
        cache_control["ttl"] = ttl
    return {"type": "text", "text": "x", "cache_control": cache_control}


def test_cache_ttl_reads_one_hour_breakpoints() -> None:
    body = _body("claude-opus-4-8", ["Read"], [_marked("1h")])
    body["messages"] = [{"role": "user", "content": [_marked("1h")]}]

    assert cache_ttl_sec(body) == 3600


def test_cache_ttl_defaults_to_five_minutes() -> None:
    unmarked = _body("claude-opus-4-8", ["Read"], "you are helpful")
    default_ttl = _body("claude-opus-4-8", ["Read"], [_marked(None)])

    assert cache_ttl_sec(unmarked) == 300
    assert cache_ttl_sec(default_ttl) == 300


def test_cache_ttl_is_the_shortest_breakpoint() -> None:
    body = _body("claude-opus-4-8", ["Read"], [_marked("1h")])
    body["messages"] = [{"role": "user", "content": [_marked("5m")]}]

    assert cache_ttl_sec(body) == 300


def test_cache_ttl_ignores_cache_control_outside_the_prefix() -> None:
    body = _body("claude-opus-4-8", ["Read"], [_marked("1h")])
    body["metadata"] = {"cache_control": {"type": "ephemeral", "ttl": "5m"}}

    assert cache_ttl_sec(body) == 3600
