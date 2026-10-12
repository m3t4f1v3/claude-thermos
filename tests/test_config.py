import pytest

from claude_thermos.config import Config, build_config, is_loopback_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8787",
        "http://localhost:8787",
        "http://[::1]:8787",
        "http://127.0.0.2:8787",
        "http://127.255.255.254",
    ],
)
def test_is_loopback_url_true(url: str) -> None:
    assert is_loopback_url(url) is True


@pytest.mark.parametrize(
    "url",
    ["https://api.anthropic.com", "https://example.test:8787"],
)
def test_is_loopback_url_false(url: str) -> None:
    assert is_loopback_url(url) is False


def test_build_config_defaults() -> None:
    config = build_config(270, 270, "2", 540, {})
    assert config == Config(
        idle_threshold_sec=270,
        warm_interval_sec=270,
        warm_max_cycles=2,
        subagent_active_window_sec=540,
        disabled=False,
    )


def test_build_config_auto_max_cycles() -> None:
    config = build_config(270, 270, "auto", 540, {})
    assert config.warm_max_cycles is None


@pytest.mark.parametrize("max_cycles_raw", ["-1", "x"])
def test_build_config_bad_max_cycles_raises(max_cycles_raw: str) -> None:
    with pytest.raises(ValueError, match=r'max-cycles must be a non-negative integer or "auto"'):
        build_config(270, 270, max_cycles_raw, 540, {})


def test_build_config_disabled_env() -> None:
    assert build_config(270, 270, "2", 540, {"CLAUDE_THERMOS_DISABLE": "1"}).disabled is True
    assert build_config(270, 270, "2", 540, {"CLAUDE_THERMOS_DISABLE": "0"}).disabled is False
    assert build_config(270, 270, "2", 540, {}).disabled is False


def test_warm_interval_for_five_minute_prefix_is_the_warm_interval() -> None:
    config = Config(warm_interval_sec=240, keepalive_interval_sec=1000)

    assert config.warm_interval_for(300) == 240


def test_warm_interval_for_one_hour_prefix_leaves_a_margin() -> None:
    assert Config().warm_interval_for(3600) == 3300


def test_warm_interval_for_one_hour_prefix_honors_an_explicit_interval() -> None:
    assert Config(keepalive_interval_sec=1800).warm_interval_for(3600) == 1800


def test_build_config_parses_keepalive_options() -> None:
    config = build_config(
        270,
        270,
        "4",
        540,
        {},
        keepalive=True,
        keepalive_interval_raw="1800",
        ping_budget_pct=15,
        session_reserve_pct=5,
        compact=True,
        compact_min_tokens=50_000,
    )

    assert config.keepalive is True
    assert config.keepalive_interval_sec == 1800
    assert config.ping_budget == 0.15
    assert config.session_reserve == 0.05
    assert config.compact is True
    assert config.compact_min_tokens == 50_000


def test_build_config_keepalive_interval_auto_is_none() -> None:
    assert build_config(270, 270, "4", 540, {}).keepalive_interval_sec is None


@pytest.mark.parametrize("raw", ["0", "-5", "soon"])
def test_build_config_rejects_bad_keepalive_interval(raw: str) -> None:
    with pytest.raises(ValueError, match="keepalive-interval"):
        build_config(270, 270, "4", 540, {}, keepalive_interval_raw=raw)


@pytest.mark.parametrize("pct", [-1, 100.5])
def test_build_config_rejects_out_of_range_percentages(pct: float) -> None:
    with pytest.raises(ValueError, match="ping-budget must be a percentage"):
        build_config(270, 270, "4", 540, {}, ping_budget_pct=pct)
    with pytest.raises(ValueError, match="session-reserve must be a percentage"):
        build_config(270, 270, "4", 540, {}, session_reserve_pct=pct)
    with pytest.raises(ValueError, match="pause-at must be a percentage"):
        build_config(270, 270, "4", 540, {}, pause_at_pct=pct)


def test_build_config_pause_at_100_never_pauses() -> None:
    assert build_config(270, 270, "4", 540, {}, pause_at_pct=95).pause_at == 0.95
    assert build_config(270, 270, "4", 540, {}, pause_at_pct=100).pause_at is None
