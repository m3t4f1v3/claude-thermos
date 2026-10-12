import pytest
from click.testing import CliRunner

from claude_thermos.cli import main
from claude_thermos.config import Config


def test_help_exits_zero() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0


def test_bad_max_cycles_reports_error() -> None:
    result = CliRunner().invoke(main, ["-n", "-1"])
    assert result.exit_code != 0
    assert 'max-cycles must be a non-negative integer or "auto"' in result.output


def test_flag_env_precedence_and_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def recorder(config: Config, claude_args: list[str], claude_bin: str = "claude") -> int:
        captured["config"] = config
        captured["claude_args"] = claude_args
        captured["claude_bin"] = claude_bin
        return 0

    monkeypatch.setattr("claude_thermos.cli.run_launcher", recorder)

    result = CliRunner().invoke(
        main,
        ["--idle", "400", "--interval", "999", "--", "chat", "-p", "hi"],
        env={"CLAUDE_THERMOS_IDLE_THRESHOLD_SEC": "300"},
    )

    assert result.exit_code == 0
    assert captured["config"].idle_threshold_sec == 400
    assert captured["config"].warm_interval_sec == 999
    assert captured["config"].subagent_active_window_sec == 540
    assert captured["claude_args"] == ["chat", "-p", "hi"]
    assert captured["claude_bin"] == "claude"

    result = CliRunner().invoke(main, [], env={"CLAUDE_THERMOS_IDLE_THRESHOLD_SEC": "300"})

    assert result.exit_code == 0
    assert captured["config"].idle_threshold_sec == 300


def test_passthrough_without_double_dash(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def recorder(config: Config, claude_args: list[str], claude_bin: str = "claude") -> int:
        captured["config"] = config
        captured["claude_args"] = claude_args
        return 0

    monkeypatch.setattr("claude_thermos.cli.run_launcher", recorder)

    result = CliRunner().invoke(main, ["chat", "-p", "hi"])

    assert result.exit_code == 0
    assert captured["claude_args"] == ["chat", "-p", "hi"]
    assert captured["config"].idle_threshold_sec == 270


def test_claude_bin_flag_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def recorder(config: Config, claude_args: list[str], claude_bin: str = "claude") -> int:
        captured["claude_bin"] = claude_bin
        captured["claude_args"] = claude_args
        return 0

    monkeypatch.setattr("claude_thermos.cli.run_launcher", recorder)

    result = CliRunner().invoke(main, ["--bin", "/path/to/claude-account1", "chat"])

    assert result.exit_code == 0
    assert captured["claude_bin"] == "/path/to/claude-account1"
    assert captured["claude_args"] == ["chat"]


def test_claude_bin_envvar(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def recorder(config: Config, claude_args: list[str], claude_bin: str = "claude") -> int:
        captured["claude_bin"] = claude_bin
        return 0

    monkeypatch.setattr("claude_thermos.cli.run_launcher", recorder)

    result = CliRunner().invoke(main, ["chat"], env={"CLAUDE_THERMOS_BIN": "/opt/claude"})

    assert result.exit_code == 0
    assert captured["claude_bin"] == "/opt/claude"


def test_serve_dispatches_to_run_server(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def recorder(config: Config, port: int, upstream: str) -> int:
        captured["config"] = config
        captured["port"] = port
        captured["upstream"] = upstream
        return 0

    monkeypatch.setattr("claude_thermos.cli.run_server", recorder)

    result = CliRunner().invoke(
        main, ["serve", "--port", "9000", "--idle", "120", "--session-ttl", "1800"]
    )

    assert result.exit_code == 0
    assert captured["port"] == 9000
    assert captured["upstream"] == "https://api.anthropic.com"
    assert captured["config"].idle_threshold_sec == 120
    assert captured["config"].session_ttl_sec == 1800


def test_serve_rejects_loopback_upstream() -> None:
    result = CliRunner().invoke(main, ["serve", "--upstream", "http://127.0.0.1:8787"])

    assert result.exit_code != 0
    assert "loopback" in result.output


@pytest.mark.parametrize("ttl", ["0", "-5"])
def test_serve_rejects_nonpositive_session_ttl(ttl: str) -> None:
    result = CliRunner().invoke(main, ["serve", "--session-ttl", ttl])

    assert result.exit_code != 0
    assert "session-ttl" in result.output.lower()


def test_serve_is_not_swallowed_by_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """The literal first token `serve` must select the daemon, not claude."""
    monkeypatch.setattr(
        "claude_thermos.cli.run_launcher",
        lambda config, args, claude_bin="claude": pytest.fail("serve routed to launcher"),
    )
    monkeypatch.setattr("claude_thermos.cli.run_server", lambda config, port, upstream: 0)

    result = CliRunner().invoke(main, ["serve"])

    assert result.exit_code == 0


def _recorded_config(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> Config:
    captured = {}

    def recorder(config: Config, claude_args: list[str], claude_bin: str = "claude") -> int:
        captured["config"] = config
        return 0

    monkeypatch.setattr("claude_thermos.cli.run_launcher", recorder)
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    return captured["config"]


def test_keepalive_and_compaction_are_on_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _recorded_config(monkeypatch, [])

    assert config.keepalive is True
    assert config.keepalive_interval_sec is None
    assert config.ping_budget == 0.10
    assert config.session_reserve == 0.05
    assert config.compact is True
    assert config.compact_min_tokens == 100_000
    assert config.pause_at == 0.95


def test_keepalive_options_reach_the_config(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _recorded_config(
        monkeypatch,
        [
            "--no-keepalive",
            "--keepalive-interval",
            "1800",
            "--ping-budget",
            "25",
            "--session-reserve",
            "5",
            "--no-compact",
            "--compact-min-tokens",
            "0",
            "--pause-at",
            "100",
        ],
    )

    assert config.keepalive is False
    assert config.keepalive_interval_sec == 1800
    assert config.ping_budget == 0.25
    assert config.session_reserve == 0.05
    assert config.compact is False
    assert config.compact_min_tokens == 0
    assert config.pause_at is None


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--ping-budget", "150"], "ping-budget must be a percentage between 0 and 100"),
        (["--session-reserve", "-1"], "session-reserve must be a percentage between 0 and 100"),
        (["--keepalive-interval", "never"], "keepalive-interval must be a positive integer"),
    ],
)
def test_bad_keepalive_options_report_errors(args: list[str], message: str) -> None:
    result = CliRunner().invoke(main, args)

    assert result.exit_code != 0
    assert message in result.output


def test_serve_takes_keepalive_options_but_never_compacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def recorder(config: Config, port: int, upstream: str) -> int:
        captured["config"] = config
        return 0

    monkeypatch.setattr("claude_thermos.cli.run_server", recorder)

    result = CliRunner().invoke(main, ["serve", "--ping-budget", "20"])

    assert result.exit_code == 0, result.output
    assert captured["config"].keepalive is True
    assert captured["config"].ping_budget == 0.20
    assert captured["config"].compact is False


def test_serve_does_not_accept_compact() -> None:
    result = CliRunner().invoke(main, ["serve", "--compact"])

    assert result.exit_code != 0
