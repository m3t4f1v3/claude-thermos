import os
import sys
from collections.abc import Callable
from dataclasses import replace
from importlib.metadata import version

import click

from claude_thermos.config import DEFAULT_UPSTREAM, Config, build_config, is_loopback_url
from claude_thermos.launcher import run_launcher
from claude_thermos.server import run_server

__all__ = ["main"]


def _warmer_options(func: Callable) -> Callable:
    """Apply the shared warmer-tuning options used by both entry points."""
    func = click.option(
        "--idle",
        "idle_threshold_sec",
        type=int,
        default=270,
        show_default=True,
        envvar="CLAUDE_THERMOS_IDLE_THRESHOLD_SEC",
        help="Idle threshold, in seconds, before a warming cycle runs.",
    )(func)
    func = click.option(
        "--interval",
        "warm_interval_sec",
        type=int,
        default=270,
        show_default=True,
        envvar="CLAUDE_THERMOS_WARM_INTERVAL_SEC",
        help="Interval, in seconds, between warming cycles.",
    )(func)
    func = click.option(
        "-n",
        "--max-cycles",
        "max_cycles_raw",
        default="4",
        show_default=True,
        envvar="CLAUDE_THERMOS_WARM_MAX_CYCLES",
        help='Maximum number of warming cycles, or "auto" for unlimited.',
    )(func)
    func = click.option(
        "--subagent-window",
        "subagent_active_window_sec",
        type=int,
        default=540,
        show_default=True,
        envvar="CLAUDE_THERMOS_SUBAGENT_ACTIVE_WINDOW_SEC",
        help="Subagent active window, in seconds.",
    )(func)
    func = click.option(
        "--keepalive/--no-keepalive",
        default=True,
        show_default=True,
        envvar="CLAUDE_THERMOS_KEEPALIVE",
        help="Keep warming after the turn ends, while the session waits on you.",
    )(func)
    func = click.option(
        "--keepalive-interval",
        "keepalive_interval_raw",
        default="auto",
        show_default=True,
        envvar="CLAUDE_THERMOS_KEEPALIVE_INTERVAL_SEC",
        help='Seconds between warms of a one-hour cache, or "auto" for 55 minutes.',
    )(func)
    func = click.option(
        "--ping-budget",
        "ping_budget_pct",
        type=float,
        default=10.0,
        show_default=True,
        envvar="CLAUDE_THERMOS_PING_BUDGET",
        help="Percent of the subscription session limit one idle spell may spend on warms.",
    )(func)
    func = click.option(
        "--session-reserve",
        "session_reserve_pct",
        type=float,
        default=5.0,
        show_default=True,
        envvar="CLAUDE_THERMOS_SESSION_RESERVE",
        help="Percent of each subscription limit that warms must leave unused.",
    )(func)
    func = click.option(
        "--pause-at",
        "pause_at_pct",
        type=float,
        default=95.0,
        show_default=True,
        envvar="CLAUDE_THERMOS_PAUSE_AT",
        help=(
            "Percent of the session limit at which to pause the conversation until the limit "
            "resets, keeping the rest for warms; 100 never pauses."
        ),
    )(func)
    return func


def _build(
    idle_threshold_sec: int,
    warm_interval_sec: int,
    max_cycles_raw: str,
    subagent_active_window_sec: int,
    keepalive: bool,
    keepalive_interval_raw: str,
    ping_budget_pct: float,
    session_reserve_pct: float,
    pause_at_pct: float,
    compact: bool = False,
    compact_min_tokens: int = 100_000,
) -> Config:
    try:
        return build_config(
            idle_threshold_sec,
            warm_interval_sec,
            max_cycles_raw,
            subagent_active_window_sec,
            os.environ,
            keepalive=keepalive,
            keepalive_interval_raw=keepalive_interval_raw,
            ping_budget_pct=ping_budget_pct,
            session_reserve_pct=session_reserve_pct,
            compact=compact,
            compact_min_tokens=compact_min_tokens,
            pause_at_pct=pause_at_pct,
        )
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


class _ThermosCLI(click.Group):
    """Group that forwards to the `launch` command for anything that is not
    the `serve` subcommand (or a top-level --version/--help), so
    `claude-thermos <claude args>` keeps forwarding straight through to claude
    while `claude-thermos serve` runs the daemon.

    It injects the `launch` command name rather than dispatching by hand, so
    option and environment-variable parsing still happen the normal way.
    """

    _PASSTHROUGH_HEAD = ("--version", "-V", "--help", "-h")

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and (args[0] in self.commands or args[0] in self._PASSTHROUGH_HEAD):
            return super().parse_args(ctx, args)
        return super().parse_args(ctx, ["launch", *args])


@click.group(
    cls=_ThermosCLI,
    context_settings={"ignore_unknown_options": True, "help_option_names": ["-h", "--help"]},
)
@click.version_option(version("claude-thermos"), "-V", "--version")
def main() -> None:
    """Launch Claude Code behind a local cache-warming reverse proxy.

    Runs claude as usual by default; `claude-thermos serve` instead runs the
    warmer as a standalone daemon that any client can share.
    """


@main.command(
    context_settings={"ignore_unknown_options": True, "help_option_names": ["-h", "--help"]}
)
@_warmer_options
@click.option(
    "--bin",
    "claude_bin",
    default="claude",
    show_default=True,
    envvar="CLAUDE_THERMOS_BIN",
    help="Name or path of the `claude` executable to launch; a bare name is looked up on PATH.",
)
@click.option(
    "--compact/--no-compact",
    default=True,
    show_default=True,
    envvar="CLAUDE_THERMOS_COMPACT",
    help="Run /compact when an ended turn can no longer be kept warm.",
)
@click.option(
    "--compact-min-tokens",
    type=click.IntRange(min=0),
    default=100_000,
    show_default=True,
    envvar="CLAUDE_THERMOS_COMPACT_MIN_TOKENS",
    help="Smallest conversation, in tokens, worth compacting.",
)
@click.argument("claude_args", nargs=-1, type=click.UNPROCESSED)
def launch(
    idle_threshold_sec: int,
    warm_interval_sec: int,
    max_cycles_raw: str,
    subagent_active_window_sec: int,
    keepalive: bool,
    keepalive_interval_raw: str,
    ping_budget_pct: float,
    session_reserve_pct: float,
    pause_at_pct: float,
    claude_bin: str,
    compact: bool,
    compact_min_tokens: int,
    claude_args: tuple[str, ...],
) -> None:
    """Launch Claude Code behind the warming proxy (the default action)."""
    config = _build(
        idle_threshold_sec,
        warm_interval_sec,
        max_cycles_raw,
        subagent_active_window_sec,
        keepalive,
        keepalive_interval_raw,
        ping_budget_pct,
        session_reserve_pct,
        pause_at_pct,
        compact,
        compact_min_tokens,
    )
    sys.exit(run_launcher(config, list(claude_args), claude_bin=claude_bin))


@main.command(context_settings={"help_option_names": ["-h", "--help"]})
@_warmer_options
@click.option(
    "--port",
    type=int,
    default=8787,
    show_default=True,
    envvar="CLAUDE_THERMOS_PORT",
    help="Loopback port the daemon listens on.",
)
@click.option(
    "--upstream",
    default=DEFAULT_UPSTREAM,
    show_default=True,
    envvar="CLAUDE_THERMOS_UPSTREAM",
    help="Real upstream Anthropic API URL to reverse-proxy to.",
)
@click.option(
    "--session-ttl",
    "session_ttl_sec",
    type=click.IntRange(min=1),
    default=3600,
    show_default=True,
    envvar="CLAUDE_THERMOS_SESSION_TTL_SEC",
    help="Seconds a session may sit idle before the daemon evicts it.",
)
def serve(
    idle_threshold_sec: int,
    warm_interval_sec: int,
    max_cycles_raw: str,
    subagent_active_window_sec: int,
    keepalive: bool,
    keepalive_interval_raw: str,
    ping_budget_pct: float,
    session_reserve_pct: float,
    pause_at_pct: float,
    port: int,
    upstream: str,
    session_ttl_sec: int,
) -> None:
    """Run the cache-warming proxy as a standalone daemon.

    Point any client at it with ANTHROPIC_BASE_URL=http://127.0.0.1:<port>.
    Unlike the default launcher, the daemon is not the parent of claude, so
    it can warm the IDE extension and several terminals from one process.
    """
    if is_loopback_url(upstream):
        raise click.BadParameter(
            f"must be the real Anthropic API, not a loopback URL: {upstream}",
            param_hint="--upstream",
        )
    config = replace(
        _build(
            idle_threshold_sec,
            warm_interval_sec,
            max_cycles_raw,
            subagent_active_window_sec,
            keepalive,
            keepalive_interval_raw,
            ping_budget_pct,
            session_reserve_pct,
            pause_at_pct,
        ),
        session_ttl_sec=session_ttl_sec,
    )
    sys.exit(run_server(config, port, upstream))
