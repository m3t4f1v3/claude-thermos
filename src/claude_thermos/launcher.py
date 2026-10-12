import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path

from mitmproxy.tools.dump import DumpMaster

from claude_thermos.config import ANTHROPIC_BASE_URL, Config
from claude_thermos.credentials import stored_login
from claude_thermos.proxy import WarmerAddon, build_master, find_free_port, hook_settings
from claude_thermos.ptyrun import PtyRunner

_SETTINGS_FLAG = "--settings"
_PROXY_READY_TIMEOUT_SEC = 5.0
_PRINT_FLAGS = frozenset({"-p", "--print"})

# Claude Code turns these off when ANTHROPIC_BASE_URL is not api.anthropic.com,
# in case a gateway mangles them. This proxy forwards requests untouched, so
# they are switched back on to match a direct connection. Without tool search
# every MCP tool definition loads into the prompt up front.
_DIRECT_CONNECTION_DEFAULTS = {
    "ENABLE_TOOL_SEARCH": "true",
    "CLAUDE_CODE_ENABLE_FINE_GRAINED_TOOL_STREAMING": "1",
}


class ProxyHandle(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...


class RealProxyHandle(ProxyHandle):
    """ProxyHandle backed by mitmproxy's DumpMaster.

    `build_master` binds `master.event_loop` to whatever asyncio loop is
    current at construction time, creating a throwaway loop via
    `asyncio.new_event_loop()` when none is running. `master.shutdown()`
    calls `master.event_loop.call_soon_threadsafe(...)` internally, so the
    master MUST be driven on that same loop or `shutdown()` will target a
    dead loop and hang forever. `start()` therefore runs `master.run()` on
    a background thread via `master.event_loop.run_until_complete(...)`
    after making that loop current on the thread, and closes the loop once
    the run completes to avoid a file descriptor leak.

    `master.run()` returning does not by itself close the listening
    sockets (mitmproxy leaves that to process exit in its own CLI), so the
    background thread also explicitly tears down the proxyserver addon's
    server instances before closing the loop.
    """

    def __init__(
        self, port: int, addon: object | None = None, upstream: str = ANTHROPIC_BASE_URL
    ) -> None:
        self._port = port
        self._addon = addon
        self._upstream = upstream
        self._master: DumpMaster | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        master = build_master(self._port, self._addon, self._upstream)
        self._master = master
        loop = master.event_loop

        def _run() -> None:
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(master.run())
                proxyserver = master.addons.get("proxyserver")
                if proxyserver is not None:
                    loop.run_until_complete(proxyserver.servers.update([]))
            finally:
                loop.close()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._master is None:
            return
        self._master.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=10)


def child_env(base_env: Mapping[str, str], port: int) -> dict[str, str]:
    """Return a copy of base_env pointing Claude at the local proxy.

    Args:
        base_env: The environment to copy.
        port: Loopback port the proxy listens on.

    Returns:
        A copy of base_env with ANTHROPIC_BASE_URL set to
        http://127.0.0.1:<port>, and the features Claude Code reserves for
        a direct connection enabled unless base_env already sets them.
    """
    env = dict(base_env)
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
    for name, value in _DIRECT_CONNECTION_DEFAULTS.items():
        env.setdefault(name, value)
    return env


def _load_settings(value: str) -> dict | None:
    """Read a `--settings` value: a path to a JSON file, or inline JSON.

    Args:
        value: The flag's value.

    Returns:
        The settings object, or None if the value is neither.
    """
    try:
        text = Path(value).read_text() if Path(value).is_file() else value
        settings = json.loads(text)
    except (OSError, ValueError):
        return None
    return settings if isinstance(settings, dict) else None


def with_hook_settings(claude_args: list[str], port: int) -> list[str]:
    """Return claude_args with the proxy's hooks added through `--settings`.

    Claude Code merges hooks from `--settings` with those in the user's
    settings files. If the caller already passes `--settings`, its hooks are
    merged with the proxy's and the flag is rewritten inline, since claude
    takes the flag once. A value that cannot be read is left for claude to
    report, and the proxy's hooks are left out.

    Args:
        claude_args: Arguments the caller is passing to `claude`.
        port: Loopback port the proxy listens on.

    Returns:
        The arguments to launch `claude` with.
    """
    ours = hook_settings(port)
    args = list(claude_args)
    for index, arg in enumerate(args):
        if arg == _SETTINGS_FLAG and index + 1 < len(args):
            span, value = slice(index, index + 2), args[index + 1]
        elif arg.startswith(_SETTINGS_FLAG + "="):
            span, value = slice(index, index + 1), arg.partition("=")[2]
        else:
            continue
        settings = _load_settings(value)
        if settings is None or not isinstance(settings.get("hooks", {}), dict):
            return args
        hooks = dict(settings.get("hooks", {}))
        for event, handlers in ours["hooks"].items():
            hooks[event] = [*hooks.get(event, []), *handlers]
        args[span] = [_SETTINGS_FLAG, json.dumps({**settings, "hooks": hooks})]
        return args
    return [_SETTINGS_FLAG, json.dumps(ours), *args]


def _wait_until_listening(port: int, timeout_sec: float = _PROXY_READY_TIMEOUT_SEC) -> None:
    """Block until the loopback port accepts connections, or the timeout
    passes. The proxy binds on its own thread shortly after `start`.

    Args:
        port: Loopback port the proxy listens on.
        timeout_sec: Longest to wait, in seconds.
    """
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.02)


def _is_interactive(claude_args: list[str]) -> bool:
    """Report whether `claude` will run its interactive UI on this terminal.

    Args:
        claude_args: Arguments being passed to `claude`.

    Returns:
        True iff stdin and stdout are terminals and print mode is not asked for.
    """
    if _PRINT_FLAGS.intersection(claude_args):
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


def _default_proxy_factory(port: int, addon: object | None) -> ProxyHandle:
    return RealProxyHandle(port, addon)


def _default_spawn(claude_bin: str, args: list[str], env: dict[str, str]) -> int:
    completed = subprocess.run([claude_bin, *args], env=env)
    return completed.returncode


def run_launcher(
    config: Config,
    claude_args: list[str],
    claude_bin: str = "claude",
    proxy_factory: Callable[[int, object | None], ProxyHandle] = _default_proxy_factory,
    spawn: Callable[[str, list[str], dict[str, str]], int] | None = None,
    interactive: bool | None = None,
) -> int:
    """Run Claude Code behind the warming proxy.

    Starts the proxy (unless config.disabled), spawns `claude` with
    child_env + claude_args, waits for it, then stops the proxy.

    An interactive session with keepalive on also gets the proxy's hooks,
    which tell the warmer which session the terminal has open. If it may be
    compacted as well, `claude` runs on a pseudo-terminal the launcher owns,
    so the warmer can type `/compact` into it; otherwise it inherits stdio.
    The pseudo-terminal's child is forked before the proxy starts its
    threads and released once the proxy is listening.

    The warmer is also given the login Claude Code stores under this
    environment, so a warm can pick up the OAuth token that replaced one
    the API has revoked while the session sat idle.

    Args:
        config: Resolved launcher configuration.
        claude_args: Extra arguments passed through to `claude`.
        claude_bin: Name or path of the `claude` executable to launch; looked
            up on PATH when it is a bare name.
        proxy_factory: Factory building a ProxyHandle from port and addon;
            injectable for testing.
        spawn: Callable that runs the claude binary and returns its exit
            code; injectable for testing. When None, the child runs on
            a pseudo-terminal if the session may be compacted, else with
            inherited stdio.
        interactive: Whether `claude` runs its interactive UI on this
            terminal; detected from the arguments and stdio when None.

    Returns:
        The child process's exit code.
    """
    port = find_free_port()
    if interactive is None:
        interactive = _is_interactive(claude_args)
    keepalive = interactive and config.keepalive and not config.disabled

    env = child_env(os.environ, port)
    args = with_hook_settings(claude_args, port) if keepalive else claude_args

    runner = None
    if spawn is None and keepalive and config.compact:
        runner = PtyRunner()
        runner.start(claude_bin, args, env)

    addon = None
    if not config.disabled:
        addon = WarmerAddon(
            config=config,
            compactor=runner.request_compact if runner is not None else None,
            single_session=True,
            login=stored_login(env),
        )

    proxy = proxy_factory(port, addon)
    proxy.start()
    try:
        if runner is not None:
            _wait_until_listening(port)
            return runner.wait()
        return (spawn or _default_spawn)(claude_bin, args, env)
    finally:
        proxy.stop()
