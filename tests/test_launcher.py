import json
import socket
import time
from pathlib import Path
from typing import Never

import pytest

from claude_thermos.config import Config
from claude_thermos.launcher import (
    ProxyHandle,
    RealProxyHandle,
    child_env,
    run_launcher,
    with_hook_settings,
)
from claude_thermos.proxy import WarmerAddon, find_free_port, hook_settings


def test_child_env_sets_base_url() -> None:
    base_env = {"PATH": "/usr/bin"}

    env = child_env(base_env, 8123)

    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8123"
    assert base_env == {"PATH": "/usr/bin"}


class _FakeProxyHandle(ProxyHandle):
    def __init__(self, port: int) -> None:
        self.port = port
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


def test_run_launcher_injects_env_and_tears_down() -> None:
    captured = {}
    handles = []

    def fake_proxy_factory(port: int, addon: object | None) -> _FakeProxyHandle:
        handle = _FakeProxyHandle(port)
        handles.append(handle)
        return handle

    def fake_spawn(claude_bin: str, args: list[str], env: dict[str, str]) -> int:
        captured["claude_bin"] = claude_bin
        captured["args"] = args
        captured["env"] = env
        return 7

    config = Config()
    exit_code = run_launcher(
        config,
        ["chat", "-p", "hi"],
        claude_bin="/custom/claude",
        proxy_factory=fake_proxy_factory,
        spawn=fake_spawn,
    )

    assert exit_code == 7
    assert captured["claude_bin"] == "/custom/claude"
    assert captured["args"] == ["chat", "-p", "hi"]
    handle = handles[0]
    assert captured["env"]["ANTHROPIC_BASE_URL"] == f"http://127.0.0.1:{handle.port}"
    assert handle.start_calls == 1
    assert handle.stop_calls == 1


def test_run_launcher_stops_proxy_on_child_error() -> None:
    handle = _FakeProxyHandle(8123)

    def fake_proxy_factory(port: int, addon: object | None) -> _FakeProxyHandle:
        return handle

    def fake_spawn(claude_bin: str, args: list[str], env: dict[str, str]) -> Never:
        raise RuntimeError("boom")

    config = Config()

    with pytest.raises(RuntimeError, match="boom"):
        run_launcher(
            config,
            [],
            proxy_factory=fake_proxy_factory,
            spawn=fake_spawn,
        )

    assert handle.stop_calls == 1


def _can_connect(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        try:
            sock.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def test_real_proxy_handle_starts_and_stops() -> None:
    port = find_free_port()
    handle = RealProxyHandle(port)

    handle.start()

    deadline = time.monotonic() + 5
    connected = False
    while time.monotonic() < deadline:
        if _can_connect(port):
            connected = True
            break
        time.sleep(0.05)
    assert connected, "proxy never started accepting connections"

    handle.stop()

    deadline = time.monotonic() + 5
    stopped = False
    while time.monotonic() < deadline:
        if not _can_connect(port):
            stopped = True
            break
        time.sleep(0.05)
    assert stopped, "proxy still accepting connections after stop()"


def test_child_env_restores_direct_connection_features() -> None:
    env = child_env({"PATH": "/usr/bin"}, 8123)

    assert env["ENABLE_TOOL_SEARCH"] == "true"
    assert env["CLAUDE_CODE_ENABLE_FINE_GRAINED_TOOL_STREAMING"] == "1"


def test_child_env_keeps_the_callers_own_choices() -> None:
    env = child_env({"ENABLE_TOOL_SEARCH": "false"}, 8123)

    assert env["ENABLE_TOOL_SEARCH"] == "false"


def _settings_arg(args: list[str]) -> dict:
    assert args.count("--settings") == 1
    return json.loads(args[args.index("--settings") + 1])


def test_with_hook_settings_prepends_the_proxy_hooks() -> None:
    args = with_hook_settings(["--model", "opus", "fix the bug"], 8123)

    assert args[0] == "--settings"
    assert args[2:] == ["--model", "opus", "fix the bug"]
    assert _settings_arg(args) == hook_settings(8123)


def test_with_hook_settings_merges_inline_settings() -> None:
    mine = {
        "theme": "dark",
        "hooks": {
            "PostCompact": [{"hooks": [{"type": "command", "command": "notify-send compacted"}]}],
            "Stop": [{"hooks": [{"type": "command", "command": "true"}]}],
        },
    }

    args = with_hook_settings(["--settings", json.dumps(mine), "-c"], 8123)

    merged = _settings_arg(args)
    ours = hook_settings(8123)["hooks"]
    assert args[2:] == ["-c"]
    assert merged["theme"] == "dark"
    assert merged["hooks"]["Stop"] == mine["hooks"]["Stop"]
    assert merged["hooks"]["PostCompact"] == mine["hooks"]["PostCompact"] + ours["PostCompact"]
    assert merged["hooks"]["UserPromptSubmit"] == ours["UserPromptSubmit"]


def test_with_hook_settings_merges_a_settings_file(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({"theme": "dark"}))

    args = with_hook_settings([f"--settings={settings_file}", "-c"], 8123)

    merged = _settings_arg(args)
    assert args[2:] == ["-c"]
    assert merged == {"theme": "dark", **hook_settings(8123)}


def test_with_hook_settings_leaves_unreadable_settings_to_claude() -> None:
    args = ["--settings", "{not json", "-c"]

    assert with_hook_settings(args, 8123) == args


def _launch_recording_args(
    config: Config, claude_args: list[str], interactive: bool | None
) -> list[str]:
    captured: dict[str, list[str]] = {}

    def fake_spawn(claude_bin: str, args: list[str], env: dict[str, str]) -> int:
        captured["args"] = args
        return 0

    run_launcher(
        config,
        claude_args,
        proxy_factory=lambda port, addon: _FakeProxyHandle(port),
        spawn=fake_spawn,
        interactive=interactive,
    )
    return captured["args"]


def test_run_launcher_adds_hooks_to_an_interactive_keepalive_session() -> None:
    args = _launch_recording_args(Config(keepalive=True), ["-c"], interactive=True)

    assert args[0] == "--settings"
    assert args[2:] == ["-c"]
    assert set(_settings_arg(args)["hooks"]) == {"UserPromptSubmit", "SessionEnd", "PostCompact"}


@pytest.mark.parametrize(
    ("config", "interactive"),
    [
        (Config(keepalive=False), True),
        (Config(keepalive=True, disabled=True), True),
        (Config(keepalive=True), False),
    ],
)
def test_run_launcher_passes_args_through_otherwise(config: Config, interactive: bool) -> None:
    assert _launch_recording_args(config, ["-c"], interactive=interactive) == ["-c"]


def test_run_launcher_treats_print_mode_as_non_interactive() -> None:
    args = _launch_recording_args(Config(keepalive=True), ["-p", "hi"], interactive=None)

    assert args == ["-p", "hi"]


def test_run_launcher_gives_the_warmer_the_login_claude_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    stored = {"claudeAiOauth": {"accessToken": "sk-ant-oat01-stored"}}
    (tmp_path / ".credentials.json").write_text(json.dumps(stored))
    addons: list[object | None] = []

    def fake_proxy_factory(port: int, addon: object | None) -> _FakeProxyHandle:
        addons.append(addon)
        return _FakeProxyHandle(port)

    run_launcher(
        Config(), [], proxy_factory=fake_proxy_factory, spawn=lambda *_: 0, interactive=False
    )

    addon = addons[0]
    assert isinstance(addon, WarmerAddon)
    assert addon._warmer is not None
    assert addon._warmer._login is not None
    assert addon._warmer._login.read() == "sk-ant-oat01-stored"
