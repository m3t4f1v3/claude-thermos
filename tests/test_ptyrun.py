import fcntl
import os
import pty
import select
import struct
import termios
import threading
import time
from collections.abc import Callable, Iterator

import pytest

from claude_thermos.ptyrun import (
    SKIP_INPUT_NOT_EMPTY,
    SKIP_NO_TERMINAL,
    InputTracker,
    PtyRunner,
)


def _dirty_after(*chunks: bytes) -> bool:
    tracker = InputTracker()
    for chunk in chunks:
        tracker.feed(chunk)
    return tracker.dirty


def test_typed_text_is_a_draft() -> None:
    assert _dirty_after(b"fix the bug") is True
    assert _dirty_after("héllo".encode()) is True


def test_enter_submits_the_draft() -> None:
    assert _dirty_after(b"fix the bug\r") is False


def test_ctrl_c_clears_the_draft() -> None:
    assert _dirty_after(b"fix the bug\x03") is False


def test_backspacing_still_counts_as_a_draft() -> None:
    assert _dirty_after(b"a\x7f") is True


@pytest.mark.parametrize(
    "sequence",
    [
        b"\x1b[A",  # cursor up
        b"\x1bOP",  # F1
        b"\x1b[<0;10;5M\x1b[<0;10;5m",  # mouse press and release
        b"\x1b[I\x1b[O",  # focus in, focus out
        b"\x1b[?62;4c",  # device attributes reply
        b"\x1b]11;rgb:0000/0000/0000\x07",  # color query reply, BEL-terminated
        b"\x1b]11;rgb:0000/0000/0000\x1b\\",  # the same, ST-terminated
        b"\x1bf",  # Alt+f
        b"\x1b[57441;1u",  # kitty report of a bare modifier key
        b"\x1b[97;5u",  # kitty Ctrl+a
    ],
)
def test_terminal_chatter_is_not_a_draft(sequence: bytes) -> None:
    assert _dirty_after(sequence) is False


def test_escape_sequence_split_across_reads_is_not_a_draft() -> None:
    assert _dirty_after(b"\x1b", b"[<0;1", b"0;5M") is False


def test_bracketed_paste_is_a_draft() -> None:
    assert _dirty_after(b"\x1b[200~") is True


def test_kitty_key_reports_are_tracked_like_plain_keys() -> None:
    assert _dirty_after(b"\x1b[97u") is True  # a
    assert _dirty_after(b"\x1b[65;2u") is True  # Shift+a
    assert _dirty_after(b"\x1b[97u", b"\x1b[13u") is False  # Enter submits
    assert _dirty_after(b"\x1b[97u", b"\x1b[13;2u") is True  # Shift+Enter is a newline
    assert _dirty_after(b"\x1b[97u", b"\x1b[99;5u") is False  # Ctrl+c clears
    assert _dirty_after(b"\x1b[97u", b"\x1b[13;1:3u") is True  # Enter released, not pressed


class _Terminal:
    """The far side of a pseudo-terminal standing in for the user's own: what
    the test writes is what the user types, what it reads is what they see."""

    def __init__(self, rows: int = 24, cols: int = 80) -> None:
        self.master_fd, self.slave_fd = pty.openpty()
        fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        self.screen = b""

    def type(self, data: bytes) -> None:
        os.write(self.master_fd, data)

    def wait_for(self, needle: bytes, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while needle not in self.screen and time.monotonic() < deadline:
            readable, _, _ = select.select([self.master_fd], [], [], 0.05)
            if readable:
                self.screen += os.read(self.master_fd, 65536)
        return needle in self.screen

    def close(self) -> None:
        os.close(self.master_fd)
        os.close(self.slave_fd)


@pytest.fixture
def terminal() -> Iterator[_Terminal]:
    term = _Terminal(rows=31, cols=97)
    yield term
    term.close()


def _spawn_with(
    terminal: _Terminal, script: str, user: Callable[[PtyRunner], None]
) -> tuple[int, list[BaseException]]:
    """Run `script` under a PtyRunner on the main thread while `user` plays
    the person at the terminal on another."""
    runner = PtyRunner(stdin_fd=terminal.slave_fd, stdout_fd=terminal.slave_fd)
    errors: list[BaseException] = []

    def play() -> None:
        try:
            user(runner)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=play, daemon=True)
    thread.start()
    exit_code = runner.spawn("sh", ["-c", script], dict(os.environ))
    thread.join(timeout=10)
    return exit_code, errors


def test_request_compact_before_spawn_has_no_terminal() -> None:
    assert PtyRunner().request_compact("sess-1") == SKIP_NO_TERMINAL


def test_spawn_relays_the_terminal_and_returns_the_exit_code(terminal: _Terminal) -> None:
    def user(runner: PtyRunner) -> None:
        assert terminal.wait_for(b"ready")
        terminal.type(b"hello\r")
        assert terminal.wait_for(b"got:hello")

    script = 'echo ready; read line; echo "got:$line"; exit 3'
    exit_code, errors = _spawn_with(terminal, script, user)

    assert errors == []
    assert exit_code == 3


def test_spawn_gives_the_child_the_window_size(terminal: _Terminal) -> None:
    def user(runner: PtyRunner) -> None:
        assert terminal.wait_for(b"size:31 97")

    exit_code, errors = _spawn_with(terminal, 'echo "size:$(stty size)"', user)

    assert errors == []
    assert exit_code == 0


def test_request_compact_types_the_command_then_submits_it(terminal: _Terminal) -> None:
    def user(runner: PtyRunner) -> None:
        assert terminal.wait_for(b"ready")
        assert runner.request_compact("sess-1") is None
        assert terminal.wait_for(b"got:/compact")

    script = 'echo ready; read line; echo "got:$line"'
    exit_code, errors = _spawn_with(terminal, script, user)

    assert errors == []
    assert exit_code == 0


def test_request_compact_leaves_a_draft_alone(terminal: _Terminal) -> None:
    def user(runner: PtyRunner) -> None:
        assert terminal.wait_for(b"ready")
        terminal.type(b"half a thought")
        assert terminal.wait_for(b"half a thought")
        assert runner.request_compact("sess-1") == SKIP_INPUT_NOT_EMPTY
        terminal.type(b"\r")
        assert terminal.wait_for(b"got:half a thought")
        # Submitted, so the prompt is empty again.
        assert runner.request_compact("sess-1") is None
        assert terminal.wait_for(b"got:/compact")

    script = 'echo ready; read a; echo "got:$a"; read b; echo "got:$b"'
    exit_code, errors = _spawn_with(terminal, script, user)

    assert errors == []
    assert exit_code == 0


def test_spawn_reports_a_missing_executable(terminal: _Terminal) -> None:
    runner = PtyRunner(stdin_fd=terminal.slave_fd, stdout_fd=terminal.slave_fd)

    assert runner.spawn("claude-thermos-no-such-binary", [], dict(os.environ)) == 127
