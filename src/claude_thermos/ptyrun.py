import errno
import fcntl
import os
import pty
import select
import shutil
import signal
import termios
import threading
import time
import tty
from types import FrameType
from typing import NoReturn

# What `request_compact` types into the prompt. The command and the Enter that
# submits it are written separately: Claude Code reads a burst that contains
# both as pasted text and would insert the Enter as a newline.
_COMPACT_COMMAND = b"/compact"
_SUBMIT = b"\r"
_SUBMIT_DELAY_SEC = 0.5

# Why a compaction could not be typed. The value is the `reason` field of the
# compact_skipped event.
SKIP_NO_TERMINAL = "no_terminal"
SKIP_INPUT_NOT_EMPTY = "input_not_empty"

_READ_SIZE = 65536
_ESC = 0x1B
_BEL = 0x07
_ENTER = (0x0D, 0x0A)
_CTRL_C = 0x03
_KITTY_ENTER = 13
_KITTY_ALT_CTRL_SUPER = 2 | 4 | 8
_KITTY_CTRL = 4
_KITTY_RELEASE = 3
# The kitty protocol reports keys that produce no text (modifiers, function
# and media keys) as code points in the Unicode private use area.
_KITTY_FUNCTIONAL = range(0xE000, 0xF900)
_PASTE_START = 200


class InputTracker:
    """Tracks whether the user has typed text they have not yet submitted.

    Fed the raw bytes the terminal sends, it tells keystrokes that add text
    to the prompt apart from everything else a terminal emits: cursor and
    function keys, mouse and focus reports, and replies to queries, all of
    which arrive as escape sequences. Enter submits the prompt and Ctrl+C
    clears it, so either marks the prompt empty again.

    The answer errs toward "has text": text deleted with backspace still
    counts, since the cost of a wrong "empty" is a command typed into the
    middle of the user's draft.
    """

    _GROUND, _ESCAPE, _CSI, _SS3, _STRING, _STRING_ESCAPE = range(6)

    def __init__(self) -> None:
        self._state = self._GROUND
        self._params = bytearray()
        self.dirty = False

    def feed(self, data: bytes) -> None:
        for byte in data:
            self._feed_byte(byte)

    def _feed_byte(self, byte: int) -> None:
        state = self._state
        if state == self._GROUND:
            if byte == _ESC:
                self._state = self._ESCAPE
            elif byte in _ENTER or byte == _CTRL_C:
                self.dirty = False
            elif byte >= 0x20 and byte != 0x7F:
                self.dirty = True
        elif state == self._ESCAPE:
            if byte == ord("["):
                self._params.clear()
                self._state = self._CSI
            elif byte == ord("O"):
                self._state = self._SS3
            elif byte in b"]P_^X":
                self._state = self._STRING
            elif byte != _ESC:
                # Alt+key: a chord, not text.
                self._state = self._GROUND
        elif state == self._CSI:
            if 0x40 <= byte <= 0x7E:
                self._end_csi(byte)
                self._state = self._GROUND
            else:
                self._params.append(byte)
        elif state == self._SS3:
            self._state = self._GROUND
        elif state == self._STRING:
            if byte == _BEL:
                self._state = self._GROUND
            elif byte == _ESC:
                self._state = self._STRING_ESCAPE
        elif state == self._STRING_ESCAPE:
            self._state = self._GROUND if byte == ord("\\") else self._STRING

    def _end_csi(self, final: int) -> None:
        """Account for the two CSI sequences that carry text or a submit:
        a kitty-protocol key report (`CSI code ; mods u`) and the start of a
        bracketed paste (`CSI 200 ~`)."""
        fields = [field.split(b":") for field in bytes(self._params).split(b";")]
        if not fields[0][0].isdigit():
            return
        code = int(fields[0][0])
        if final == ord("~") and code == _PASTE_START:
            self.dirty = True
        if final != ord("u"):
            return
        # The second field is `modifiers:event`, with modifiers offset by one.
        mods = fields[1] if len(fields) > 1 else []
        modifiers = int(mods[0]) - 1 if mods and mods[0].isdigit() else 0
        if len(mods) > 1 and mods[1].isdigit() and int(mods[1]) == _KITTY_RELEASE:
            return
        if code == _KITTY_ENTER and modifiers == 0:
            self.dirty = False
        elif code == ord("c") and modifiers == _KITTY_CTRL:
            self.dirty = False
        elif code < 0x20 or code == 0x7F or code in _KITTY_FUNCTIONAL:
            return
        elif not modifiers & _KITTY_ALT_CTRL_SUPER:
            self.dirty = True


class PtyRunner:
    """Runs a child on a pseudo-terminal it owns, so it can type into it.

    The child sees an ordinary terminal: everything the user types is
    relayed to it and everything it draws is relayed back, with window
    resizes and job-control stops passed through. Owning the terminal is
    what lets `request_compact` enter a slash command as if the user had,
    which is the only way to compact a running interactive session from
    outside it.
    """

    def __init__(self, stdin_fd: int = 0, stdout_fd: int = 1) -> None:
        self._stdin_fd = stdin_fd
        self._stdout_fd = stdout_fd
        self._lock = threading.Lock()
        self._input = InputTracker()
        # Bytes waiting to be typed, each with the monotonic time it is due.
        self._pending: list[tuple[float, bytes]] = []
        self._master_fd: int | None = None
        self._pid = 0
        self._gate_w = -1
        # Set by signal handlers, which do nothing else but wake the relay
        # loop; the loop acts on them.
        self._resized = False
        self._terminated = False
        self._wake_r, self._wake_w = os.pipe()
        os.set_blocking(self._wake_w, False)

    def request_compact(self, session_id: str) -> str | None:
        """Type `/compact` into the child's prompt and submit it.

        Safe to call from any thread; the relay loop does the typing.

        Args:
            session_id: The session to compact. The runner fronts a single
                terminal, so this is whichever session that terminal shows.

        Returns:
            None once the keystrokes are queued, or the reason they were
            not: the child is not running, or the prompt holds a draft.
        """
        with self._lock:
            if self._master_fd is None:
                return SKIP_NO_TERMINAL
            if self._input.dirty:
                return SKIP_INPUT_NOT_EMPTY
            now = time.monotonic()
            self._pending.append((now, _COMPACT_COMMAND))
            self._pending.append((now + _SUBMIT_DELAY_SEC, _SUBMIT))
        self._wake()
        return None

    def start(self, claude_bin: str, args: list[str], env: dict[str, str]) -> None:
        """Fork the child onto a new pseudo-terminal, held just short of exec.

        Forking is kept apart from `wait` so the launcher can fork while
        this process is still single-threaded, start the proxy's threads,
        and only then let the child run: a fork taken with other threads
        alive can deadlock the child, and a child let loose before the
        proxy listens would find nothing at ANTHROPIC_BASE_URL.

        Args:
            claude_bin: Name or path of the executable to run.
            args: Arguments passed to it.
            env: Its environment.
        """
        executable = shutil.which(claude_bin, path=env.get("PATH")) or claude_bin
        argv = [claude_bin, *args]
        gate_r, gate_w = os.pipe()
        pid, master_fd = pty.fork()
        if pid == 0:
            os.close(gate_w)
            _supervise(executable, argv, env, gate_r)
        os.close(gate_r)
        self._pid = pid
        self._gate_w = gate_w
        self._copy_window_size(master_fd)
        with self._lock:
            self._master_fd = master_fd

    def wait(self) -> int:
        """Release the child started by `start` and relay this terminal to
        it until it exits.

        Returns:
            The child's exit code, or 128 + N if signal N killed it.
        """
        pid, master_fd = self._pid, self._master_fd
        assert master_fd is not None, "wait() called before start()"
        saved_tty = termios.tcgetattr(self._stdin_fd)
        handlers = {
            signal.SIGWINCH: self._on_winch,
            signal.SIGCHLD: self._on_wake,
            signal.SIGTERM: self._on_terminate,
            signal.SIGHUP: self._on_terminate,
        }
        previous = {sig: signal.signal(sig, handler) for sig, handler in handlers.items()}
        try:
            tty.setraw(self._stdin_fd)
            os.write(self._gate_w, b"\0")
            os.close(self._gate_w)
            return self._relay(pid, master_fd, saved_tty)
        finally:
            with self._lock:
                self._master_fd = None
                self._pending.clear()
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, saved_tty)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            os.close(master_fd)

    def spawn(self, claude_bin: str, args: list[str], env: dict[str, str]) -> int:
        """Run the child to completion, relaying this terminal to it.

        Args:
            claude_bin: Name or path of the executable to run.
            args: Arguments passed to it.
            env: Its environment.

        Returns:
            The child's exit code, or 128 + N if signal N killed it.
        """
        self.start(claude_bin, args, env)
        return self.wait()

    def _relay(self, pid: int, master_fd: int, saved_tty: list) -> int:
        """Shuttle bytes between the real terminal and the child until the
        child exits, typing queued input when it falls due."""
        fds = [self._stdin_fd, master_fd, self._wake_r]
        while True:
            status = self._reap(pid, master_fd, saved_tty)
            if status is not None:
                return status
            if self._terminated:
                os.kill(pid, signal.SIGHUP)
                return _exit_code(os.waitpid(pid, 0)[1])
            if self._resized:
                self._resized = False
                self._copy_window_size(master_fd)
            readable, _, _ = select.select(fds, [], [], self._type_due(master_fd))
            if self._wake_r in readable:
                os.read(self._wake_r, _READ_SIZE)
            if master_fd in readable:
                try:
                    output = os.read(master_fd, _READ_SIZE)
                except OSError as exc:
                    if exc.errno != errno.EIO:
                        raise
                    output = b""
                if not output:
                    # The child closed its terminal; all that is left is to reap it.
                    return _exit_code(os.waitpid(pid, 0)[1])
                _write_all(self._stdout_fd, output)
            if self._stdin_fd in readable:
                typed = os.read(self._stdin_fd, _READ_SIZE)
                if not typed:
                    fds.remove(self._stdin_fd)
                    continue
                with self._lock:
                    self._input.feed(typed)
                _write_all(master_fd, typed)

    def _type_due(self, master_fd: int) -> float | None:
        """Type every queued input that has fallen due.

        Returns:
            Seconds until the next queued input is due, or None if the
            queue is empty.
        """
        while True:
            with self._lock:
                if not self._pending:
                    return None
                due, data = self._pending[0]
                wait = due - time.monotonic()
                if wait > 0:
                    return wait
                self._pending.pop(0)
            _write_all(master_fd, data)

    def _reap(self, pid: int, master_fd: int, saved_tty: list) -> int | None:
        """Check on the child without blocking.

        A child that stopped itself (Ctrl+Z) takes this process down with
        it, handing the terminal back to the shell; both carry on when the
        shell resumes this process. Stopping with SIGTSTP rather than
        SIGSTOP means that with no job-control shell to resume it, this
        process is not stopped at all and the child simply carries on.

        Returns:
            The child's exit code if it has exited, else None.
        """
        reaped, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
        if reaped == 0:
            return None
        if os.WIFSTOPPED(status):
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, saved_tty)
            os.kill(os.getpid(), signal.SIGTSTP)
            tty.setraw(self._stdin_fd)
            self._copy_window_size(master_fd)
            os.kill(pid, signal.SIGCONT)
            return None
        return _exit_code(status)

    def _copy_window_size(self, master_fd: int) -> None:
        try:
            size = fcntl.ioctl(self._stdout_fd, termios.TIOCGWINSZ, b"\0" * 8)
            fcntl.ioctl(master_fd, termios.TIOCSWINSZ, size)
        except OSError:
            pass

    def _on_winch(self, signum: int, frame: FrameType | None) -> None:
        self._resized = True
        self._wake()

    def _on_terminate(self, signum: int, frame: FrameType | None) -> None:
        self._terminated = True
        self._wake()

    def _on_wake(self, signum: int, frame: FrameType | None) -> None:
        self._wake()

    def _wake(self) -> None:
        try:
            os.write(self._wake_w, b"\0")
        except BlockingIOError:
            pass


def _supervise(executable: str, argv: list[str], env: dict[str, str], gate_r: int) -> NoReturn:
    """Run the real child as a foreground job of the new pseudo-terminal.

    Called in the process `pty.fork` created, which leads the terminal's
    session. Exec'ing the child here directly would leave it in an orphaned
    process group, where the kernel discards SIGTSTP and Ctrl+Z could never
    suspend it. So this process stays on as the parent a shell would be: it
    puts the child in its own foreground process group, and when the child
    stops, stops itself to tell the launcher, then continues the child once
    the launcher continues it.

    Args:
        executable: Path of the executable to run.
        argv: Its argument vector.
        env: Its environment.
        gate_r: Read end of the pipe that releases the exec; EOF on it means
            the launcher died first.
    """
    code = 127
    try:
        # A background process changing the foreground group is sent SIGTTOU.
        signal.signal(signal.SIGTTOU, signal.SIG_IGN)
        job = os.fork()
        if job == 0:
            os.setpgid(0, 0)
            os.tcsetpgrp(0, os.getpgrp())
            signal.signal(signal.SIGTTOU, signal.SIG_DFL)
            if os.read(gate_r, 1):
                os.execve(executable, argv, env)
            os._exit(127)
        os.close(gate_r)
        while True:
            _, status = os.waitpid(job, os.WUNTRACED)
            if not os.WIFSTOPPED(status):
                code = _exit_code(status)
                break
            os.kill(os.getpid(), signal.SIGSTOP)
            os.tcsetpgrp(0, job)
            os.killpg(job, signal.SIGCONT)
    except OSError:
        pass
    os._exit(code)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _exit_code(status: int) -> int:
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return os.WEXITSTATUS(status)
