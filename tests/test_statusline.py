import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from claude_thermos.statusline import cache_status, clock_time, main, read_events, render

NOW = 10_000.0


def _status(expires_at: float | None, caching_observed: bool = True) -> dict:
    return {
        "session_id": "s1",
        "prompt_cache": {
            "caching_observed": caching_observed,
            "warm": expires_at is not None and expires_at > NOW,
            "ttl": "1h",
            "expires_at": expires_at,
        },
    }


def _warm(ts: float, ttl_refreshed: bool = True) -> dict:
    return {"ts": ts, "event": "warm_result", "ttl_refreshed": ttl_refreshed, "ttl_sec": 3600}


def _write_log(root: Path, session_id: str, events: list[dict]) -> None:
    session_dir = root / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


def test_no_warms_reports_claude_codes_expiry() -> None:
    assert cache_status(_status(NOW + 600), [], NOW) == "cache warm · 10m left"


def test_cold_without_warms() -> None:
    assert cache_status(_status(NOW - 1), [], NOW) == "cache cold"


def test_warm_after_claude_codes_expiry_keeps_cache_warm() -> None:
    # Claude Code's last request was 70 minutes ago; a warm 30 minutes ago
    # renewed the one-hour TTL.
    status = _status(NOW - 600)
    assert (
        cache_status(status, [_warm(NOW - 1800)], NOW) == "cache warm · 30m left · kept by thermos"
    )


def test_earlier_warm_does_not_shorten_a_later_request() -> None:
    assert cache_status(_status(NOW + 3000), [_warm(NOW - 1800)], NOW) == "cache warm · 50m left"


def test_compaction_discards_earlier_warms() -> None:
    events = [_warm(NOW - 1800), {"ts": NOW - 1700, "event": "compacted"}]
    assert cache_status(_status(NOW - 600), events, NOW) == "cache cold"


def test_warm_that_missed_the_cache_is_ignored() -> None:
    events = [_warm(NOW - 1800, ttl_refreshed=False)]
    assert cache_status(_status(NOW - 600), events, NOW) == "cache cold"


def test_under_a_minute_left() -> None:
    assert cache_status(_status(NOW + 30), [], NOW) == "cache warm · <1m left"


def test_blank_before_any_cache_use() -> None:
    assert cache_status({"session_id": "s1"}, [], NOW) == ""
    assert cache_status(_status(None, caching_observed=False), [], NOW) == ""


def test_read_events_skips_bad_lines(tmp_path: Path) -> None:
    _write_log(tmp_path, "s1", [_warm(NOW)])
    with (tmp_path / "s1" / "events.jsonl").open("a") as log:
        log.write("not json\n[1, 2]\n")
    assert read_events("s1", tmp_path) == [_warm(NOW)]


def test_read_events_refuses_paths_outside_the_log_root(tmp_path: Path) -> None:
    _write_log(tmp_path, "s1", [_warm(NOW)])
    assert read_events("../s1", tmp_path / "logs") == []
    assert read_events("..", tmp_path) == []
    assert read_events("missing", tmp_path) == []


def test_render_reads_the_sessions_log(tmp_path: Path) -> None:
    _write_log(tmp_path, "s1", [_warm(NOW - 1800)])
    stdin = json.dumps(_status(NOW - 600))
    assert render(stdin, tmp_path, NOW) == "cache warm · 30m left · kept by thermos"


def test_render_tolerates_bad_input(tmp_path: Path) -> None:
    assert render("not json", tmp_path, NOW) == ""
    assert render("[]", tmp_path, NOW) == ""
    assert render(json.dumps({"session_id": 5}), tmp_path, NOW) == ""


def test_main_prints_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr("claude_thermos.statusline.default_log_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(_status(4_000_000_000.0))))
    main()
    assert capsys.readouterr().out.startswith("cache warm · ")


def test_statusline_skips_the_proxy_imports() -> None:
    # Claude Code runs the status line on every refresh, so it must start fast.
    code = "import sys, claude_thermos.statusline; print('mitmproxy' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "False"


def test_pause_is_shown_until_it_lifts() -> None:
    paused = [{"ts": NOW - 60, "event": "paused", "resumes_at": NOW + 7200}]
    lifted = [*paused, {"ts": NOW - 30, "event": "pause_lifted", "reason": "prompt"}]
    status = _status(NOW + 600)

    assert cache_status(status, paused, NOW) == (
        f"paused by thermos until {clock_time(NOW + 7200)} · cache warm · 10m left"
    )
    assert cache_status(status, lifted, NOW) == "cache warm · 10m left"
    assert cache_status(status, paused, NOW + 7200) == "cache cold"
