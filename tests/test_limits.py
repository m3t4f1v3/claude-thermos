from claude_thermos.limits import LimitWindow, RateLimits, parse_rate_limits, rejection_headers

_HEADERS = {
    "anthropic-ratelimit-unified-status": "allowed",
    "anthropic-ratelimit-unified-5h-status": "allowed",
    "anthropic-ratelimit-unified-5h-utilization": "0.4",
    "anthropic-ratelimit-unified-5h-reset": "1000",
    "anthropic-ratelimit-unified-7d-status": "allowed",
    "anthropic-ratelimit-unified-7d-utilization": "0.37",
    "anthropic-ratelimit-unified-7d-reset": "5000",
    "anthropic-ratelimit-unified-overage-status": "rejected",
    "request-id": "req_1",
}


def test_parse_reads_every_window() -> None:
    limits = parse_rate_limits(_HEADERS)

    assert limits is not None
    assert limits.windows == {
        "5h": LimitWindow(utilization=0.4, resets_at=1000.0, status="allowed"),
        "7d": LimitWindow(utilization=0.37, resets_at=5000.0, status="allowed"),
    }


def test_parse_matches_header_names_case_insensitively() -> None:
    limits = parse_rate_limits({"Anthropic-Ratelimit-Unified-5h-Utilization": "0.25"})

    assert limits is not None
    assert limits.session_utilization(now=0) == 0.25


def test_parse_returns_none_without_utilization_headers() -> None:
    assert parse_rate_limits({"request-id": "req_1", "content-type": "text/event-stream"}) is None


def test_parse_skips_a_malformed_utilization() -> None:
    assert parse_rate_limits({"anthropic-ratelimit-unified-5h-utilization": "n/a"}) is None


def test_session_utilization_is_none_without_a_session_window() -> None:
    limits = RateLimits({"7d": LimitWindow(utilization=0.5)})

    assert limits.session_utilization(now=0) is None
    assert limits.session_resets_at() is None


def test_utilization_drops_to_zero_once_the_window_rolls_over() -> None:
    limits = parse_rate_limits(_HEADERS)

    assert limits is not None
    assert limits.session_utilization(now=999) == 0.4
    assert limits.session_utilization(now=1000) == 0.0


def test_headroom_follows_the_fullest_window() -> None:
    limits = RateLimits(
        {
            "5h": LimitWindow(utilization=0.4, resets_at=1000),
            "7d": LimitWindow(utilization=0.9, resets_at=5000),
        }
    )

    assert abs(limits.headroom(now=0) - 0.1) < 1e-9
    # Once the weekly window rolls over, the session window is the tight one.
    assert abs(limits.headroom(now=5000) - 1.0) < 1e-9
    assert abs(limits.headroom(now=999) - 0.1) < 1e-9


def test_headroom_never_goes_negative() -> None:
    limits = RateLimits({"5h": LimitWindow(utilization=1.2)})

    assert limits.headroom(now=0) == 0.0


def test_rejected_window_blocks_until_it_resets() -> None:
    limits = RateLimits(
        {
            "5h": LimitWindow(utilization=1.0, resets_at=1000, status="rejected"),
            "7d": LimitWindow(utilization=0.5, resets_at=5000, status="allowed"),
        }
    )

    assert limits.allowed(now=999) is False
    assert limits.allowed(now=1000) is True


def test_warning_status_is_still_allowed() -> None:
    limits = RateLimits({"5h": LimitWindow(utilization=0.9, status="allowed_warning")})

    assert limits.allowed(now=0) is True


def test_overage_status_decides_whether_a_limit_stops_the_session() -> None:
    allowed = parse_rate_limits(
        {
            "anthropic-ratelimit-unified-5h-utilization": "0.99",
            "anthropic-ratelimit-unified-overage-status": "allowed",
        }
    )
    refused = parse_rate_limits(
        {
            "anthropic-ratelimit-unified-5h-utilization": "0.99",
            "anthropic-ratelimit-unified-overage-status": "rejected",
        }
    )
    unreported = parse_rate_limits({"anthropic-ratelimit-unified-5h-utilization": "0.99"})

    assert allowed is not None and allowed.overflow_covered() is True
    assert refused is not None and refused.overflow_covered() is False
    assert unreported is not None and unreported.overflow_covered() is False


def test_rejection_headers_report_the_session_limit_reached() -> None:
    limits = RateLimits(
        {
            "5h": LimitWindow(utilization=0.96, resets_at=18_000, status="allowed_warning"),
            "7d": LimitWindow(utilization=0.4, resets_at=600_000, status="allowed"),
        }
    )

    headers = rejection_headers(limits, resets_at=18_000.4, now=3_000)

    assert headers["anthropic-ratelimit-unified-status"] == "rejected"
    assert headers["anthropic-ratelimit-unified-reset"] == "18000"
    assert headers["anthropic-ratelimit-unified-representative-claim"] == "five_hour"
    assert headers["anthropic-ratelimit-unified-5h-status"] == "rejected"
    assert headers["anthropic-ratelimit-unified-overage-status"] == "rejected"
    assert headers["anthropic-ratelimit-unified-7d-utilization"] == "0.4"
    assert headers["anthropic-ratelimit-unified-7d-status"] == "allowed"
    assert headers["retry-after"] == "15001"
    assert headers["x-should-retry"] == "false"
    # Claude Code parses its own headers back the same way.
    parsed = parse_rate_limits(headers)
    assert parsed is not None and parsed.allowed(3_000) is False
