# claude-thermos

> **This is a fork** of [izeigerman/claude-thermos](https://github.com/izeigerman/claude-thermos).
> It adds [keepalive after the turn ends](#keepalive-session-budget-and-compaction), a budget
> drawn from your subscription's session limit, and a `/compact` fallback. It also fixes two
> things upstream gets wrong for subscribers: it warms a one-hour cache on a one-hour schedule
> rather than every 4.5 minutes, and it streams responses through instead of holding each one
> until it finishes.

**Stop paying to rebuild your Claude Code cache.** When your main agent waits on a subagent for more than 5 minutes, its prompt cache silently expires, and the next turn re-encodes your entire conversation at the write rate instead of reading it back cheap. On long sessions with many subagents that's roughly 20% of your bill. `claude-thermos` keeps the cache warm so you never pay that tax.

## Use

Run Claude Code exactly as you normally would, but through `claude-thermos` with `uvx`:

```bash
uvx claude-thermos                     # instead of: claude
uvx claude-thermos -p "fix the bug"    # any claude args pass straight through
```

Requires Python 3.11+ and the `claude` CLI on your `PATH`.

Use Claude Code 2.1.285 or later. Behind a custom `ANTHROPIC_BASE_URL`, older versions hold
models to a 200K context window unless you select a `[1m]` variant, so a 1M-window model such as
Opus 4.7+ or Sonnet 5 auto-compacts at about 167K tokens instead of about 967K. `claude --version`
shows which you have.

That's it. Warming runs automatically in the background. To disable it for a run without changing the command, set `CLAUDE_THERMOS_DISABLE=1`.

Tuning (all optional):

| Flag                | Default | Meaning                                                     |
|---------------------|---------|-------------------------------------------------------------|
| `--idle`            | `270`   | Seconds the main agent must be idle before warming kicks in |
| `--interval`        | `270`   | Seconds between warming cycles                              |
| `--max-cycles`      | `4`     | Max warms per idle episode (`auto` for unlimited)           |
| `--subagent-window` | `540`   | Seconds a subagent counts as "still active"                 |
| `--keepalive` / `--no-keepalive` | on | Keep warming after the turn ends, while the session waits on you |
| `--keepalive-interval` | `auto` | Seconds between warms of a one-hour cache (`auto` is 55 minutes) |
| `--ping-budget`     | `10`    | Percent of your session limit one idle spell may spend on warms |
| `--session-reserve` | `5`     | Percent of each subscription limit that warms must leave unused |
| `--pause-at`        | `95`    | Percent of your session limit at which the conversation pauses until it resets (`100` never pauses) |
| `--compact` / `--no-compact` | on | Run `/compact` when an ended turn can no longer be kept warm |
| `--compact-min-tokens` | `100000` | Smallest conversation worth compacting                  |

`--idle` and `--interval` apply to a five-minute cache. A one-hour cache, which is what a Claude
subscription gives the main conversation, is warmed on `--keepalive-interval` instead.

### Choosing which `claude` to run

By default `claude-thermos` launches the `claude` found on your `PATH`. Point
it at a different binary with `--bin`, or the `CLAUDE_THERMOS_BIN` environment
variable. A bare name is looked up on `PATH` (so `--bin claude-nightly`
works); a full path is used as-is. This is handy for a vendored build or a
wrapper that exports a different `CLAUDE_CONFIG_DIR` per account.

```bash
claude-thermos --bin /path/to/bin/claude -p "fix the bug"
# or
export CLAUDE_THERMOS_BIN=/path/to/bin/claude
claude-thermos -p "fix the bug"
```

The flag must come before any passthrough `claude` args.

## Keepalive, session budget, and compaction

Upstream only warms while the main agent waits on a subagent. This fork also covers the case
where the turn has **ended** and you have walked away: you run something that fills the context,
leave, and come back after the cache has expired to a full-price rewrite of the whole
conversation.

Once a turn ends, the launcher replays the last request every 55 minutes (for a one-hour cache)
with `max_tokens: 1`, exactly as upstream does during a subagent wait. Each warm is a cache read
of the whole conversation, so on a subscription it counts against your limits. Three caps decide
when an idle spell stops being warmed:

- **`--max-cycles`** (default 4): at most this many warms. Four warms keep a one-hour cache alive
  for about 4.6 hours after the turn ends. A warm costs 0.1x and the rewrite it avoids costs 2x,
  so past roughly 20 warms the keepalive costs more than it saves.
- **`--ping-budget`** (default 10): the warms may use at most this percent of your five-hour
  session limit. Spend is measured from the utilization the API reports on each response, which
  is account-wide and rounded to whole percents, so usage by your other sessions on the same
  account in the meantime counts too.
- **`--session-reserve`** (default 5): a warm is skipped if it would leave less than this
  percent of the session limit, or of the weekly limit, unused.

When a cap stops the warming and the turn has ended, the launcher types `/compact` into the
session while the cache is still warm. The compaction reads the conversation at the cached rate,
and what you come back to is a summary that is cheap to rewrite. It is skipped when:

- the conversation is under `--compact-min-tokens`, where a summary saves little;
- the turn is still running a tool or waiting on a permission prompt;
- you left text in the prompt, since the command would be typed into your draft;
- the session limit is already exhausted, or the cache has already expired (for example the
  machine slept through a warm). Neither is worth paying full price for.

With an API key there are no subscription limits to read, so only `--max-cycles` applies.

### When Claude Code changes its login token

On a subscription, Claude Code swaps its OAuth access token for a new one about every eight
hours, and the old one is revoked at once. A warm replays a request the session sent earlier, so
an idle spell that runs across a swap would otherwise be left warming with a dead token.

A warm therefore authenticates with the newest credentials any request in the session carried,
subagent and side requests included. If the API still answers `401`, the launcher reads the
current token from the login Claude Code stores in `$CLAUDE_CONFIG_DIR/.credentials.json`
(`~/.claude/.credentials.json` by default) and sends the warm again. It only does so for a token
it has itself seen in that file, so a login belonging to another account is never sent.

Warming stops with the reason `unauthorized` when there is no newer token to try: the stored
login has ended, the token came from somewhere else (`CLAUDE_CODE_OAUTH_TOKEN`, a wrapper that
sets `CLAUDE_CONFIG_DIR` only for `claude`, macOS, where the login is in the Keychain), or the
proxy is the daemon, which does not know its clients' config directories. It starts again if the
session sends a request with different credentials while the cache is still alive. If you use a
per-account wrapper, export `CLAUDE_CONFIG_DIR` before `claude-thermos` rather than inside
`--bin`, so the launcher reads the same login `claude` does.

### Pausing before the session limit runs out

Running out of the five-hour session limit stops everything until it resets: the API refuses
every request, warms and `/compact` included. A long turn that runs the limit dry while you are
away therefore leaves a cache that expires an hour later, and if the reset is further off than
that, you come back to a full rewrite.

So once the session limit is used past `--pause-at` (default 95%), the proxy stops the
conversation at the next request boundary and keeps what is left of the limit for warms. It
answers the conversation's requests with the same refusal the API sends at the limit, so Claude
Code does what it does at a real limit: it shows `Usage limit reached · continuing automatically
at <reset>` and waits in the open session. The warmer keeps the cache alive until then, past
`--max-cycles` and the reserve, which exist for exactly this; only the limit itself stops it. At
the reset Claude Code sends the request again and the turn carries on against a warm cache.

- It only pauses when the reset is far enough away that the cache would otherwise expire first.
  Closer than that, the limit comes back while the cache is still alive, so nothing is gained.
- A pause holds until the reset, even once a new one would no longer start, because typing into
  Claude Code's wait would cancel it.
- A prompt you send yourself past the threshold lifts the pause until the reset: you have chosen
  to spend the rest of the limit.
- It never pauses when paid extra usage would carry you past the limit, or for a session no
  terminal has open.
- The status line below shows `paused by thermos until <time>` while it holds, since Claude Code
  reports the pause as an ordinary usage limit.

To make this work the launcher does three things upstream does not:

- It runs `claude` on a pseudo-terminal it owns, which is the only way to enter a slash command
  into a running interactive session. Resizes and Ctrl+Z pass through. `--no-compact` turns this
  off and runs `claude` on your terminal directly.
- It passes `--settings` with three HTTP hooks (`UserPromptSubmit`, `SessionEnd`, `PostCompact`)
  that post to the proxy. They tell the warmer which session your terminal has open, so a session
  you `/clear`ed or a `claude -p` child that merely passed through the proxy is never kept warm,
  and they report compactions, so a summarized-away prefix is not warmed. They merge with your
  own hooks. After a compaction Claude Code shows a `PostCompact ... completed successfully` line.
- It sets `ENABLE_TOOL_SEARCH=true` and `CLAUDE_CODE_ENABLE_FINE_GRAINED_TOOL_STREAMING=1` unless
  you set them yourself. Claude Code turns both off behind any `ANTHROPIC_BASE_URL`, and without
  tool search every MCP tool definition is loaded into the prompt up front.

Limitations: the `/compact` keystrokes go to whatever the terminal is showing, so a dialog you
left open (`/usage`, `/config`) will receive them instead. A session you quit is not warmed,
because the proxy exits with it, so resuming it more than an hour after its last request starts
cold. Running out of a usage limit anyway, by lifting a pause or with `--pause-at 100`, has the
same effect when the reset is more than an hour away.

### Seeing whether the cache is really warm

Claude Code works out its own prompt-cache readout (the `Prompt cache (main)` line in `/usage`,
and `prompt_cache` in status line input) from the requests it sent, and never sees the warms. An
hour after your last turn it reports the cache cold, even when a warm has kept it alive. That
readout can't be corrected from outside Claude Code, so the fork ships a status line that adds the
warms back in:

```json
"statusLine": {
  "type": "command",
  "command": "claude-thermos-statusline",
  "refreshInterval": 60
}
```

It shows `cache warm · 24m left · kept by thermos` while a warm is holding the cache open,
`cache warm · 24m left` while Claude Code's own last request still is, and `cache cold` once both
have lapsed. It reads the session's event log, so in a session not run through thermos it just
reports Claude Code's own expiry. `refreshInterval` matters: a warm fires while Claude Code is
idle, so nothing else would redraw the line. It is a separate script from `claude-thermos` so each
refresh starts in milliseconds. To combine it with an existing status line, pipe the same input to
it from your own script and append its output. In [daemon mode](#daemon-mode-shared-proxy-for-the-ide-and-multiple-terminals)
there is no terminal to type into, so `serve` never compacts; it keeps a session warm only if
you add the hooks to your own settings, pointed at `http://127.0.0.1:<port>/claude-thermos/hook`,
and only until `--session-ttl`.

## Daemon mode (shared proxy for the IDE and multiple terminals)

The default command warms only the `claude` process it launches. Clients that
launch `claude` themselves — the **VSCode/Claude Code extension**, which spawns
its own bundled binary — never go through it, and neither do other terminals.

`claude-thermos serve` runs the warming proxy as a **standalone daemon** on a
fixed loopback port. Point any client at it and they all share one warmer:

```bash
claude-thermos serve --port 8787          # run the daemon (Ctrl-C / SIGTERM to stop)

# then, for any client:
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
claude -p "fix the bug"                    # terminal — warmed by the daemon
```

For the VSCode extension, make sure its process inherits that environment
variable (on macOS, `launchctl setenv ANTHROPIC_BASE_URL http://127.0.0.1:8787`
before launching the app; or export it in the shell you start the editor from).
The extension honors `ANTHROPIC_BASE_URL`, so its traffic then flows through the
daemon and its main agent stays warm while subagents run.

The daemon observes traffic exactly like the launcher and already tracks many
sessions at once, so a single daemon serves every client on the machine. It
evicts sessions idle longer than `--session-ttl` (default `3600s`) so it can run
indefinitely.

Tuning: `serve` accepts the same `--idle/--interval/--max-cycles/--subagent-window`
flags as the default command, plus:

| Flag            | Default                     | Meaning                                       |
|-----------------|-----------------------------|-----------------------------------------------|
| `--port`        | `8787`                      | Loopback port the daemon listens on           |
| `--upstream`    | `https://api.anthropic.com` | Real API the proxy reverse-proxies to         |
| `--session-ttl` | `3600`                      | Seconds a session may sit idle before eviction|

> **Caveat:** `--upstream` must be the real API, never the daemon's own loopback
> address — otherwise the proxy would forward to itself. `serve` rejects a
> loopback upstream, so if you export `ANTHROPIC_BASE_URL` globally, still start
> the daemon with an explicit `--upstream https://api.anthropic.com`.

## Why your cache keeps expiring

Claude Code's prompt cache uses a **5-minute TTL** with an API key, and a **1-hour TTL** for the
main conversation on a Claude subscription within its included usage. Every turn, your whole conversation history is served from cache at **0.1x** the input price instead of being re-sent at full price, as long as the cache stays alive.

The cache expires if more than the TTL passes between requests on the same prefix. On a 5-minute TTL the dominant trigger for that gap is not you thinking. It's the main agent **blocked on a subagent that runs longer than 5 minutes**. A subagent has a different system prompt and tool set, so its requests have a *different* cache prefix and never refresh the main agent's. While the subagent works, the main agent's cached history ages untouched; past 5 minutes it's gone. When the subagent returns, the main agent resumes with a byte-identical, append-only history, and finds its cache missing, forcing a full re-encode at the **1.25x** write rate.

By then the history is large, so the re-encode is expensive: individual collapses re-write 200K to 500K tokens. Measured across roughly 185 local sessions, these rebuilds accounted for about **22% of the total bill**, money spent re-encoding content that was already cached moments earlier.

## How it works

`claude-thermos` launches Claude Code behind a small local reverse proxy (it points `ANTHROPIC_BASE_URL` at a loopback port; all traffic still goes to the real Anthropic API).

1. **Observe.** The proxy watches `/v1/messages` traffic and groups it into sessions and *lineages*, a lineage being one cache prefix, keyed by model + tool set + system text. The first tool-bearing lineage is the **main** agent; the rest are subagents.
2. **Detect the danger window.** When the main lineage goes idle *and* a subagent is actively running, the main prefix is at risk of expiring.
3. **Warm.** On an interval under the 5-minute TTL, it replays the main agent's last real request as a **warm request**: identical cacheable prefix, but `max_tokens: 1` and no streaming. The single token is thrown away; the point is the prefill, which reads and refreshes the full cached prefix. Warm requests go **directly** to the API, never through the proxy, so they can't disturb real traffic.
4. **Result.** When the subagent finishes, the main agent's cache is still warm. It pays a cheap read instead of a full rewrite.

Each warm costs a cache read (0.1x); each rewrite it prevents would have cost a write (1.25x) on a much larger prefix, so the trade is heavily in your favor.

## Event logs & savings

Every session writes to:

```
~/.claude-thermos/logs/<session_id>/
├── events.jsonl    # append-only structured event stream
└── summary.json    # rollup totals, written when the session ends
```

`events.jsonl` records each request/response's token usage plus every warming decision (`warm_fired`, `warm_result`, `cap_reached`, `resume_detected`, and so on). On a subscription, `warm_result` and `cap_reached` carry `session_utilization` and `ping_spent` (both fractions of the session limit), `cap_reached` carries the `reason` warming stopped, and `compact_requested` / `compact_skipped` / `compacted` record what became of the conversation. `paused` (with `resumes_at`) and `pause_lifted` (with the `reason`: `reset` or `prompt`) bracket a pause. A `warm_error` marked `retried` was refused as unauthorized and sent again with the token from Claude Code's stored login. `summary.json` is the rollup you'll usually read:

| Field                    | Meaning                                                                                    |
|--------------------------|--------------------------------------------------------------------------------------------|
| `warms_fired`            | Warm requests sent                                                                         |
| `cache_read_total`       | Tokens read back by those warms                                                            |
| `episodes`               | Idle-with-subagent episodes that ended in a successful resume (a rewrite actually avoided) |
| `rewrite_avoided_tokens` | Tokens that *would have* been re-written, summed across episodes                           |
| `warm_cost`              | What warming cost you: `0.1 × cache_read_total`                                            |
| `rewrite_avoided_cost`   | What it saved: `1.25 × rewrite_avoided_tokens` on a 5-minute cache, `2 ×` on a 1-hour one  |
| `net_savings`            | `rewrite_avoided_cost − warm_cost`                                                         |

All three cost figures are in **base-input-token units** (token counts already weighted by their cache multiplier). To turn `net_savings` into dollars, **multiply it by your model's price per input token**:

```
dollars saved ≈ net_savings × (input token price)
```

For example, at an input price of $3 / 1M tokens, a `net_savings` of `1_200_000` is about `1_200_000 × $3 / 1_000_000 = $3.60` saved that session.
