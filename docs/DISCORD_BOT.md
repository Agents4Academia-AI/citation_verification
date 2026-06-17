# Discord bot (`cverify-bot`)

A thin Discord front-end for the citation verifier. One slash command,
**`/check`**, pipes an arXiv paper through the same
`citation_verifier.orchestrator.run_verification` the CLI uses and posts the
hallucination report back to the channel.

It is a *front-end only*: it owns no verification logic, depends only on the
package's public surface, and degrades to a clear message on any failure. Code
lives in [`src/citation_verifier/bot/`](../src/citation_verifier/bot/).

## Commands

| Command | What it does |
| --- | --- |
| `/check paper [backend] [full]` | Verify the citations in an arXiv paper (🧪 test sample by default). |
| `/help` | Usage + how to read a result (the verdict legend). |
| `/ping` | Health check (gateway latency + current mode). |

`/check` accepts **any** of these for `paper` (all normalize to the same id):

- `2505.03335` — bare arXiv id (also `2505.03335v2`)
- `https://arxiv.org/abs/2505.03335`
- `https://arxiv.org/pdf/2505.03335`

Options:
- **backend** — `agentic` (default; fast, free, grounds existence against
  Crossref/arXiv) or `claude_code` (deeper LLM check; slower, spends tokens).
- **full** — `false` (default) runs a 🧪 **test sample** of the first
  `BOT_TEST_LIMIT` (default 5) citations, loudly labeled PARTIAL; `true`
  verifies **all** citations — the real, whole-paper verdict.

The reply is a compact embed (headline + counts + the flagged citations) with
the **full per-citation report attached as a `.md` file**.

## Test phase: the default is a sample

During the testing phase a bare `/check` is **not** a full-paper verdict — it
verifies only the first `BOT_TEST_LIMIT` (default 5) citations and labels the
result loudly: a 🧪 **TEST SAMPLE** banner, an amber embed (never green), counts
shown as "5 of 142 (sample)", and a `-sample.md` report. Pass **`full:true`** for
the real, whole-paper verdict (a green embed when clean, a `-full.md` report).

A `full:true` run on a large bibliography may take minutes; it still delivers —
past Discord's 15-min interaction window the bot posts a channel message that
@-mentions you. There is **no** silent global cap.

> Caching is **off by default** during the testing phase (`BOT_USE_CACHE=0`), so
> every `/check` runs fresh. Set `BOT_USE_CACHE=1` to resume a prior
> `report.json` — repeat checks of the same paper then return instantly, marked
> ♻️ CACHED.

---

## Make it available on your server

### 1. Install

```bash
uv pip install -e '.[bot]'      # or: pip install -e '.[bot]'
```

### 2. Get a bot token

1. Go to <https://discord.com/developers/applications> → **New Application**.
2. **Bot** tab → **Reset Token** → copy the token.
3. No privileged intents are required (slash commands don't need Message
   Content / Server Members).

Put the token in `.env` (gitignored — never commit it):

```dotenv
DISCORD_BOT_TOKEN=your-bot-token-here
# Optional but recommended: register /check INSTANTLY in one server.
# Enable Developer Mode, right-click the server icon → Copy Server ID.
DISCORD_GUILD_ID=1516408471208202260
# For the deeper backend, set a real model id (the claude_code backend uses it):
MODEL_JUDGE=claude-sonnet-4-6
```

See [`.env.example`](../.env.example) for every knob (`BOT_DEFAULT_BACKEND`,
`BOT_TEST_LIMIT`, `BOT_USE_CACHE`, …).

### 3. Invite the bot to your server  ← the step that "makes it available"

Open this URL (it requests the `bot` + `applications.commands` scopes and just
the permissions needed to post a report — View Channel, Send Messages, Embed
Links, Attach Files), pick your server, **Authorize**:

```
https://discord.com/oauth2/authorize?client_id=<YOUR_APP_ID>&permissions=52224&scope=bot+applications.commands
```

Replace `<YOUR_APP_ID>` with your application's **Application ID** (Developer
Portal → General Information). The running bot also prints a ready-to-click
invite URL with the id already filled in (see the `Invite URL:` log line on
startup).

> You must invite the bot **before** the guild command sync can work. If the bot
> isn't in the server yet you'll see `Guild sync … 403 Forbidden (Missing
> Access)` — that's expected; invite it, then restart.

### 4. Run it

```bash
cverify-bot                       # console script
# or
python -m citation_verifier.bot
```

On startup it syncs `/check`, `/help`, `/ping` to your `DISCORD_GUILD_ID`
(**instant**). Without a guild id it does a **global** sync, which can take up to
~1 hour to appear.

### 5. Use it

In any channel the bot can see (e.g. your test channel
`#…` / id `1516409153906540554`):

```
/check 2505.03335                                      # 🧪 test sample (first 5)
/check https://arxiv.org/abs/2505.03335
/check https://arxiv.org/pdf/2505.03335
/check paper:2505.03335 backend:claude_code full:true   # full, whole-paper verdict
```

---

## How to read a result

| Badge | Meaning |
| --- | --- |
| 🔴 **Fabricated** | The cited paper could not be found in any source — likely hallucinated. |
| 🟠 **Doesn't support** | The source is real but doesn't back the sentence that cites it. |
| 🟡 **Unverified** | Couldn't confirm (no DOI/arXiv match — e.g. a blog post or system card). |
| 🟢 **OK** | Exists and supports the claim. |

The attached report (`citation-report-<id>-sample.md` for a test run, `-full.md`
for `full:true`) is the full SKILL.md table + summary + token/cost footer for
every citation; a test-sample report opens with a 🧪 scope header.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `DISCORD_BOT_TOKEN is not set` | Add it to `.env` or export it. |
| `LoginFailure` | Token is wrong/rotated — reset it in the Developer Portal. |
| `Guild sync … 403 Missing Access` | Bot isn't in that server yet — use the invite URL, then restart. |
| `/check` not showing up | Guild sync needs the bot invited; global sync takes ~1h. Set `DISCORD_GUILD_ID`. |
| Check takes minutes | A bare `/check` is already a fast 5-citation 🧪 test sample; a `full:true` run on a large bibliography may take minutes but still delivers (the bot @mentions you past the 15-min window). Don't add a global cap. |
| `claude_code` errors on model id | Set a valid `MODEL_JUDGE` in `.env` (e.g. `claude-sonnet-4-6`). |
