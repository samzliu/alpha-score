# Using alpha-score as an MCP server

`alpha-score-mcp` (or `alpha-score mcp`) runs a stdio MCP server. Any MCP client can call it: Claude Code, Claude
Desktop, Cursor, Windsurf or your own agent. It reads the same configuration as the CLI. Usually that is just
`FIREWORKS_API_KEY`, passed in the server's `env` block.

![Claude Code using the alpha-score MCP server](screenshots/mcp-claude-code.png)

*A real Claude Code session (`claude -p` with this server); see `docs/screenshots/mcp-claude-code.html`. Claude
noticed on its own that the single most surprising span was an Old Yiddish quotation, which is unusual vocabulary
rather than an idea. That is why reports include the spans and not just a number.*

## Tools

| tool | arguments | returns |
|---|---|---|
| `score_text` | `text`, `title?`, `model?` | one report |
| `compare_texts` | `texts` (title → text), `model?` | reports ranked by alpha |
| `score_file` | `path` (.txt/.md file or folder), `model?` | reports ranked by alpha |
| `list_models` | | reference models, whether each is pay-per-token, and benchmark notes |
| `check_setup` | `model?` | whether the key is set and the provider returns prompt logprobs |

A report contains:
- `alpha`, `verdict` (`high` / `medium` / `low` / `noise`) and a one-line `summary`.
- `percentiles` vs. the benchmark classes (Claude essays, Wikipedia, high-alpha essays, crank posts).
- `surprising_spans`: the tokens the model least expected, in context, with the token in `[[…]]`.
- `top_passages`, `components` (`conv`, `tail`, `coherence`), and `usage` (tokens and USD).

The server's instructions tell the client what alpha means and that it does not check truth.

**Latency and cost:** a 2,000-word essay is 20 passages. It takes about 20–60 s and costs about $0.01 on the default
model. Set `ALPHA_SCORE_MAX_PASSAGES` lower for faster, noisier scores.

## Install

```bash
pip install "git+https://github.com/samzliu/alpha-score"     # or: pip install -e . from a clone
# or run it without installing:  uvx --from "git+https://github.com/samzliu/alpha-score" alpha-score-mcp
```

## Claude Code

```bash
claude mcp add alpha-score -e FIREWORKS_API_KEY=fw_... -- alpha-score-mcp
# with uvx (no install):
claude mcp add alpha-score -e FIREWORKS_API_KEY=fw_... -- uvx --from "git+https://github.com/samzliu/alpha-score" alpha-score-mcp
```

Or add it to a project's `.mcp.json`:

```json
{
  "mcpServers": {
    "alpha-score": {
      "command": "alpha-score-mcp",
      "env": { "FIREWORKS_API_KEY": "${FIREWORKS_API_KEY}" }
    }
  }
}
```

## Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS, or
`%APPDATA%\Claude\claude_desktop_config.json` on Windows, then restart Claude Desktop:

```json
{
  "mcpServers": {
    "alpha-score": {
      "command": "/absolute/path/to/alpha-score-mcp",
      "env": { "FIREWORKS_API_KEY": "fw_...", "ALPHA_SCORE_MODEL": "glm-5.3-flash" }
    }
  }
}
```

Desktop apps don't inherit your shell `PATH`, so use the absolute path from `which alpha-score-mcp`, or
`"command": "uvx"` with `"args": ["--from", "git+https://github.com/samzliu/alpha-score", "alpha-score-mcp"]`.

## Cursor / Windsurf / others

Use the same `mcpServers` block in `~/.cursor/mcp.json`, or in the client's equivalent MCP config file.

## Options (env vars in the `env` block)

| variable | default | |
|---|---|---|
| `FIREWORKS_API_KEY` | required | [fireworks.ai/account/api-keys](https://fireworks.ai/account/api-keys) |
| `ALPHA_SCORE_MODEL` | `glm-5.3-flash` | any key from `alpha-score models` |
| `ALPHA_SCORE_MAX_PASSAGES` | `20` | passages per document (cost and latency scale with this) |
| `ALPHA_SCORE_REPEATS` | `1` | average N self-sample draws: less noise (document SD 0.26 → 0.26/√N), N× cost and time |
| `ALPHA_SCORE_WORKERS` | `16` | concurrent requests; lower it if you hit 429 rate limits |
| `ALPHA_SCORE_PROVIDER`, `ALPHA_SCORE_BASE_URL`, `ALPHA_SCORE_API_KEY_ENV` | | for a self-hosted `vllm serve` (`provider=openai`) |
| `ALPHA_SCORE_DEPLOYMENT`, `FIREWORKS_ACCOUNT_ID` | | to score through an existing dedicated Fireworks deployment |
| `ALPHA_SCORE_CONFIG` | | path to a TOML config file (see `alpha-score.example.toml`) |

## Test it

```bash
FIREWORKS_API_KEY=fw_... python tests/mcp_smoke.py    # starts the server over stdio and calls every tool (~$0.01)
```
