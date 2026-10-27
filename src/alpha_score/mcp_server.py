"""MCP server (stdio): lets Claude Desktop, Claude Code, Cursor, etc. score text for alpha.

    alpha-score mcp            # or: alpha-score-mcp, or: python -m alpha_score.mcp_server

Configuration comes from the same env vars / config file as the CLI (see config.py); in an MCP client config, pass
FIREWORKS_API_KEY (and optionally ALPHA_SCORE_MODEL) in the server's "env" block.
"""

import asyncio
import os

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP

from .client import probe as _probe
from .config import load_config
from .models import MODELS
from .scorer import CAVEAT, Scorer, explain

INSTRUCTIONS = f"""Scores text for "alpha": genuinely new content relative to what a reference language model would
write itself, measured from token log-probabilities (no LLM judge). Higher alpha = more novel ideas in fluent prose.
Each report has `verdict` (high / medium / low / noise), `percentiles` (where the score falls among benchmark
Wikipedia articles, Claude essays, high-alpha human essays and crank posts scored with the same model), and
`surprising_spans` (where the model was most surprised). Needs ~300+ words for a stable score; each ~100-word passage
costs about $0.0006 and a full document takes 20-60 s. Scores carry sampling noise (SD ~0.26 between runs), so treat
differences under ~0.5 between single texts as ties. {CAVEAT}"""

mcp = FastMCP("alpha-score", instructions=INSTRUCTIONS)


def _scorer(model):
    return Scorer(load_config(model=model or None))


def _with_explanations(reports, sc):
    for r in reports:
        r["summary"] = explain(r)
    return dict(reports=reports, usage=sc.cost(), caveat=CAVEAT)


@mcp.tool()
async def score_text(text: str, title: str = "text", model: str = "") -> dict:
    """Score one text for alpha (novel, valuable content vs. a reference LLM). Pass the full text, ideally 300+
    words. `model` is optional (a key from list_models; default glm-5.3-flash). Returns alpha, verdict, benchmark
    percentiles, the most surprising spans and the highest-alpha passages."""
    sc = _scorer(model)
    r = await asyncio.to_thread(sc.score_text, text, title)
    return _with_explanations([r], sc)


@mcp.tool()
async def compare_texts(texts: dict[str, str], model: str = "") -> dict:
    """Score several texts with the same reference model and rank them by alpha. `texts` maps a title to the
    text. Use this to compare drafts, candidate essays, or sources on one topic."""
    sc = _scorer(model)
    rs = await asyncio.to_thread(sc.score_texts, texts)
    return _with_explanations(rs, sc)


@mcp.tool()
async def score_file(path: str, model: str = "") -> dict:
    """Score a local .txt/.md file, or every .txt/.md file in a folder (one document per file), ranked by alpha."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise ValueError(f"no such file or folder: {path}")
    sc = _scorer(model)
    rs = await asyncio.to_thread(sc.score_path, path)
    return _with_explanations(rs, sc)


@mcp.tool()
def list_models() -> list[dict]:
    """Reference models available for scoring, with whether they are pay-per-token and benchmark notes."""
    return [dict(key=k, kind=m["kind"], size=m["params"], pay_per_token=m["serverless"],
                 benchmarked=bool(m["tested"]), note=m["note"]) for k, m in MODELS.items()]


@mcp.tool()
async def check_setup(model: str = "") -> dict:
    """Check the API key and that the provider returns the prompt log-probabilities scoring needs."""
    cfg = load_config(model=model or None)
    lines = []
    try:
        ok = await asyncio.to_thread(_probe, Scorer(cfg).client(), lines.append)
    except Exception as e:  # missing key, unknown model, ...
        ok, lines = False, lines + [str(e)]
    d = cfg.describe()
    return dict(ok=ok, provider=d["provider"], model=d["model"], api_key=d["api_key"], probe=lines)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
