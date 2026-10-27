"""Live MCP smoke test: starts `alpha-score mcp` over stdio and calls every tool (needs FIREWORKS_API_KEY).

    python tests/mcp_smoke.py
"""

import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

g = lambda o, *names: next(getattr(o, n) for n in names if hasattr(o, n))  # mcp 1.x camelCase / 2.x snake_case
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def main():
    params = StdioServerParameters(command=sys.executable, args=["-m", "alpha_score.mcp_server"],
                                   env={**os.environ, "ALPHA_SCORE_MAX_PASSAGES": "3"})
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        init = await s.initialize()
        print("server:", g(init, "server_info", "serverInfo").name, "| instructions:", (init.instructions or "")[:80], "…")
        print("tools:", [t.name for t in (await s.list_tools()).tools])
        for name, args in [
            ("check_setup", {}),
            ("list_models", {}),
            ("score_text", dict(text=open(os.path.join(ROOT, "examples/high_gefilte.txt")).read(), title="gefilte")),
            ("compare_texts", dict(texts={n: open(os.path.join(ROOT, f"examples/{n}.txt")).read()
                                          for n in ("wiki_gefilte", "claude_gefilte", "gib_markov")})),
            ("score_file", dict(path=os.path.join(ROOT, "examples/crank_riots.txt"))),
        ]:
            res = await s.call_tool(name, args)
            assert not g(res, "is_error", "isError"), res.content[0].text
            data = g(res, "structured_content", "structuredContent") or json.loads(res.content[0].text)
            if isinstance(data, dict) and set(data) == {"result"}:
                data = data["result"]  # list return values are wrapped
            if name.startswith(("score", "compare")):
                print(f"{name}: " + "; ".join(r["summary"][:70] for r in data["reports"]), "| usd", data["usage"]["usd"])
            else:
                print(f"{name}: {json.dumps(data)[:150]}")
    print("MCP smoke test PASS")


asyncio.run(main())
