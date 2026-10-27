"""alpha-score command line.

    alpha-score score essay.txt                 # one file
    alpha-score score essays/ other.md          # several documents, ranked
    pbpaste | alpha-score score -               # stdin
    alpha-score score essay.txt --json          # machine-readable report
    alpha-score models | config | probe | mcp | deploy | delete
"""

import argparse
import json
import sys
import time

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import load_config
from .models import MODELS
from .scorer import CAVEAT, CLASS_NAMES, Scorer, explain

VERDICT_STYLE = dict(high="bold green", medium="yellow", low="dim", noise="red", uncalibrated="cyan")


def _scorer(a):
    return Scorer(load_config(model=a.model, provider=a.provider, base_url=a.base_url, deployment=a.deployment,
                              workers=a.workers, max_passages=a.max_passages, repeats=a.repeats,
                              api_key_env=a.key_env))


def cmd_score(a, con):
    sc = _scorer(a)
    if a.deploy:
        with sc.temporary_deployment(log=con.print):
            return _score(a, con, sc)
    _score(a, con, sc)


def _score(a, con, sc):
    t0 = time.time()
    with con.status(f"scoring with [bold]{sc.model_key}[/] …", spinner="dots"):
        if a.text is not None or a.paths == ["-"]:
            text = a.text if a.text is not None else sys.stdin.read()
            reports = [sc.score_text(text, a.title)]
        else:
            reports = []
            for p in a.paths:
                reports += sc.score_path(p)
            reports.sort(key=lambda r: -(r["alpha"] if r["alpha"] is not None else -1e9))
    cost = sc.cost()
    if a.json:
        print(json.dumps(dict(reports=reports, usage=cost, seconds=round(time.time() - t0, 1)), indent=1))
        return
    render(con, reports, sc.model_key, cost, time.time() - t0, details=a.details or len(reports) == 1)


def render(con, reports, model, cost, secs, details=False):
    t = Table(box=box.SIMPLE_HEAVY, title=f"alpha score · reference model [bold]{model}[/]", title_justify="left")
    t.add_column("document", style="bold", overflow="fold")
    t.add_column("alpha", justify="right")
    t.add_column("verdict")
    cols = [c for c in ("claude", "wiki", "high") if any(c in r["percentiles"] for r in reports)]
    for c in cols:
        t.add_column(f"> {CLASS_NAMES[c]}", justify="right")
    t.add_column("passages", justify="right", style="dim")
    for r in reports:
        a = "GATED" if r["alpha"] is None else f"{r['alpha']:+.2f}"
        t.add_row(escape(r["title"]), a, f"[{VERDICT_STYLE[r['verdict']]}]{r['verdict']}[/]",
                  *[f"{r['percentiles'][c]}%" if c in r["percentiles"] else "–" for c in cols], str(r["n_passages"]))
    con.print(t)
    if details:
        for r in reports:
            body = [explain(r)]
            if not r["gated"]:
                body.append("\n[bold]Where the model was most surprised[/] (surprising token highlighted):")
                for s in r["surprising_spans"]:
                    pre, rest = s["context"].split("[[", 1)
                    tok, post = rest.split("]]", 1)
                    ctx = f"{escape(pre)}[reverse]{escape(tok)}[/reverse]{escape(post)}".replace("\n", " ")
                    body.append(f"  [dim]{s['nll']:>5.1f}[/]  …{ctx}…")
                tp = r["top_passages"][0]
                body.append(f"\n[bold]Highest-alpha passage[/] (#{tp['idx']}, {tp['uzzi']:+.2f}): [italic]{escape(tp['text'])}[/]")
            c = r["components"]
            body.append(f"\n[dim]conv {c['conv']:+.2f} · tail {c['tail']:+.2f} · coherence {c['coherence']:.2f}[/]")
            con.print(Panel("\n".join(body), title=escape(r["title"]), title_align="left", border_style="dim"))
    usd = f"${cost['usd']:.3f}" if cost.get("usd") is not None else "n/a"
    con.print(f"[dim]{cost['requests']} requests · {cost['prompt_tokens']:,} in / {cost['completion_tokens']:,} out tokens"
              f" · {usd} · {secs:.0f}s\n{CAVEAT}[/]")


def cmd_models(a, con):
    t = Table(box=box.SIMPLE_HEAVY, title="reference models (pass the key with --model)", title_justify="left")
    for c in ("key", "kind", "size", "Fireworks", "benchmarked", "note"):
        t.add_column(c, overflow="fold")
    for k, m in MODELS.items():
        fw = ("pay-per-token" if m["serverless"] else "deployment") if m["fireworks"] else "upload"
        t.add_row(k, m["kind"], m["params"], fw, ", ".join(m["tested"]) or "–", m["note"])
    con.print(t)
    if a.serverless:
        from .fireworks import list_serverless

        cfg = load_config()
        if not cfg.api_key:
            return con.print(f"[red]set ${cfg.api_key_env} to list serverless models[/]")
        ours = {m["fireworks"] for m in MODELS.values()}
        con.print("Fireworks pay-per-token models visible to your key (* = in registry):")
        for n in list_serverless(cfg.api_key):
            con.print(("  * " if n in ours else "    ") + n)


def cmd_config(a, con):
    cfg = load_config()
    for k, v in cfg.describe().items():
        con.print(f"[bold]{k:<13}[/] {v}")


def cmd_probe(a, con):
    from .client import probe

    sc = _scorer(a)
    ok = probe(sc.client(), log=con.print)
    sys.exit(0 if ok else 1)


def cmd_deploy(a, con):
    from .fireworks import Fireworks
    from .models import resolve

    cfg = load_config()
    if not (cfg.api_key and cfg.account_id):
        sys.exit(f"needs ${cfg.api_key_env} and FIREWORKS_ACCOUNT_ID")
    fw = Fireworks(cfg.api_key, cfg.account_id)
    if a.cmd == "delete":
        fw.delete(a.deployment or sys.exit("--deployment required"))
        return con.print(f"deleted {a.deployment}")
    base = resolve(a.model)["fireworks"]
    dep = fw.create(base, a.accelerator, a.count, a.precision)
    con.print(f"created deployment {dep} of {base} — billed per GPU-second until you run: alpha-score delete --deployment {dep}")
    try:
        fw.wait(dep, log=con.print)
        con.print(f"model ready after {fw.client(dep, base).wait_ready(log=con.print):.0f}s more")
    except BaseException:
        fw.delete(dep)
        con.print(f"deleted {dep} after failure")
        raise
    con.print(f"\nscore with: alpha-score score <files> --model {a.model} --deployment {dep}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="alpha-score", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"alpha-score {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", help="registry key (see `alpha-score models`) or provider model id")
        p.add_argument("--provider", choices=["fireworks", "openai"])
        p.add_argument("--base-url", help="provider=openai: e.g. http://localhost:8000/v1 for `vllm serve`")
        p.add_argument("--key-env", help="env var holding the API key (default FIREWORKS_API_KEY)")
        p.add_argument("--deployment", help="existing Fireworks deployment id (base models)")
        p.add_argument("--workers", type=int)
        p.add_argument("--max-passages", type=int, help="~100-word passages per document (default 20)")
        p.add_argument("--repeats", type=int, help="average N independent self-sample draws (less noise, N x cost)")

    p = sub.add_parser("score", help="score files, folders, stdin (-) or --text")
    p.add_argument("paths", nargs="*", default=["-"])
    p.add_argument("--text", help="score this string instead of files")
    p.add_argument("--title", default="text", help="name for stdin/--text input")
    p.add_argument("--json", action="store_true", help="print the full report as JSON")
    p.add_argument("--details", action="store_true", help="explain every document (default: only when scoring one)")
    p.add_argument("--deploy", action="store_true",
                   help="Fireworks base models: create a dedicated deployment, score, then delete it (~$8/GPU-hour)")
    common(p)
    p = sub.add_parser("probe", help="check that a provider/model returns prompt logprobs")
    common(p)
    p = sub.add_parser("models", help="list reference models")
    p.add_argument("--serverless", action="store_true", help="also list Fireworks pay-per-token models for your key")
    sub.add_parser("config", help="show the resolved configuration")
    sub.add_parser("mcp", help="run the MCP server on stdio")
    p = sub.add_parser("deploy", help="create a dedicated Fireworks deployment (base models; billed per GPU-second)")
    p.add_argument("--model", required=True)
    p.add_argument("--accelerator")
    p.add_argument("--count", type=int)
    p.add_argument("--precision")
    p = sub.add_parser("delete", help="delete a Fireworks deployment")
    p.add_argument("--deployment", required=True)
    a = ap.parse_args(argv)

    con = Console(stderr=a.cmd == "score" and getattr(a, "json", False))
    if a.cmd == "mcp":
        from .mcp_server import main as mcp_main

        return mcp_main()
    try:
        dict(score=cmd_score, models=cmd_models, config=cmd_config, probe=cmd_probe, deploy=cmd_deploy,
             delete=cmd_deploy)[a.cmd](a, con)
    except ValueError as e:
        con.print(f"[red]{escape(str(e))}[/]")
        sys.exit(2)


if __name__ == "__main__":
    main()
