"""Regenerate the CLI screenshots in docs/screenshots/ from real runs (needs FIREWORKS_API_KEY; costs ~$0.10).

    python docs/make_screenshots.py            # SVG (rich) + PNG (headless Chrome, if installed)
"""

import os
import re
import subprocess
import time
from pathlib import Path

from rich.console import Console

from alpha_score.cli import cmd_models, render
from alpha_score.scorer import Scorer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def shot(name, title, fn, width=112):
    con = Console(record=True, width=width, force_terminal=True, color_system="truecolor")
    fn(con)
    svg = OUT / f"{name}.svg"
    con.save_svg(str(svg), title=title)
    if os.path.exists(CHROME):
        w, h = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg.read_text()).groups())
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        f"--screenshot={OUT / f'{name}.png'}", f"--window-size={int(w)},{int(h)}", svg.as_uri()],
                       check=True, capture_output=True)
    print("wrote", svg)


def ranked(con):
    con.print("[bold green]$[/] alpha-score score examples/ --repeats 3")
    sc, t0 = Scorer(repeats=3), time.time()
    reports = sc.score_path(str(ROOT / "examples"))
    render(con, reports, sc.model_key, sc.cost(), time.time() - t0)


def detail(con):
    con.print("[bold green]$[/] alpha-score score examples/high_gefilte.txt")
    sc, t0 = Scorer(), time.time()
    reports = sc.score_path(str(ROOT / "examples" / "high_gefilte.txt"))
    render(con, reports, sc.model_key, sc.cost(), time.time() - t0, details=True)


def models(con):
    con.print("[bold green]$[/] alpha-score models")
    cmd_models(type("A", (), dict(serverless=False))(), con)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    shot("cli-ranking", "alpha-score score examples/ --repeats 3", ranked)
    if "--ranking-only" in __import__("sys").argv:
        raise SystemExit
    shot("cli-detail", "alpha-score score examples/high_gefilte.txt", detail)
    shot("cli-models", "alpha-score models", models, width=150)
