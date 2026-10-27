"""Copy this repo to a new folder that is safe to publish (see DATA.md).

    python scripts/make_public_copy.py ../alpha-score-public                 # strip third-party text
    python scripts/make_public_copy.py ../alpha-score-public --anonymize     # also hide crank authors' names

What it does:
  - eval/data/*.jsonl: drops `text`, `tokens` and `sample0` (third-party prose and anything reconstructable from it).
    Scores, per-token NLLs and labels stay, so eval/evaluate.py and scripts/build_calibration.py still run.
  - examples/: keeps only the texts we can redistribute (Claude-written, Wikipedia (CC BY-SA, attributed in DATA.md),
    generated gibberish). Third-party posts are listed in DATA.md with links instead.
  - --anonymize: renames the crank documents, whose ids name their authors (crank_<author>_<topic>), to crank_01..06
    in eval data and eval/RESULTS.md.
  - skips .venv, caches, .git and local config.
The screenshots in docs/ quote short snippets of examples/high_gefilte.txt; review them yourself.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_EXAMPLES = {"claude_gefilte.txt", "wiki_gefilte.txt", "gib_markov.txt"}
SKIP = shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", ".git", "*.egg-info", "alpha-score.toml", ".env")
DROP = ("text", "tokens", "sample0")


def anon_map():
    """crank_<author>_<topic> -> crank_NN (the only document ids that name a person)."""
    docs = sorted({json.loads(l)["doc"] for f in ("chunks_main.jsonl", "chunks_restyle.jsonl")
                   for l in open(ROOT / "eval" / "data" / f)})
    cranks = [d for d in docs if re.match(r"^crank_[a-z]+_\w+$", d)]
    return {d: f"crank_{i:02d}" for i, d in enumerate(cranks, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--anonymize", action="store_true")
    a = ap.parse_args()
    dest = Path(a.dest).resolve()
    if dest.exists():
        raise SystemExit(f"{dest} exists; pick a new folder")
    shutil.copytree(ROOT, dest, ignore=SKIP)
    for f in (dest / "examples").iterdir():
        if f.name not in PUBLIC_EXAMPLES:
            f.unlink()
    rename = anon_map() if a.anonymize else {}
    for f in (dest / "eval" / "data").glob("*.jsonl"):
        rows = [json.loads(l) for l in open(f)]
        with open(f, "w") as out:
            for r in rows:
                for k in DROP:
                    r.pop(k, None)
                r["doc"] = rename.get(r["doc"], r["doc"])
                out.write(json.dumps(r) + "\n")
    if rename:
        for f in [dest / "eval" / "RESULTS.md"]:
            s = f.read_text()
            for old in sorted(rename, key=len, reverse=True):
                s = s.replace(old, rename[old])
            f.write_text(s)
    (dest / "verify" / "api_vs_native.jsonl").unlink(missing_ok=True)  # contains passage text
    print(f"wrote {dest}: {len(PUBLIC_EXAMPLES)} examples kept, text stripped from eval/data"
          + (f", {len(rename)} document ids anonymized" if rename else ""))
    print("Before publishing: review docs/screenshots/, README examples, and DATA.md.")


if __name__ == "__main__":
    main()
