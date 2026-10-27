"""Evaluate alpha scores on the labeled benchmark and write a markdown report.

    python eval/evaluate.py --main eval/data/qwen72b_main.jsonl --restyle eval/data/qwen72b_restyle.jsonl --name "Qwen2.5-72B base"

--main     scorer output for eval/data/chunks_main.jsonl (608 passages, 50 documents; `source` gives the class)
--restyle  optional scorer output for eval/data/chunks_restyle.jsonl (the 2x2 voice/content test)
--md       append the report to this markdown file
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # works uninstalled too
from alpha_score.core import COHERENCE_MIN, score_docs  # noqa: E402

CLASSES = ["wiki", "claude", "crank", "gib"]
NEG_INF = -1e9


def auroc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def boot_ci(pos, neg, n=1000, seed=0):
    rng = random.Random(seed)
    xs = sorted(auroc([rng.choice(pos) for _ in pos], [rng.choice(neg) for _ in neg]) for _ in range(n))
    return xs[int(0.025 * n)], xs[int(0.975 * n)]


def load(path):
    rows = [json.loads(l) for l in open(path)]
    src = {r["doc"]: r.get("source") or r["doc"].split("_")[0] for r in rows}
    docs = score_docs(rows)
    for d, s in docs.items():
        s["source"] = src[d]
    return docs


CELLS = {"A": "high_{}", "B": "aivoice_{}", "C": "humanvoice_{}", "D": "claude_{}"}


def restyle_slugs(allv, rs):
    slugs = sorted(d[len("aivoice_"):] for d in rs if d.startswith("aivoice_"))
    return [s for s in slugs if all(t.format(s) in allv for t in CELLS.values())]


def two_by_two(allv, slugs, k):
    """Cell means, content and voice effects (pooled-SD units), and B > C count for score k."""
    v = {c: [allv[t.format(s)][k] for s in slugs] for c, t in CELLS.items()}
    m = {c: sum(x) / len(x) for c, x in v.items()}
    pool = [x for xs in v.values() for x in xs]
    mu = sum(pool) / len(pool)
    sd = (sum((x - mu) ** 2 for x in pool) / len(pool)) ** 0.5
    content = ((m["A"] + m["B"]) - (m["C"] + m["D"])) / 2 / sd
    voice = ((m["A"] + m["C"]) - (m["B"] + m["D"])) / 2 / sd
    return m, content, voice, sum(b > c for b, c in zip(v["B"], v["C"]))


def summary(main, restyle=""):
    """Headline numbers for one model: alpha AUROC per class, gate counts, 2x2 effects."""
    docs = load(main)
    a = lambda s: s["alpha"] if s["alpha"] is not None else NEG_INF
    hi = [a(s) for s in docs.values() if s["source"] == "high"]
    out = {c: auroc(hi, [a(s) for s in docs.values() if s["source"] == c]) for c in CLASSES}
    out["all"] = auroc(hi, [a(s) for s in docs.values() if s["source"] != "high"])
    out["gib_gated"] = sum(s["gated"] for s in docs.values() if s["source"] == "gib")
    out["real_gated"] = sum(s["gated"] for s in docs.values() if s["source"] != "gib")
    out["n_real"] = sum(s["source"] != "gib" for s in docs.values())
    if restyle:
        rs = load(restyle)
        _, out["content"], out["voice"], _ = two_by_two({**docs, **rs}, restyle_slugs({**docs, **rs}, rs), "uzzi")
    return out


def report(main, restyle, name):
    out = []
    p = out.append
    docs = load(main)
    val = lambda s, k: (s["alpha"] if s["alpha"] is not None else NEG_INF) if k == "alpha" else s[k]
    p(f"## {name}\n")
    p(f"{len(docs)} documents from `{os.path.basename(main)}`. Document score = mean over its ~100-word passages.\n")

    p("### Class means\n")
    p("| class | docs | alpha (ungated docs) | uzzi | conv | tail | coherence | gated |")
    p("|---|---|---|---|---|---|---|---|")
    for c in ["high"] + CLASSES:
        ds = [s for s in docs.values() if s["source"] == c]
        al = [s["alpha"] for s in ds if s["alpha"] is not None]
        m = lambda k: sum(s[k] for s in ds) / len(ds)
        p(f"| {c} | {len(ds)} | {sum(al) / len(al):+.2f} |" if al else f"| {c} | {len(ds)} | – |",)
        out[-1] += f" {m('uzzi'):+.2f} | {m('conv'):+.2f} | {m('tail'):+.2f} | {m('coherence'):.2f} | {sum(s['gated'] for s in ds)}/{len(ds)} |"

    p(f"\n### Gate (coherence < {COHERENCE_MIN})\n")
    real = [s for s in docs.values() if s["source"] != "gib"]
    gib = [s for s in docs.values() if s["source"] == "gib"]
    p(f"- gibberish caught: {sum(s['gated'] for s in gib)}/{len(gib)}; real documents wrongly gated: {sum(s['gated'] for s in real)}/{len(real)}")
    p(f"- margins: lowest real coherence {min(s['coherence'] for s in real):.2f} vs highest gibberish {max(s['coherence'] for s in gib):.2f}; "
      f"lowest real conv {min(s['conv'] for s in real):+.2f} vs highest gibberish {max(s['conv'] for s in gib):+.2f}")

    p("\n### AUROC: high-alpha documents vs each class (95% bootstrap CI)\n")
    p("Probability that a random high-alpha document outscores a random document of the other class. 0.5 = chance.\n")
    p("| score | " + " | ".join(CLASSES) + " | all |")
    p("|---|" + "---|" * (len(CLASSES) + 1))
    hi = [s for s in docs.values() if s["source"] == "high"]
    for k, label in [("alpha", "**alpha** (gated uzzi)"), ("uzzi", "uzzi, no gate"), ("tail", "tail only"), ("conv", "conv only")]:
        cells = []
        for c in CLASSES + ["all"]:
            neg = [val(s, k) for s in docs.values() if (s["source"] != "high" if c == "all" else s["source"] == c)]
            pos = [val(s, k) for s in hi]
            lo, up = boot_ci(pos, neg)
            cells.append(f"{auroc(pos, neg):.2f} ({lo:.2f}–{up:.2f})")
        p(f"| {label} | " + " | ".join(cells) + " |")

    atl = sorted(d[len("high_atl_"):] for d in docs if d.startswith("high_atl_") and f"claude_atl_{d[9:]}" in docs)
    if atl:
        wins = [s for s in atl if docs[f"high_atl_{s}"]["uzzi"] > docs[f"claude_atl_{s}"]["uzzi"]]
        p(f"\n### Same title, human vs Claude\n\nThe Atlantic essay outscores Claude's essay with the same title on "
          f"{len(wins)}/{len(atl)} titles. Losses: {', '.join(s for s in atl if s not in wins) or 'none'}.")

    p("\n### Ranking (top 15 and bottom 5 ungated documents)\n")
    p("| rank | document | class | alpha | conv | tail |")
    p("|---|---|---|---|---|---|")
    ranked = sorted([(d, s) for d, s in docs.items() if not s["gated"]], key=lambda x: -x[1]["alpha"])
    for i, (d, s) in enumerate(ranked):
        if i < 15 or i >= len(ranked) - 5:
            p(f"| {i + 1} | {d} | {s['source']} | {s['alpha']:+.2f} | {s['conv']:+.2f} | {s['tail']:+.2f} |")
        elif i == 15:
            p("| … | | | | | |")

    if restyle:
        rs = load(restyle)
        allv = {**docs, **rs}
        slugs = restyle_slugs(allv, rs)
        p(f"\n### Voice vs content (2x2, {len(slugs)} matched topics)\n")
        p("A = high-alpha original (human voice), B = A rewritten by Claude in its own voice keeping every idea, "
          "C = Claude's same-topic essay rewritten in the human author's voice, D = Claude's original. "
          "A score that tracks ideas gives A≈B > C≈D; an AI-text detector gives A≈C > B≈D. "
          "Effects are in pooled-SD units.\n")
        p("| score | A hi/human | B hi/AI | C lo/human | D lo/AI | content effect | voice effect | B > C |")
        p("|---|---|---|---|---|---|---|---|")
        for k in ["uzzi", "tail", "conv"]:
            m, content, voice, bc = two_by_two(allv, slugs, k)
            p(f"| {k} | " + " | ".join(f"{m[c]:+.2f}" for c in "ABCD") + f" | {content:+.2f} | {voice:+.2f} | {bc}/{len(slugs)} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--restyle", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    text = report(a.main, a.restyle, a.name or os.path.basename(a.main))
    print(text)
    if a.md:
        open(a.md, "a").write(text + "\n")
