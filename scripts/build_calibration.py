"""Rebuild src/alpha_score/calibration.json from the scored benchmark files in eval/data/.

    python scripts/build_calibration.py

For each reference model, stores the document-level alpha of every benchmark document by class, so a new score can be
placed relative to them ("higher than 94% of the Claude essays scored with this model"). Add a model by scoring
eval/data/chunks_main.jsonl with it (`alpha-score score eval/data/chunks_main.jsonl --model <key> --out ...`) and adding
the output file to FILES.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from alpha_score.core import score_docs  # noqa: E402

FILES = {
    "glm-5.3-flash": "glm53flash_main.jsonl",
    "qwen2.5-72b": "qwen72b_main.jsonl",
    "qwen2.5-72b-instruct": "qwen72b_instruct_main.jsonl",
    "qwen2.5-7b": "qwen7b_main.jsonl",
    "qwen2.5-7b-instruct": "qwen7b_instruct_main.jsonl",
}

out = {}
for key, f in FILES.items():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "eval", "data", f))]
    src = {r["doc"]: r.get("source") or r["doc"].split("_")[0] for r in rows}
    classes = {}
    for d, s in score_docs(rows).items():
        if s["alpha"] is not None:
            classes.setdefault(src[d], []).append(round(s["alpha"], 3))
    out[key] = {c: sorted(v) for c, v in sorted(classes.items())}
path = os.path.join(ROOT, "src", "alpha_score", "calibration.json")
json.dump(out, open(path, "w"), indent=1)
print(f"wrote {path}: " + ", ".join(f"{k} ({sum(map(len, v.values()))} docs)" for k, v in out.items()))
