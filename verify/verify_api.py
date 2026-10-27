"""Check that the API client (alpha_score/client.py, any OpenAI-compatible endpoint) reproduces scripts/modal_backend.py.

Starts `vllm serve` (an OpenAI-compatible server, the same API surface Fireworks/Together expose) inside a Modal
container, runs client.score_passage against it on a subset of passages, and compares with the native scores.

    modal run verify/verify_api.py --native eval/data/qwen7b_main.jsonl     # from the alpha_score/ folder
"""

import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
app = modal.App("alpha-score-verify")
hf_cache = modal.Volume.from_name("llm-surprise-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.10.1.1", "transformers==4.55.4", "requests")
    .env({"HF_HOME": "/cache"})
    .add_local_python_source("alpha_score")  # the installed package (pip install -e .)
)


@app.function(image=image, gpu="H100", volumes={"/cache": hf_cache}, timeout=3600)
def run(rows, model):
    import subprocess
    import sys
    import time

    import requests

    from alpha_score.client import Client, probe, score_passage
    from alpha_score.core import shuffled_copies

    srv = subprocess.Popen(["vllm", "serve", model, "--port", "8000", "--max-model-len", "2048", "--max-logprobs", "20"])
    for _ in range(600):
        try:
            if requests.get("http://localhost:8000/health", timeout=2).ok:
                break
        except requests.ConnectionError:
            pass
        time.sleep(2)
    cl = Client("http://localhost:8000/v1", model)
    probe(cl)
    from concurrent.futures import ThreadPoolExecutor

    texts = [r["text"] for r in rows]
    shufs = shuffled_copies(texts)
    with ThreadPoolExecutor(32) as ex:
        res = list(ex.map(lambda i: score_passage(cl, texts[i], shufs[i], i), range(len(rows))))
    srv.terminate()
    return res


@app.local_entrypoint()
def main(native: str, model: str = "Qwen/Qwen2.5-7B", per_doc: int = 2):
    nat = [json.loads(l) for l in open(native)]
    sub, seen = [], {}
    for r in nat:  # first `per_doc` passages of every document
        if seen.get(r["doc"], 0) < per_doc:
            seen[r["doc"]] = seen.get(r["doc"], 0) + 1
            sub.append(r)
    print(f"{len(sub)} passages")
    res = run.remote([dict(text=r["text"]) for r in sub], model)
    # NB: shuffled copies here come from a fresh RNG over the subset, so coherence is compared loosely
    out = os.path.join(HERE, "api_vs_native.jsonl")
    with open(out, "w") as f:
        for r, a in zip(sub, res):
            f.write(json.dumps(dict(doc=r["doc"], idx=r["idx"], source=r.get("source"),
                                    native={k: r[k] for k in ("med", "p90", "self_med", "self_p90", "uzzi", "tail", "conv", "coherence")},
                                    api={k: a[k] for k in ("med", "p90", "self_med", "self_p90", "uzzi", "tail", "conv", "coherence")},
                                    native_tok=r["tok_nll"], api_tok=a["tok_nll"])) + "\n")

    def corr(x, y):
        mx, my = sum(x) / len(x), sum(y) / len(y)
        sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
        return sxy / (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5

    rows = [json.loads(l) for l in open(out)]
    exact = sum(r["native_tok"] == r["api_tok"] for r in rows)
    maxdiff = max(max((abs(a - b) for a, b in zip(r["native_tok"], r["api_tok"])), default=0) for r in rows)
    print(f"\ntarget token NLLs identical length: {sum(len(r['native_tok']) == len(r['api_tok']) for r in rows)}/{len(rows)}, "
          f"exactly equal: {exact}/{len(rows)}, max |diff| = {maxdiff:.3f}")
    print(f"{'stat':<10}{'pearson r':>10}{'mean native':>13}{'mean api':>10}")
    for k in ("med", "p90", "self_med", "self_p90", "conv", "tail", "uzzi", "coherence"):
        x, y = [r["native"][k] for r in rows], [r["api"][k] for r in rows]
        print(f"{k:<10}{corr(x, y):>10.3f}{sum(x) / len(x):>13.3f}{sum(y) / len(y):>10.3f}")
    print(f"wrote {out}")
