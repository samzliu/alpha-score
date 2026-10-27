"""Score texts with an open-weights reference model served by vLLM on your own Modal GPUs (pip install modal; the
backend used for the original Qwen benchmarks). Needs the package installed locally (pip install -e .).

    modal run scripts/modal_backend.py --path examples/                       # every .txt in a folder (one doc per file)
    modal run scripts/modal_backend.py --path essay.txt --out essay.jsonl
    modal run scripts/modal_backend.py --chunks passages.jsonl                # pre-chunked: {"doc", "idx", "text", ...} per line
    modal run scripts/modal_backend.py --path examples/ --model qwen2.5-7b      # any key from models.py (GPU picked for you)
    modal run scripts/modal_backend.py --path examples/ --model Qwen/Qwen2.5-32B --gpu H100:2   # or any ungated HF model id

Writes one JSON line per passage (stats from alpha_score.passage_stats) and prints a document table.
Weights are cached in the Modal volume below; any ungated Hugging Face base model works.
"""

import glob
import json
import os

import modal

app = modal.App("alpha-score")
hf_cache = modal.Volume.from_name("llm-surprise-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.10.1.1", "transformers==4.55.4", "hf_transfer")  # vllm 0.10.1 breaks on newer transformers
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
)

PREFIX_TOKENS, MAX_TOKENS, K = 32, 256, 6  # keep in sync with src/alpha_score/core.py (not importable in the container)


@app.function(image=image, gpu="H100:4", volumes={"/cache": hf_cache}, timeout=4 * 3600)
def token_nlls_remote(texts, shuffled, model, n_gpu):
    """For each passage: target-token NLLs, K sampled continuations' NLLs, mean NLL of passage and shuffled copy."""
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.py", "*.model", "*.tiktoken", "*.txt"])
    hf_cache.commit()
    llm = LLM(model, tensor_parallel_size=n_gpu, max_model_len=2048, gpu_memory_utilization=0.9,
              trust_remote_code=True, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    bos = [tok.bos_token_id] if tok.bos_token_id is not None and "qwen" not in model.lower() else []
    sep = bos or tok.encode("\n", add_special_tokens=False)[:1]
    enc = lambda t: tok.encode(t, add_special_tokens=False)[:MAX_TOKENS]

    def token_nlls(seqs, n_scored):
        outs = llm.generate([TokensPrompt(prompt_token_ids=s) for s in seqs],
                            SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0), use_tqdm=False)
        return [[-d[t].logprob for d, t in zip(o.prompt_logprobs[len(s) - n :], s[len(s) - n :])]
                for o, s, n in zip(outs, seqs, n_scored)]

    ids = [enc(t) for t in texts]
    prefixes = [bos + x[:PREFIX_TOKENS] for x in ids]
    n_tgt = [len(x) - PREFIX_TOKENS for x in ids]
    target = token_nlls([p + x[PREFIX_TOKENS:] for p, x in zip(prefixes, ids)], n_tgt)
    gens = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prefixes],
                        [SamplingParams(n=K, temperature=1.0, top_k=50, max_tokens=n, min_tokens=n, seed=i)
                         for i, n in enumerate(n_tgt)], use_tqdm=False)
    # re-score samples exactly like targets: vLLM's sampler-reported logprobs disagree with prompt scoring
    sseqs = [p + list(c.token_ids) for p, g in zip(prefixes, gens) for c in g.outputs]
    samples = token_nlls(sseqs, [len(q) - len(prefixes[i // K]) for i, q in enumerate(sseqs)])
    mean = lambda v: sum(v) / len(v)
    full = [mean(v) for v in token_nlls([sep + x for x in ids], [len(x) for x in ids])]
    sh_ids = [enc(t) for t in shuffled]
    shuf = [mean(v) for v in token_nlls([sep + x for x in sh_ids], [len(x) for x in sh_ids])]
    return [dict(target=target[i], samples=samples[i * K : (i + 1) * K], nll_full=full[i], nll_shuf=shuf[i],
                 sample0=gens[i].outputs[0].text[:300]) for i in range(len(texts))]


def load_rows(path, chunks):
    from alpha_score.core import chunk

    if chunks:
        return [json.loads(l) for l in open(chunks)]
    files = sorted(glob.glob(os.path.join(path, "*.txt"))) if os.path.isdir(path) else [path]
    rows = []
    for f in files:
        doc = os.path.basename(f).rsplit(".", 1)[0]
        rows += [dict(doc=doc, idx=i, text=c) for i, c in enumerate(chunk(open(f).read()))]
    return rows


@app.local_entrypoint()
def main(path: str = "", chunks: str = "", model: str = "qwen2.5-72b", gpu: str = "", out: str = "scores.jsonl"):
    from alpha_score.core import passage_stats, print_table, score_docs, shuffled_copies
    from alpha_score.models import resolve

    m = resolve(model)
    if not m["hf"]:
        raise SystemExit(f"{model} has no open weights on Hugging Face; use fireworks_backend.py")
    model, gpu = m["hf"], gpu or m["modal_gpu"] or "H100:4"
    rows = load_rows(path, chunks)
    if not rows:
        raise SystemExit("no passages (documents need at least ~40 words)")
    print(f"{len(rows)} passages from {len({r['doc'] for r in rows})} documents -> {model} on {gpu}")
    texts = [r["text"] for r in rows]
    n_gpu = int(gpu.split(":")[1]) if ":" in gpu else 1
    raw = token_nlls_remote.with_options(gpu=gpu).remote(texts, shuffled_copies(texts), model, n_gpu)
    with open(out, "w") as f:
        for r, x in zip(rows, raw):
            r = dict(r, **passage_stats(x["target"], x["samples"], x["nll_full"], x["nll_shuf"]), model=model,
                     sample0=x["sample0"], tok_nll=[round(v, 3) for v in x["target"]])
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out}\n")
    print_table(score_docs(map(json.loads, open(out))))
