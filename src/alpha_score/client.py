"""Get the per-token NLLs alpha_score needs from any OpenAI-compatible /v1/completions endpoint that returns
PROMPT-token logprobs (`echo: true` + `logprobs`).

Verified against `vllm serve` (identical target-side stats to the Modal backend, see verify/) and live against
Fireworks' pay-per-token API. Does NOT work with OpenRouter or OpenAI (no prompt logprobs; see README).
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from .core import K, MAX_TOKENS, PREFIX_TOKENS, chunk, passage_stats, shuffled_copies


class Client:
    def __init__(self, base_url, model, key=None, top_k=True, timeout=120, extra=None):
        self.url = base_url.rstrip("/") + "/completions"
        self.model, self.top_k, self.timeout = model, top_k, timeout
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.extra = extra or {}  # provider-specific request fields, e.g. {"raw_output": False}
        self.usage = dict(prompt_tokens=0, completion_tokens=0, requests=0)  # as reported by the provider
        self._lock = threading.Lock()

    def complete(self, prompt, retries=10, **params):  # 429s can last minutes on shared serverless pools
        for attempt in range(retries):
            try:
                r = requests.post(self.url, headers=self.headers, timeout=self.timeout,
                                  json=dict(model=self.model, prompt=prompt, **self.extra, **params))
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
                if r.status_code >= 400:
                    raise RuntimeError(f"{r.status_code}: {r.text[:300]}")  # bad request: retrying won't help
                out = r.json()
                u = out.get("usage") or {}
                with self._lock:
                    self.usage["requests"] += 1
                    for k in ("prompt_tokens", "completion_tokens"):
                        self.usage[k] += u.get(k) or 0
                return out
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout):
                if attempt == retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 30))

    def wait_ready(self, timeout_s=1800, log=print):
        """Block until the model answers (dedicated deployments can take many minutes to scale up from zero)."""
        t0 = time.time()
        while True:
            try:
                self.complete("Hello", retries=1, max_tokens=1)
                return time.time() - t0
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout, RuntimeError) as e:
                if time.time() - t0 > timeout_s:
                    raise TimeoutError(f"model not ready after {timeout_s}s: {e}")
                log(f"  waiting for model ({int(time.time() - t0)}s): {str(e)[:100]}")
                time.sleep(20)

    def echo(self, text):
        """[(token, char_offset, logprob)] for the prompt tokens of `text` (leading special tokens dropped)."""
        r = self.complete(text, max_tokens=1, echo=True, logprobs=1, temperature=0)
        ch = r["choices"][0]
        lp = ch.get("logprobs") or {}
        if not lp.get("tokens") and isinstance(r.get("prompt"), list) and r["prompt"]:
            lp = r["prompt"][0].get("logprobs") or {}  # Together returns prompt logprobs at the top level
        toks, lps, offs = lp.get("tokens"), lp.get("token_logprobs"), lp.get("text_offset")
        if not toks or lps is None:
            raise RuntimeError("endpoint returned no prompt logprobs for echo=True; this provider/model cannot be used")
        if offs is None:  # rebuild offsets from token strings
            offs, pos = [], 0
            for t in toks:
                offs.append(pos)
                pos += len(t)
        n_prompt = sum(o < len(text) for o in offs)  # drops the 1 generated token
        ent = list(zip(toks, offs, lps))[:n_prompt]
        while ent and not text.startswith(ent[0][0]) and ent[0][1] == 0 and len(ent) > 1 and ent[1][1] == 0:
            ent = ent[1:]  # BOS-like special token at offset 0
        return ent

    def sample(self, prompt, n_tokens, seed):
        params = dict(max_tokens=n_tokens, temperature=1.0, top_p=1.0, seed=seed)
        if self.top_k:
            params["top_k"] = 50  # matches modal_backend / HF generate default
        try:
            return [c["text"] for c in self.complete(prompt, n=K, **params)["choices"]]
        except (requests.HTTPError, RuntimeError):  # provider without n>1 (or n capped, e.g. DeepInfra n<=4)
            return [self.complete(prompt, **dict(params, seed=seed * K + j))["choices"][0]["text"] for j in range(K)]


def score_passage(cl, text, shuf, seed):
    """Per-passage stats (see core.passage_stats), plus the target tokens and their NLLs for highlighting."""
    ent = cl.echo(text)[:MAX_TOKENS]
    if len(ent) <= PREFIX_TOKENS + 1:
        raise ValueError("passage too short")
    target = [-lp for _, _, lp in ent[PREFIX_TOKENS:]]
    prefix = text[: ent[PREFIX_TOKENS][1]]
    samples, texts = [], cl.sample(prefix, len(target), seed)
    for s in texts:
        se = [e for e in cl.echo(prefix + s) if e[1] >= len(prefix) and e[2] is not None][: len(target)]
        if se:
            samples.append([-lp for _, _, lp in se])

    def mean_nll(t):
        e = [lp for _, o, lp in cl.echo("\n" + t)[:MAX_TOKENS + 1] if o >= 1 and lp is not None]
        return -sum(e) / len(e)

    return dict(passage_stats(target, samples, mean_nll(text), mean_nll(shuf)),
                sample0=texts[0][:300] if texts else "", tok_nll=[round(v, 3) for v in target],
                tokens=[t for t, _, _ in ent[PREFIX_TOKENS:]])


def score_rows(cl, rows, workers=16, repeats=1):
    """Score passage rows ({"doc", "idx", "text", ...}); returns the rows with stats merged in, in order.

    repeats > 1 draws fresh self-samples that many times and averages the passage stats. The real-text side is
    deterministic, so this only reduces the sampling noise of the baseline (cost scales with repeats)."""
    texts = [r["text"] for r in rows]
    shufs = shuffled_copies(texts)
    jobs = [(i, j) for j in range(repeats) for i in range(len(rows))]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda ij: score_passage(cl, texts[ij[0]], shufs[ij[0]], ij[0] + 100003 * ij[1]), jobs))
    out = []
    for i, r in enumerate(rows):
        runs = res[i::len(rows)]
        s = dict(runs[0])
        for k in AVERAGED:
            s[k] = sum(x[k] for x in runs) / len(runs)
        out.append(dict(r, **s, model=cl.model, repeats=repeats))
    return out


AVERAGED = ("self_med", "self_p90", "conv", "tail", "uzzi", "coherence")


def rows_from_text(text, doc="text", max_chunks=None):
    return [dict(doc=doc, idx=i, text=c) for i, c in enumerate(chunk(text, *([max_chunks] if max_chunks else [])))]


def rows_from_path(path, max_chunks=None):
    """A .txt/.md file, a folder of them (one document per file), or a pre-chunked .jsonl."""
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path)]
    if os.path.isdir(path):
        files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith((".txt", ".md")))
    else:
        files = [path]
    rows = []
    for f in files:
        rows += rows_from_text(open(f, encoding="utf-8").read(), os.path.basename(f).rsplit(".", 1)[0], max_chunks)
    return rows


def probe(cl, log=print):
    """Check that an endpoint can be used; returns True if prompt logprobs work."""
    text = "The committee met on Tuesday to discuss the budget. After a long debate, the members agreed to postpone the vote."
    try:
        ent = cl.echo(text)
        log(f"echo + prompt logprobs : OK   ({len(ent)} prompt tokens, first scored: {ent[1]})")
    except Exception as e:
        log(f"echo + prompt logprobs : FAIL ({e})")
        return False
    for name, params in [("n=6 samples", dict(n=K)), ("top_k", dict(top_k=50)), ("seed", dict(seed=1))]:
        try:
            r = cl.complete(text[:40], max_tokens=5, temperature=1.0, **params)
            log(f"{name:<23}: OK   ({len(r['choices'])} choice(s))")
        except Exception as e:  # n>1 falls back to K single calls; top_k / seed are nice-to-have
            log(f"{name:<23}: not supported ({str(e)[:80]}){' -> falls back to 6 calls' if name.startswith('n=') else ''}")
    t = cl.complete("Once upon a time", max_tokens=12, temperature=0)["choices"][0]["text"]
    log(f"raw continuation       : {t!r}  (a chat template would show up here as an assistant-style reply)")
    return True
