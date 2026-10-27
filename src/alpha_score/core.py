"""Core of the alpha score: chunking, the Uzzi-style statistic, the coherence gate, document aggregation.

Pure Python, no dependencies. The backends (client.py for OpenAI-compatible APIs, scripts/modal_backend.py for vLLM on
Modal) only produce per-token negative log-likelihoods (NLLs) from a reference model; everything that turns those into
a score is here.

Per ~100-word passage, with reference model M:
  target tokens   = the passage minus its first PREFIX_TOKENS tokens, scored given that prefix
  self tokens     = K continuations M samples from the same prefix (temp 1, top_k 50), scored the same way
  med, p90        = median / 90th-percentile NLL of the target tokens
  self_med, p90   = the same over the pooled self tokens
  conv  = self_med - med     > 0: the passage's typical token is at least as predictable as M's own writing
  tail  = p90 - self_p90     > 0: its most surprising tokens are more surprising than M's own most surprising
  uzzi  = conv + tail        (Uzzi et al. 2013: conventional core + atypical tail)
  coherence = mean NLL(word-shuffled passage) - mean NLL(passage)   (~0 for word salad, 3-4 for prose)

Document score = mean over its passages. A document is GATED (alpha = None) if it looks like noise:
  coherence < COHERENCE_MIN   (coherence uses no sampling, so the gate gives the same answer on every run)
"""

import random
import re

PREFIX_TOKENS = 32
MAX_TOKENS = 256
K = 6
WORDS_PER_CHUNK = 100
MAX_CHUNKS = 20
# Coherence alone separates all gibberish from all real text on every model tested (real >= 2.89, gibberish <= 1.64).
# A conv backstop (conv < -2.5) was dropped: conv depends on the random self-samples, and on a 4-passage crank post it
# crossed -2.5 on one of two runs. Recheck with eval/evaluate.py when changing models.
COHERENCE_MIN = 2.5


def clean(text):
    text = re.sub(r"^#+\s*", "", text, flags=re.M)  # markdown headings
    text = re.sub(r"\*\*?|__", "", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.M)
    text = re.sub(r"^=+.*?=+\s*$", "", text, flags=re.M)  # wiki section headers
    return text


def chunk(text, max_chunks=MAX_CHUNKS):
    """Split a document into ~100-word passages on sentence boundaries (at most max_chunks)."""
    sents = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", clean(text)).strip())
    chunks, cur = [], []
    for s in sents:
        cur.append(s)
        if sum(len(x.split()) for x in cur) >= WORDS_PER_CHUNK:
            chunks.append(" ".join(cur))
            cur = []
    if cur and sum(len(x.split()) for x in cur) >= 40:
        chunks.append(" ".join(cur))
    return chunks[:max_chunks]


def shuffled_copies(texts, seed=0):
    """Word-shuffled copy of each passage (one shared RNG, in order, so runs are reproducible)."""
    rng = random.Random(seed)
    out = []
    for t in texts:
        w = t.split()
        rng.shuffle(w)
        out.append(" ".join(w))
    return out


def quantile(x, q):
    x = sorted(x)
    return x[min(len(x) - 1, int(q * len(x)))] if x else float("nan")


def passage_stats(target_nlls, sample_nlls, nll_full, nll_shuf):
    """target_nlls: per-token NLLs of the real continuation; sample_nlls: list of K such lists for M's samples;
    nll_full / nll_shuf: mean NLL of the whole passage / its word-shuffled copy."""
    pooled = [v for s in sample_nlls for v in s]
    med, p90 = quantile(target_nlls, 0.5), quantile(target_nlls, 0.9)
    self_med, self_p90 = quantile(pooled, 0.5), quantile(pooled, 0.9)
    conv, tail = self_med - med, p90 - self_p90
    return dict(med=med, p90=p90, self_med=self_med, self_p90=self_p90, conv=conv, tail=tail, uzzi=conv + tail,
                coherence=nll_shuf - nll_full, n_tokens=len(target_nlls))


def doc_score(passages):
    """Aggregate passage_stats dicts of one document into its score."""
    mean = lambda k: sum(p[k] for p in passages) / len(passages)
    d = {k: mean(k) for k in ("uzzi", "conv", "tail", "coherence")}
    reasons = []
    if d["coherence"] < COHERENCE_MIN:
        reasons.append(f"coherence {d['coherence']:.2f} < {COHERENCE_MIN}")
    d.update(n_passages=len(passages), gated=bool(reasons), gate_reason="; ".join(reasons),
             alpha=None if reasons else d["uzzi"])
    return d


def score_docs(rows):
    """rows: passage dicts with 'doc' plus passage_stats fields -> {doc: doc_score}."""
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r["doc"], []).append(r)
    return {d: doc_score(ps) for d, ps in by_doc.items()}


def print_table(docs):
    print(f"{'document':<42}{'alpha':>8}{'uzzi':>8}{'conv':>8}{'tail':>8}{'coh':>7}{'n':>4}  note")
    for d, s in sorted(docs.items(), key=lambda kv: -(kv[1]["alpha"] if kv[1]["alpha"] is not None else -1e9)):
        a = f"{s['alpha']:+.2f}" if s["alpha"] is not None else "GATED"
        print(f"{d:<42}{a:>8}{s['uzzi']:>+8.2f}{s['conv']:>+8.2f}{s['tail']:>+8.2f}{s['coherence']:>7.2f}"
              f"{s['n_passages']:>4}  {s['gate_reason']}")
