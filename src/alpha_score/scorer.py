"""High-level API: score texts and turn the raw statistics into a readable report.

    from alpha_score import Scorer
    report = Scorer().score_text(open("essay.txt").read(), title="essay")
    report["alpha"], report["verdict"], report["percentiles"]
"""

import json
import math
from contextlib import contextmanager
from importlib import resources

from .client import Client, rows_from_path, rows_from_text, score_rows
from .config import load_config
from .core import doc_score
from .fireworks import INFERENCE, Fireworks
from .models import MODELS, resolve

CAVEAT = ("alpha measures how much a text departs from what the reference model would have written, inside otherwise "
          "fluent prose. It does not check truth: conspiracy and crank writing scores close to genuine insight.")
CLASS_NAMES = dict(high="high-alpha essays", claude="Claude essays", wiki="Wikipedia articles", crank="crank posts")


def _calibration():
    try:
        return json.loads(resources.files("alpha_score").joinpath("calibration.json").read_text())
    except FileNotFoundError:
        return {}


CALIBRATION = _calibration()


class Scorer:
    def __init__(self, config=None, **overrides):
        self.cfg = config or load_config(**overrides)
        self.entry = resolve(self.cfg.model)
        self._client = None

    @property
    def model_key(self):
        return self.entry["key"]

    def client(self, log=print):
        if self._client:
            return self._client
        cfg, e = self.cfg, self.entry
        key = cfg.api_key
        if cfg.provider == "openai":
            if not cfg.base_url:
                raise ValueError("provider 'openai' needs base_url (config file, ALPHA_SCORE_BASE_URL or --base-url)")
            self._client = Client(cfg.base_url, e["hf"] if e["key"] in MODELS and e["hf"] else cfg.model, key)
        elif cfg.provider == "fireworks":
            if not key:
                raise ValueError(f"set ${cfg.api_key_env} to your Fireworks API key (fireworks.ai/account/api-keys)")
            if not e["fireworks"]:
                raise ValueError(f"{cfg.model} is not in Fireworks' library; upload it with firectl and pass its id")
            if cfg.deployment:
                if not cfg.account_id:
                    raise ValueError("a Fireworks deployment needs FIREWORKS_ACCOUNT_ID (or account_id in the config)")
                self._client = Fireworks(key, cfg.account_id).client(cfg.deployment, e["fireworks"])
            elif e["serverless"] or e["key"] not in MODELS:  # raw ids are assumed to be serverless
                self._client = Client(INFERENCE, e["fireworks"], key)
            else:
                raise ValueError(f"{cfg.model} is a {e['kind']} model Fireworks only serves on a dedicated deployment: "
                                 f"run `alpha-score deploy --model {cfg.model}` (~$8/GPU-hour) and pass --deployment, "
                                 f"or use the pay-per-token default (glm-5.3-flash)")
        else:
            raise ValueError(f"unknown provider {cfg.provider!r} (use 'fireworks' or 'openai')")
        return self._client

    @contextmanager
    def temporary_deployment(self, log=print, accelerator=None, count=None, precision=None):
        """Create a dedicated Fireworks deployment of the model, score through it, and always delete it."""
        cfg = self.cfg
        if not (cfg.api_key and cfg.account_id):
            raise ValueError(f"a Fireworks deployment needs ${cfg.api_key_env} and FIREWORKS_ACCOUNT_ID")
        if not self.entry["fireworks"]:
            raise ValueError(f"{cfg.model} is not in Fireworks' library")
        fw, base = Fireworks(cfg.api_key, cfg.account_id), self.entry["fireworks"]
        dep = fw.create(base, accelerator, count, precision)
        log(f"created deployment {dep} of {base} (billed per GPU-second until deleted)")
        try:
            d = fw.wait(dep, log=log)
            log(f"deployment READY: {d.get('acceleratorCount')}x {d.get('acceleratorType')}")
            self._client = fw.client(dep, base)
            log(f"model answering after {self._client.wait_ready(log=log):.0f}s more")
            yield dep
        finally:
            fw.delete(dep)
            log(f"deleted deployment {dep}")

    def score_rows(self, rows):
        return score_rows(self.client(), rows, self.cfg.workers, self.cfg.repeats)

    def score_text(self, text, title="text"):
        rows = rows_from_text(text, title, self.cfg.max_passages)
        if not rows:
            raise ValueError("text too short: alpha needs at least ~40 words (one passage); ~300+ words is better")
        return self.reports(self.score_rows(rows))[0]

    def score_texts(self, texts):
        """{title: text} -> list of reports, highest alpha first."""
        rows = [r for t, x in texts.items() for r in rows_from_text(x, t, self.cfg.max_passages)]
        if not rows:
            raise ValueError("texts too short: alpha needs at least ~40 words per text")
        return self.reports(self.score_rows(rows))

    def score_path(self, path):
        rows = rows_from_path(path, self.cfg.max_passages)
        if not rows:
            raise ValueError(f"no passages in {path} (documents need at least ~40 words)")
        return self.reports(self.score_rows(rows))

    def reports(self, scored_rows):
        by_doc = {}
        for r in scored_rows:
            by_doc.setdefault(r["doc"], []).append(r)
        out = [report(doc, ps, self.model_key) for doc, ps in by_doc.items()]
        out.sort(key=lambda x: -(x["alpha"] if x["alpha"] is not None else -math.inf))
        return out

    def cost(self):
        """Tokens used so far by this scorer and the USD cost for pay-per-token models with a known price."""
        u = dict(self._client.usage) if self._client else dict(prompt_tokens=0, completion_tokens=0, requests=0)
        p = self.entry["price"]
        u["usd"] = round((u["prompt_tokens"] * p[0] + u["completion_tokens"] * p[1]) / 1e6, 4) if p else None
        return u


def percentile(x, xs):
    return round(100 * sum(v < x for v in xs) / len(xs)) if xs else None


def verdict(alpha, cal):
    if alpha is None:
        return "noise"
    if not cal:
        return "uncalibrated"
    high = sorted(cal.get("high", []))
    claude = sorted(cal.get("claude", []))
    if high and alpha >= high[len(high) // 2]:
        return "high"
    if claude and alpha >= claude[int(0.75 * (len(claude) - 1))]:
        return "medium"
    return "low"


def surprising_spans(passages, n=5, window=10):
    """The n tokens the reference model found most surprising, each with a little surrounding text."""
    cands = []
    for p in passages:
        toks, nll = p.get("tokens"), p.get("tok_nll")
        if not toks or not nll:
            continue
        for i, v in enumerate(nll):
            if toks[i].strip() and len(toks[i].strip()) > 1:
                cands.append((v, p["idx"], i, toks))
    cands.sort(key=lambda c: -c[0])
    out, used = [], set()
    for v, idx, i, toks in cands:
        if any(idx == u and abs(i - j) < window for u, j in used):
            continue
        used.add((idx, i))
        out.append(dict(token=toks[i].strip(), nll=round(v, 2),
                        context="".join(toks[max(0, i - window):i]).lstrip() + "[[" + toks[i] + "]]" + "".join(toks[i + 1:i + 4])))
        if len(out) == n:
            break
    return out


def report(doc, passages, model_key):
    d = doc_score(passages)
    cal = CALIBRATION.get(model_key)
    alpha = d["alpha"]
    pct = {c: percentile(alpha, cal[c]) for c in ("claude", "wiki", "high", "crank") if c in cal} if cal and alpha is not None else {}
    ranked = sorted(passages, key=lambda p: -p["uzzi"])
    snip = lambda p: dict(idx=p["idx"], uzzi=round(p["uzzi"], 2), text=p["text"][:240] + ("…" if len(p["text"]) > 240 else ""))
    return dict(
        title=doc, alpha=None if alpha is None else round(alpha, 3), verdict=verdict(alpha, cal),
        gated=d["gated"], gate_reason=d["gate_reason"] or None,
        percentiles=pct, model=model_key,
        components=dict(conv=round(d["conv"], 3), tail=round(d["tail"], 3), coherence=round(d["coherence"], 3)),
        n_passages=d["n_passages"],
        top_passages=[snip(p) for p in ranked[:3]], bottom_passage=snip(ranked[-1]) if len(ranked) > 3 else None,
        surprising_spans=surprising_spans(passages),
    )


def explain(r):
    """One readable paragraph for a report."""
    if r["gated"]:
        return (f"GATED as noise ({r['gate_reason']}). The text does not read as coherent prose to the reference model, "
                f"so it gets no alpha score.")
    s = f"alpha {r['alpha']:+.2f} ({r['verdict']})"
    if r["percentiles"]:
        s += " — higher than " + ", ".join(f"{v}% of {CLASS_NAMES[c]}" for c, v in r["percentiles"].items()) + \
             f" in the benchmark (scored with {r['model']})"
    else:
        s += f" — no benchmark calibration for {r['model']}: compare only with other texts scored by the same model"
    return s + "."

