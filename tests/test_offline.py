"""Offline tests (no API key, no cost): core math, config precedence, the scoring client and the Fireworks
deployment lifecycle against a mock OpenAI-compatible / Fireworks server.

    pytest -q
"""

import json
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from alpha_score import cli, client, config, core, fireworks, scorer

ROOT = Path(__file__).resolve().parent.parent
TEXT = (ROOT / "examples" / "high_gefilte.txt").read_text()


# ---------------------------------------------------------------- core

def test_chunk_sizes():
    chunks = core.chunk(TEXT)
    assert 5 <= len(chunks) <= core.MAX_CHUNKS
    assert all(len(c.split()) >= 40 for c in chunks)
    assert len(core.chunk(TEXT, max_chunks=2)) == 2


def test_passage_stats_and_gate():
    s = core.passage_stats([1, 2, 3, 10], [[1, 1, 2, 2], [1, 2, 2, 3]], nll_full=2.0, nll_shuf=6.0)
    assert s["conv"] == s["self_med"] - s["med"] and s["uzzi"] == s["conv"] + s["tail"] and s["coherence"] == 4.0
    ok = core.doc_score([dict(s, uzzi=1.0, conv=0.0, tail=1.0, coherence=4.0)])
    assert ok["alpha"] == 1.0 and not ok["gated"]
    noise = core.doc_score([dict(s, uzzi=3.0, conv=-4.0, tail=7.0, coherence=0.2)])
    assert noise["alpha"] is None and "coherence" in noise["gate_reason"]
    assert not core.doc_score([dict(s, uzzi=1.0, conv=-4.0, tail=5.0, coherence=3.0)])["gated"]  # conv never gates


def test_shuffle_is_reproducible():
    assert core.shuffled_copies(["a b c d e f"]) == core.shuffled_copies(["a b c d e f"])


# ---------------------------------------------------------------- config

def test_config_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for v in list(config.ENV.values()) + ["ALPHA_SCORE_CONFIG"]:
        monkeypatch.delenv(v, raising=False)
    assert config.load_config().model == "glm-5.3-flash"
    (tmp_path / "alpha-score.toml").write_text('model = "qwen2.5-72b"\nworkers = 4\n')
    c = config.load_config()
    assert (c.model, c.workers, c.source) == ("qwen2.5-72b", 4, "alpha-score.toml")
    monkeypatch.setenv("ALPHA_SCORE_MODEL", "gpt-oss-120b")
    assert config.load_config().model == "gpt-oss-120b"
    assert config.load_config(model="x").model == "x"
    (tmp_path / "alpha-score.toml").write_text('api_key = "secret"\n')
    with pytest.raises(ValueError, match="api_key_env"):
        config.load_config()


def test_describe_masks_key(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_abcdefghijklmnop")
    assert "abcdefghij" not in config.load_config().describe()["api_key"]


# ---------------------------------------------------------------- mock server

class Mock(BaseHTTPRequestHandler):
    log, state = [], {}

    def log_message(self, *a):
        pass

    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if "/deployments" in self.path:
            Mock.log.append(("create", self.path, b))
            return self.send(200, {"state": "CREATING"})
        if not self.path.endswith("/completions"):
            return self.send(404, {})
        if "deployments/" in b["model"]:
            if "#" not in b["model"]:
                return self.send(404, {"error": "model not found"})  # forces the fallback model string
            Mock.state["warm"] = Mock.state.get("warm", 0) + 1
            if Mock.state["warm"] <= 2:
                return self.send(503, {"error": "scaling up"})
        usage = {"prompt_tokens": 10, "completion_tokens": b.get("max_tokens", 1)}
        text = b["prompt"]
        if b.get("echo"):
            toks = re.findall(r"\s*\S+", text) + [" gen"]
            offs, pos = [], 0
            for t in toks:
                offs.append(pos)
                pos += len(t)
            lps = [None] + [-random.uniform(0.1, 6) for _ in toks[1:]]
            return self.send(200, {"usage": usage, "choices": [{"text": text + " gen", "logprobs": {
                "tokens": toks, "token_logprobs": lps, "text_offset": offs}}]})
        words = ["the", "a", "cat", "idea", "sat"]
        return self.send(200, {"usage": usage, "choices": [
            {"text": " " + " ".join(random.choice(words) for _ in range(b["max_tokens"]))} for _ in range(b.get("n", 1))]})

    def do_GET(self):
        Mock.state["polls"] = Mock.state.get("polls", 0) + 1
        Mock.log.append(("get", self.path))
        self.send(200, {"state": "READY" if Mock.state["polls"] > 1 else "CREATING",
                        "acceleratorType": "NVIDIA_H100_80GB", "acceleratorCount": 4})

    def do_DELETE(self):
        Mock.log.append(("delete", self.path))
        self.send(200, {})


@pytest.fixture
def mock(monkeypatch):
    Mock.log.clear()
    Mock.state.clear()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    monkeypatch.setattr(fireworks, "API", f"{base}/v1")
    monkeypatch.setattr(fireworks, "INFERENCE", f"{base}/inference/v1")
    monkeypatch.setattr(scorer, "INFERENCE", f"{base}/inference/v1")
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test")
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "acct")
    monkeypatch.delenv("ALPHA_SCORE_MODEL", raising=False)
    yield base
    srv.shutdown()


def test_serverless_report(mock):
    sc = scorer.Scorer(workers=4, max_passages=3)
    r = sc.score_text(TEXT, "gefilte")
    assert r["n_passages"] == 3 and r["model"] == "glm-5.3-flash"
    # random mock logprobs usually look like noise to the gate; real calibration is checked separately below
    assert r["verdict"] in {"high", "medium", "low", "noise"} and r["surprising_spans"]
    assert (r["alpha"] is None) == r["gated"] and (set(r["percentiles"]) == {"claude", "wiki", "high", "crank"}) != r["gated"]
    assert "[[" in r["surprising_spans"][0]["context"]
    u = sc.cost()
    assert u["requests"] > 0 and u["usd"] > 0


def test_openai_provider_sends_hf_id(mock):
    sc = scorer.Scorer(provider="openai", base_url=f"{mock}/inference/v1", model="qwen2.5-7b")
    assert sc.client().model == "Qwen/Qwen2.5-7B"


def test_base_model_without_deployment_is_refused(mock):
    with pytest.raises(ValueError, match="dedicated deployment"):
        scorer.Scorer(model="qwen2.5-72b").client()


def test_deploy_score_delete_lifecycle(mock, capsys):
    cli.main(["score", str(ROOT / "examples" / "high_gefilte.txt"), "--model", "qwen2.5-72b", "--deploy",
              "--workers", "4", "--max-passages", "2", "--json"])
    rep = json.loads(capsys.readouterr().out)["reports"][0]
    create = next(l for l in Mock.log if l[0] == "create")
    assert create[2]["baseModel"] == "accounts/fireworks/models/qwen2p5-72b"
    assert create[2]["autoscalingPolicy"]["scaleToZeroWindow"] == "300s"
    assert Mock.log[-1][0] == "delete" and "ignoreChecks=true" in Mock.log[-1][1]
    assert rep["n_passages"] == 2 and rep["model"] == "qwen2.5-72b"


def test_deployment_deleted_on_failure(mock, monkeypatch):
    monkeypatch.setattr(client, "score_rows", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(scorer, "score_rows", client.score_rows)
    with pytest.raises(RuntimeError, match="boom"):
        cli.main(["score", str(ROOT / "examples" / "high_gefilte.txt"), "--model", "qwen2.5-72b", "--deploy"])
    assert Mock.log[-1][0] == "delete"


def test_verdict_and_percentiles_from_calibration():
    cal = scorer.CALIBRATION["glm-5.3-flash"]
    assert scorer.verdict(None, cal) == "noise"
    assert scorer.verdict(max(cal["high"]), cal) == "high"
    assert scorer.verdict(min(cal["wiki"]) - 1, cal) == "low"
    assert scorer.verdict(1.0, None) == "uncalibrated"
    assert scorer.percentile(10.0, cal["claude"]) == 100 and scorer.percentile(-10.0, cal["claude"]) == 0


def test_calibration_covers_benchmarked_models():
    for k in ("glm-5.3-flash", "qwen2.5-72b", "qwen2.5-72b-instruct"):
        assert set(scorer.CALIBRATION[k]) >= {"high", "claude", "wiki", "crank"}


def test_repeats_average_self_samples(mock):
    sc = scorer.Scorer(workers=4, max_passages=2, repeats=3)
    rows = sc.score_rows(client.rows_from_text(TEXT, "g", 2))
    assert len(rows) == 2 and all(r["repeats"] == 3 for r in rows)
    assert all(abs(r["uzzi"] - (r["conv"] + r["tail"])) < 1e-9 for r in rows)
    assert sc.cost()["requests"] >= 3 * 2 * 9
