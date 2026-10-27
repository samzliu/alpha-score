# Verification

These checks were run while the code was a set of scripts. In this package, `api_backend.py` is
`src/alpha_score/client.py`, `fireworks_backend.py` is the `alpha-score` CLI plus `src/alpha_score/fireworks.py`,
`modal_backend.py` is `scripts/modal_backend.py`, and `alpha_score.py` is `src/alpha_score/core.py`. The scoring math
was copied unchanged. The package's own tests are `tests/test_offline.py` (which includes the section-4 lifecycle
mock) and `tests/mcp_smoke.py`. `../value_vllm.py` and `../text_alpha_vllm.py` are the original research scripts and
are not part of this repo.

## 1. `modal_backend.py` reproduces the research pipeline

The packaged backend was run on all 608 benchmark passages with Qwen2.5-7B base (`eval/data/qwen7b_main.jsonl`) and compared with the original research scripts' output (`../value_vllm.py`, `../text_alpha_vllm.py`), using the same seeds:

| stat | agreement |
|---|---|
| target token NLLs | identical on 608/608 passages (max diff 0.000) |
| med, p90 | r = 1.0000, max diff 0.0000 |
| self_med / self_p90 | r = 1.0000 / 0.9995 (GPU batch nondeterminism in sampling) |
| coherence | r = 1.0000, max diff 0.004 |

## 2. `api_backend.py` matches `modal_backend.py` over an OpenAI-compatible API

`verify/verify_api.py` starts `vllm serve Qwen/Qwen2.5-7B` (the OpenAI-compatible server) on Modal and runs `api_backend.py` against it, on 100 passages (the first 2 of every benchmark document). Raw output: `api_vs_native.jsonl` (not included in the public copy, because it contains passage text).

`probe` against the server: echo + prompt logprobs OK, n=6 OK, top_k OK, seed OK, raw continuation (no chat template).

| stat | pearson r, per passage | mean native | mean api |
|---|---|---|---|
| med | 1.000 | 2.044 | 2.045 |
| p90 | 1.000 | 6.427 | 6.429 |
| coherence | 0.997 | 3.527 | 3.533 |
| self_med | 0.758 | 1.650 | 1.612 |
| self_p90 | 0.674 | 4.735 | 4.687 |
| tail | 0.981 | 1.692 | 1.742 |
| uzzi | 0.948 | 1.297 | 1.309 |

- **Target side:** exact up to tokenization (93/100 passages bit-identical).
- **Self-sample side:** differs only because the API draws different random samples. That is the sampling noise of K=6 continuations, not a bug.
- **Document level:** with just 2 passages per document, uzzi agrees at r = 0.95 (mean |diff| 0.17). Real documents have 4–20 passages, so the agreement there is tighter.

## 3. OpenRouter cannot be used (tested 2026-10-22)

`/api/v1/completions` was called with `echo: true, logprobs: 1, prompt_logprobs: 0`, with and without `provider.require_parameters`. The 12 models tried were llama-3.3-70b-instruct, qwen3-235b, qwen3.5-397b, qwen3-32b, deepseek-v3.2, deepseek-v4-flash, deepseek-chat-v3.1, glm-5.3, kimi-k3, hermes-4-405b, mistral-nemo and gpt-3.5-turbo-instruct.

- **No log-probs:** every response came back without any `logprobs`. Several providers also applied a chat template: the "completion" was an assistant reply.
- **No base models:** OpenRouter's model list (468 models) contains no base models.
- **Switching to instruct models doesn't help either.** In the catalog, `qwen/qwen-2.5-72b-instruct` and `qwen/qwen-2.5-7b-instruct` don't list `logprobs` as a supported parameter. Models that do list it, such as llama-3.3-70b-instruct, return logprobs only for generated chat tokens; echo on /completions still returns nothing. Scoring with output logprobs alone would take one call per token, and a target token outside the top 20 gets no score at all. Those are exactly the high-surprise tokens the tail statistic depends on.

## 4. `fireworks_backend.py` lifecycle (offline mock)

`python verify/test_fireworks_mock.py` runs the real script against a local mock of Fireworks' control plane and
completions API. It checks:
- the create body (`baseModel`, `deploymentShape: default`, min 0 / max 1 replicas, 5-minute scale-to-zero safety net);
- polling CREATING → READY;
- the fallback between the two documented model-string forms (`accounts/<acct>/deployments/<id>` and `<base>#accounts/<acct>/deployments/<id>`);
- retrying 503s while the replica scales up from zero;
- scoring in Fireworks' legacy logprobs format (`tokens`, `token_logprobs`, `text_offset`);
- DELETE with `ignoreChecks=true` at the end.

Result: PASS. Not yet run against the real Fireworks API, because no key was available in this environment.

## 5. Fireworks pay-per-token, live (2026-10-22)

`probe --serverless` against Fireworks' serverless `/inference/v1/completions`:

| model | echo + prompt logprobs | raw text (no chat template) | n=6 | top_k / seed |
|---|---|---|---|---|
| glm-5p3-flash | OK | OK | OK | OK |
| gpt-oss-120b | OK | OK | OK | OK |
| nemotron-lightning-3p5-30b-a3b | OK | OK | 400 (falls back to 6 calls) | OK |
| deepseek-v4p1-flash | **400** when `echo` and `logprobs` are combined (each alone works) | | | |

Fireworks returns logprobs for all prompt tokens except the first, with `max_tokens=0` or `1`. Qwen2.5-72B-Instruct and
Llama-3.3-70B-Instruct are listed as "Serverless: Not supported" (dedicated deployment only).

Full benchmark (608 passages) and 2x2 voice test (361) on GLM-5.3-Flash, and the benchmark on gpt-oss-120b: see
`eval/RESULTS.md`. GLM-5.3-Flash: AUROC 0.92, gate 4/4 and 0/46, voice test passes. gpt-oss-120b: AUROC 0.49, gate
wrongly rejects 12/46 real documents. At 16 concurrent workers, gpt-oss-120b hit sustained 429s on the second run;
the client now retries 10 times with backoff up to 30 s.
