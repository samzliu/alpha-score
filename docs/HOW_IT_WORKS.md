# How alpha-score works

**Alpha** is content a strong language model would not have written itself, delivered in otherwise fluent prose.
The score uses only per-token log-probabilities from an open-weights reference model. There is no judge model, no
training and no labels: the reference model is never asked whether the text is good. Its own predictions are the
baseline.

## The statistic

The score adapts Uzzi et al. (2013, *Science*). They found that the most successful scientific papers combine a
**highly conventional core with an atypical tail**. Here the "combinations" are tokens, and "typical" means "what
the reference model itself would write".

The text is split into ~100-word passages on sentence boundaries, up to 20 per document. For each passage:

1. **Real tokens.** Split off the first 32 tokens as a **prefix**. Score the surprisal (negative log-likelihood, NLL)
   of every remaining real token given everything before it.
2. **Self baseline.** Let the model write **6 continuations of its own** from the same prefix, with temperature 1,
   top-k 50 and the same length. Score those tokens the same way. This shows how surprising the model's own writing
   is to itself.
3. **Compare the two distributions of per-token surprisal.**
   - `conv = self_median − median`: is the passage's *typical* word at least as predictable as the model's own?
     Fluent prose sits near 0, and noise is strongly negative.
   - `tail = p90 − self_p90`: are the passage's *most surprising* 10% of words more surprising than the model's own
     most surprising words? This is where specific claims, names, turns of argument and ideas the model didn't
     anticipate show up.
   - `uzzi = conv + tail`
4. **Gibberish gate.**
   - `coherence` is the mean NLL of a word-shuffled copy minus the mean NLL of the passage. Word salad scores about
     0, and prose scores 3–4.
   - A document is **gated** (it gets no score) if its mean coherence is below 2.5.
   - On every model tested, this separates every gibberish document from every real one: real text ≥ 2.89, gibberish
     ≤ 1.64.
   - Coherence uses no sampling, so the gate gives the same answer on every run.
5. **Document alpha** is the mean `uzzi` over the document's passages, unless the document is gated.

Each passage costs about 9 API calls: one echo of the passage, one call that samples 6 continuations, 6 echoes to
rescore the samples, and 2 coherence echoes. With GLM-5.3-Flash on Fireworks that is about $0.0006 per passage, or
about $0.01 for a 2,000-word essay.

### Why it needs prompt-token log-probabilities

Steps 1, 2 and 4 score text the model did *not* generate. That needs a completions endpoint that returns
log-probabilities for the **prompt** (`echo: true` + `logprobs`). As of October 2026:

| provider | prompt logprobs |
|---|---|
| Fireworks (serverless GLM-5.3-Flash, gpt-oss-120b, Nemotron; on-demand deployments) | **yes** (live-tested) |
| `vllm serve` (self-hosted) | **yes** (verified identical to native vLLM) |
| Together | documented, untested |
| OpenRouter | no: tested on 12 models, and no logprobs come back at all |
| OpenAI, Anthropic, Gemini, Groq, Cerebras, Novita | no: generated-token logprobs only, or none at all |

Scoring only generated tokens would take one call per token, and any token outside the top 20 would get no score.
Those are exactly the surprising tokens the tail statistic depends on.

## Reading a score

Scores are only comparable **within one reference model**. The CLI and MCP tools place each score among the
benchmark documents scored with the same model; the percentiles come from `src/alpha_score/calibration.json`.

| class (benchmark) | GLM-5.3-Flash | Qwen2.5-72B base |
|---|---|---|
| high-taste human essays (3 blog posts + 14 Atlantic essays published after the models' cutoff) | **+1.10** | **+1.58** |
| conspiracy / crank Substack posts | +0.85 | +1.30 |
| Claude Opus essays from a one-line prompt, matched titles | +0.15 | +1.07 |
| Wikipedia, matched topics | −0.57 | +0.62 |
| gibberish (random chars, word salad, Markov, shuffled essays) | gated 4/4 | gated 4/4 |

**Verdicts:**
- `high`: at or above the median high-alpha essay.
- `medium`: above 75% of Claude essays.
- `low`: everything else.
- `noise`: gated.

## It tracks ideas, not voice

The obvious worry is that this is an AI-text detector in disguise. To rule that out, Claude rewrote each of 16
high-alpha essays in its own voice, keeping every idea. It also rewrote its own same-topic essays in the human
author's voice. On GLM-5.3-Flash:

| | ideas: high-alpha | ideas: Claude's |
|---|---|---|
| human voice | +1.07 | +0.32 |
| Claude's voice | +1.18 | +0.16 |

The content effect is +1.48 SD and the voice effect is +0.03 SD. Changing who wrote the words barely moves the
score, and changing the ideas moves it a lot. Qwen2.5-72B base gives the same result (+1.38 / −0.15).

## Benchmark results

AUROC is the probability that a random high-alpha document outscores a random document of the other class. Full
tables are in [`eval/RESULTS.md`](../eval/RESULTS.md).

| reference model | vs Wikipedia | vs Claude | vs crank | vs gibberish | vs all |
|---|---|---|---|---|---|
| **GLM-5.3-Flash** (Fireworks, pay-per-token; default) | 1.00 | 0.96 | 0.67 | 1.00 | **0.92** |
| Qwen2.5-72B base (Modal / Fireworks deployment) | 0.99 | 0.87 | 0.66 | 1.00 | 0.87 |
| Qwen2.5-72B instruct | 0.86 | 0.93 | 0.65 | 1.00 | 0.87 |
| Qwen2.5-7B base | 0.94 | 0.72 | 0.63 | 1.00 | 0.78 |
| Qwen2.5-7B instruct | 0.78 | 0.76 | 0.53 | 1.00 | 0.75 |
| gpt-oss-120b (Fireworks, pay-per-token) | 0.63 | 0.27 | 0.75 | 0.82 | 0.49 (fails) |

Reproduce any row by scoring `eval/data/chunks_main.jsonl` and running `python eval/evaluate.py --main <scores>`.

## Limitations

- **It does not check truth.** Conspiracy posts are fluent and full of claims the model didn't expect, so they score
  high: AUROC vs. crank is about 0.67 on every model, from only 6 crank documents. Alpha measures *surprising
  content*, not *true* or *valuable* content. Nothing we tried as a value axis worked: P(True) from the model,
  influence on Wikipedia text, or within-document information gain.
- **The benchmark is small.** It has 50 documents, 17 of them high-alpha. The gate thresholds were set on this same
  data, though with a wide margin (lowest real coherence 2.89 vs. highest gibberish 1.64).
- **Runs are noisy.** A document's alpha has a run-to-run SD of about 0.26, because the self-samples are random.
  See [Run-to-run noise](#run-to-run-noise). Use `--repeats` when a decision hinges on a small difference.
- **Text the model has seen** in its pretraining data scores low however good it is. Score text written after the
  model's training cutoff, or accept that famous texts look "unsurprising".
- **Concrete specifics count as surprise.** Names, numbers, quotes and foreign-language snippets the model lacks
  raise `tail`. `surprising_spans` shows them, so you can tell an unusual word from an unusual idea.
- **Hosted models change.** A provider can update or retire a pay-per-token model, which shifts the scale. Record
  the model and date with your scores, and re-run `eval/` before comparing across months.

## Run-to-run noise

We scored the full benchmark twice with GLM-5.3-Flash (`eval/data/glm53flash_main.jsonl` and `…_rerun.jsonl`).
The real-text side (target NLLs and coherence) is deterministic. All of the noise comes from the 6 random
self-samples, and Fireworks does not make sampling reproducible from a seed.

| | single run | `--repeats 2` (average of both runs) |
|---|---|---|
| document alpha, run-to-run SD | **0.26** (correlation between runs r = 0.86) | ~0.18 (expected: SD / √2) |
| passage alpha, run-to-run SD | 0.74 (r = 0.71) | ~0.52 |
| AUROC vs all | 0.92 / 0.93 (run 1 / run 2) | 0.93 |
| gate decisions | identical on both runs | identical |

So:
- **Rankings across classes are stable.** Both runs separate high-alpha essays from Claude and Wikipedia at 0.96–1.00.
- **A single document's score moves by about ±0.25.** Treat two texts within about 0.5 of each other as tied, or use
  `--repeats 2` or `3` (cost and time scale linearly). The largest change seen was 1.05, on a 2-passage document
  (`high_coffee`). Fewer passages means more noise, so prefer texts of 300+ words.

## Choosing a reference model

`alpha-score models` lists the registry in `src/alpha_score/models.py`.

- **Pay-per-token (recommended):** `glm-5.3-flash`. Nothing to deploy, and the best benchmark result.
  `nemotron-lightning-30b` passes the API probe but has not been benchmarked. `gpt-oss-120b` fails the benchmark.
- **Base models:** `qwen2.5-72b`, `qwen2.5-7b`, `mixtral-8x22b`, `ernie-4.5-300b`, `deepseek-v3.1-base`, …
  - A base model's idea of "predictable" is its whole pretraining corpus, which is arguably the most principled
    baseline.
  - No provider serves large base models pay-per-token. Use `alpha-score score --deploy` (a temporary Fireworks
    deployment at about $8/GPU-hour), `alpha-score deploy`, or `scripts/modal_backend.py` on your own Modal GPUs.
- **Instruct models** score raw text with no chat template. Qwen2.5-72B instruct matches base overall: it is better
  vs. Claude and worse vs. Wikipedia.
- **Any other model:** score `eval/data/chunks_main.jsonl`, run `eval/evaluate.py`, then add the output to
  `scripts/build_calibration.py` so the CLI can show percentiles for it.

## Background

This came out of a longer exploration. These were tried and dropped:
- `self_gap`: mean NLL minus mean self-sample NLL. Similar to `uzzi`, but it ranks gibberish highest.
- `style_gap` / `content_gap`: a split between function words and content words.
- Within-document information gain: word salad gains as much as essays do.
- Few-shot P(True) per claim: mostly detects encyclopedic register.
- Influence on Wikipedia text: negative for every class.
- Activation probes: did not generalize across documents.

References:
- Uzzi, Mukherjee, Stringer & Jones (2013), *Atypical combinations and scientific impact*, Science 342:468.
- Elgammal & Saleh (2015), *Quantifying creativity in art networks* (arXiv:1506.00711).
- Bao et al. (2024), *Fast-DetectGPT*. Its sampling-discrepancy statistic is the closest relative of the
  self-sample baseline.
