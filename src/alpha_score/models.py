"""Registry of open-source reference models (base and instruct), shared by all backends.

Pass a registry key (e.g. `--model glm-5.3-flash`) to the CLI, the MCP server or scripts/modal_backend.py; a raw
Hugging Face id (Modal) or Fireworks model id still works for anything not listed.

  hf          Hugging Face repo for scripts/modal_backend.py (must be ungated: there is no HF token on Modal)
  modal_gpu   GPU spec that fits it in bf16 with vLLM
  kind        base (recommended) or instruct
  fireworks   Fireworks model id (base models need an on-demand deployment: Fireworks serves none serverless).
              None = not in Fireworks' library (upload it yourself with firectl, then pass the id)
  tested      which backends have produced benchmark scores with it (see eval/RESULTS.md)
"""

MODELS = {
    "qwen2.5-7b": dict(hf="Qwen/Qwen2.5-7B", modal_gpu="H100:1", fireworks="accounts/fireworks/models/qwen-v2p5-7b",
                       params="7B", tested=["modal"], note="cheap smoke tests; noticeably weaker (AUROC 0.78 vs 0.87)"),
    "qwen2.5-14b": dict(hf="Qwen/Qwen2.5-14B", modal_gpu="H100:1", fireworks="accounts/fireworks/models/qwen2p5-14b",
                        params="14B", tested=[], note=""),
    "qwen2.5-32b": dict(hf="Qwen/Qwen2.5-32B", modal_gpu="H100:2", fireworks="accounts/fireworks/models/qwen2p5-32b",
                        params="32B", tested=[], note=""),
    "qwen2.5-72b": dict(hf="Qwen/Qwen2.5-72B", modal_gpu="H100:4", fireworks="accounts/fireworks/models/qwen2p5-72b",
                        params="72B", tested=["modal"], note="DEFAULT; the gate and reference table are calibrated on it"),
    "mistral-nemo-12b": dict(hf="mistralai/Mistral-Nemo-Base-2407", modal_gpu="H100:1",
                             fireworks="accounts/fireworks/models/mistral-nemo-base-2407", params="12B", tested=[], note="HF repo is gated: Modal needs an HF token secret"),
    "mixtral-8x22b": dict(hf="mistralai/Mixtral-8x22B-v0.1", modal_gpu="H100:8",
                          fireworks="accounts/fireworks/models/mixtral-8x22b", params="141B MoE (39B active)", tested=[],
                          note="other model family, useful cross-check; HF repo gated (Modal needs HF token)"),
    "ernie-4.5-300b": dict(hf="baidu/ERNIE-4.5-300B-A47B-PT", modal_gpu="H200:8",
                           fireworks="accounts/fireworks/models/ernie-4p5-300b-a47b-pt", params="300B MoE (47B active)",
                           tested=[], note="largest base model in Fireworks' library"),
    "deepseek-v3.1-base": dict(hf="deepseek-ai/DeepSeek-V3.1-Base", modal_gpu="H200:8", fireworks=None,
                               params="671B MoE (37B active)", tested=["modal (self_gap only)"],
                               note="strongest tested (Claude-vs-human AUROC 0.93 on self_gap); ~$15 per 600 passages"),
    # --- instruct models: viable substitutes (they pass the voice-vs-content test), but scores run ~0.4 higher,
    # so use the instruct reference table in the README; worse at separating essays from Wikipedia (AUROC 0.86 vs 0.99).
    "qwen2.5-72b-instruct": dict(hf="Qwen/Qwen2.5-72B-Instruct", modal_gpu="H100:4",
                                 fireworks="accounts/fireworks/models/qwen2p5-72b-instruct", params="72B", kind="instruct",
                                 tested=["modal"], note="AUROC 0.87 overall, 0.93 vs Claude"),
    "qwen2.5-7b-instruct": dict(hf="Qwen/Qwen2.5-7B-Instruct", modal_gpu="H100:1",
                                fireworks="accounts/fireworks/models/qwen2p5-7b-instruct", params="7B", kind="instruct",
                                tested=["modal"], note="AUROC 0.75"),
    "llama-3.3-70b-instruct": dict(hf="meta-llama/Llama-3.3-70B-Instruct", modal_gpu="H100:4",
                                   fireworks="accounts/fireworks/models/llama-v3p3-70b-instruct", params="70B",
                                   kind="instruct", tested=[], note="other family; HF repo gated (Modal needs HF token)"),
    # --- pay-per-token on Fireworks serverless (live-tested 2026-10-22). None of the models above is served per token
    # anywhere, so these need no deployment: fireworks_backend.py calls them directly.
    "glm-5.3-flash": dict(hf=None, modal_gpu=None, fireworks="accounts/fireworks/models/glm-5p3-flash", params="MoE",
                          kind="instruct", serverless=True, tested=["fireworks"], price=(0.15, 0.50),
                          note="FIREWORKS DEFAULT; best tested (AUROC 0.92 overall, 1.00 vs wiki, 0.96 vs Claude); "
                               "$0.15/$0.50 per M tokens, ~$0.40 per 600 passages"),
    "gpt-oss-120b": dict(hf="openai/gpt-oss-120b", modal_gpu="H100:2", fireworks="accounts/fireworks/models/gpt-oss-120b",
                         params="117B MoE (5B active)", kind="instruct", serverless=True, tested=["fireworks"], price=(0.15, 0.60),
                         note="DO NOT USE: AUROC 0.49, gates 12/46 real docs (reasoning model, MXFP4)"),
    "nemotron-lightning-30b": dict(hf=None, modal_gpu=None,
                                   fireworks="accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b", params="30B MoE (3B active)",
                                   kind="instruct", serverless=True, tested=[], note="passes probe (n>1 rejected: 6x calls); not benchmarked"),
    # deepseek-v4.1-flash is serverless on Fireworks but returns 400 when echo and logprobs are combined
}
for _m in MODELS.values():
    _m.setdefault("kind", "base")
    _m.setdefault("serverless", False)
    _m.setdefault("price", None)  # (input, output) USD per million tokens, pay-per-token models only
DEFAULT = "qwen2.5-72b"  # Modal
FIREWORKS_DEFAULT = "glm-5.3-flash"  # pay-per-token, no deployment


def resolve(name):
    """Registry key -> entry; otherwise treat `name` as a raw model id."""
    if name in MODELS:
        return dict(MODELS[name], key=name)
    return dict(hf=name, modal_gpu=None, fireworks=name, params="?", kind="?", serverless=False, price=None, tested=[], note="not in registry", key=name)

