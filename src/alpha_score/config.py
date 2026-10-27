"""Configuration: explicit arguments > environment variables > config file > defaults.

Config file (TOML), first found of: $ALPHA_SCORE_CONFIG, ./alpha-score.toml, ~/.config/alpha-score/config.toml

    provider = "fireworks"          # "fireworks" or "openai" (any OpenAI-compatible /v1/completions with echo)
    model = "glm-5.3-flash"         # key from `alpha-score models`, or a raw provider model id
    api_key_env = "FIREWORKS_API_KEY"   # name of the env var that holds the key (keep keys out of the file)
    # base_url = "http://localhost:8000/v1"   # provider = "openai" only
    # deployment = "alpha-score-1a2b3c4d"     # an existing Fireworks deployment (base models)
    # account_id = "my-account"                # Fireworks account id, needed for deployments only
    workers = 16                    # concurrent requests
    max_passages = 20               # ~100-word passages scored per document (longer texts are truncated)
    repeats = 1                     # average this many independent self-sample draws (less noise, cost x repeats)

Environment variables: ALPHA_SCORE_PROVIDER, ALPHA_SCORE_MODEL, ALPHA_SCORE_BASE_URL, ALPHA_SCORE_API_KEY_ENV,
ALPHA_SCORE_DEPLOYMENT, ALPHA_SCORE_WORKERS, ALPHA_SCORE_MAX_PASSAGES, ALPHA_SCORE_REPEATS, FIREWORKS_ACCOUNT_ID, and the API key itself
(FIREWORKS_API_KEY by default).
"""

import os
import tomllib
from dataclasses import asdict, dataclass, fields

from .models import FIREWORKS_DEFAULT


@dataclass
class Config:
    provider: str = "fireworks"
    model: str = FIREWORKS_DEFAULT
    api_key_env: str = "FIREWORKS_API_KEY"
    base_url: str | None = None
    deployment: str | None = None
    account_id: str | None = None
    workers: int = 16
    max_passages: int = 20
    repeats: int = 1
    source: str = "defaults"  # where the file part came from (for `alpha-score config`)

    @property
    def api_key(self):
        return os.environ.get(self.api_key_env)

    def describe(self):
        d = asdict(self)
        key = self.api_key
        d["api_key"] = f"{key[:5]}…{key[-3:]} (from ${self.api_key_env})" if key else f"NOT SET (${self.api_key_env})"
        return d


ENV = dict(provider="ALPHA_SCORE_PROVIDER", model="ALPHA_SCORE_MODEL", base_url="ALPHA_SCORE_BASE_URL",
           api_key_env="ALPHA_SCORE_API_KEY_ENV", deployment="ALPHA_SCORE_DEPLOYMENT", workers="ALPHA_SCORE_WORKERS",
           max_passages="ALPHA_SCORE_MAX_PASSAGES", repeats="ALPHA_SCORE_REPEATS", account_id="FIREWORKS_ACCOUNT_ID")


def config_path():
    for p in (os.environ.get("ALPHA_SCORE_CONFIG"), "alpha-score.toml",
              os.path.expanduser("~/.config/alpha-score/config.toml")):
        if p and os.path.exists(p):
            return p
    return None


def load_config(**overrides):
    cfg, names = {}, {f.name for f in fields(Config)}
    path = config_path()
    if path:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        if "api_key" in data:
            raise ValueError(f"{path}: put the key in an env var and set api_key_env, not api_key")
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
        cfg.update(data, source=path)
    for name, var in ENV.items():
        if os.environ.get(var):
            cfg[name] = os.environ[var]
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    for k in ("workers", "max_passages", "repeats"):
        if k in cfg:
            cfg[k] = int(cfg[k])
    if cfg.get("provider", "fireworks") == "openai" and "api_key_env" not in cfg:
        cfg["api_key_env"] = "OPENAI_API_KEY"
    return Config(**cfg)
