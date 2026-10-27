"""alpha-score: how much genuinely new content a text has, relative to a reference language model."""

from .config import Config, load_config
from .core import doc_score, passage_stats
from .scorer import CAVEAT, Scorer, explain

__version__ = "0.1.0"
__all__ = ["Scorer", "Config", "load_config", "explain", "CAVEAT", "doc_score", "passage_stats", "__version__"]
