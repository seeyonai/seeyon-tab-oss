"""Backend-independent synthetic episode generation and validated disk access."""
from .factory import GENERATOR_VERSION, SPLIT_FAMILIES, sample_episode
from .corpus import EpisodeCorpus, model_metadata, prediction_view, validate_episode

__all__ = [
    "GENERATOR_VERSION", "SPLIT_FAMILIES", "sample_episode", "EpisodeCorpus",
    "model_metadata", "prediction_view", "validate_episode",
]

