"""
MLX Music - Native music generation library for Apple Silicon.

This library provides MLX-native implementations of music generation models,
optimized for Apple Silicon (M1/M2/M3/M4) hardware.

Supported Models:
- ACE-Step: Text-to-music generation with lyrics support

Example:
    >>> from mlx_music import ACEStep
    >>> model = ACEStep.from_pretrained("ACE-Step/ACE-Step-v1-3.5B")
    >>> audio = model.generate(
    ...     prompt="upbeat electronic dance music",
    ...     lyrics="Verse 1: Dancing through the night...",
    ...     duration=30.0
    ... )
"""

__version__ = "0.1.0"

from mlx_music.models.ace_step import ACEStep

__all__ = ["ACEStep", "__version__"]
