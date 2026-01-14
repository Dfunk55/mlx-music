"""
ACE-Step model implementation for MLX.

ACE-Step is a diffusion-based music generation model that transforms
text prompts and lyrics into full songs.

Architecture:
- Linear Transformer (24 blocks, 2560 dim)
- DCAE (Deep Compression AutoEncoder) for audio latents
- HiFi-GAN vocoder for audio synthesis
- UMT5 text encoder for text conditioning
"""

from mlx_music.models.ace_step.model import ACEStep

__all__ = ["ACEStep"]
