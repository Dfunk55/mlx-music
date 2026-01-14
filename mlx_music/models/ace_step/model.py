"""
ACE-Step main model class.

Provides a high-level interface for loading and generating
music with the ACE-Step model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_music.models.ace_step.scheduler import (
    FlowMatchEulerDiscreteScheduler,
    get_scheduler,
    retrieve_timesteps,
)
from mlx_music.models.ace_step.transformer import ACEStepConfig, ACEStepTransformer
from mlx_music.weights.weight_loader import (
    download_model,
    load_ace_step_weights,
)


@dataclass
class GenerationConfig:
    """Configuration for audio generation."""

    # Duration and quality
    duration: float = 30.0  # seconds
    sample_rate: int = 44100

    # Diffusion parameters
    num_inference_steps: int = 60
    guidance_scale: float = 15.0
    guidance_scale_text: float = 5.0
    guidance_scale_lyric: float = 2.5

    # Scheduler
    scheduler_type: str = "euler"
    shift: float = 3.0

    # Generation
    seed: Optional[int] = None
    batch_size: int = 1


@dataclass
class GenerationOutput:
    """Output from music generation."""

    audio: np.ndarray  # Shape: (channels, samples) or (samples,)
    sample_rate: int
    duration: float
    latents: Optional[mx.array] = None


class ACEStep:
    """
    ACE-Step Music Generation Model.

    High-level interface for loading and generating music.

    Example:
        >>> model = ACEStep.from_pretrained("ACE-Step/ACE-Step-v1-3.5B")
        >>> output = model.generate(
        ...     prompt="upbeat electronic dance music",
        ...     lyrics="Verse 1: Dancing through the night...",
        ...     duration=30.0
        ... )
        >>> # Save audio
        >>> import soundfile as sf
        >>> sf.write("output.wav", output.audio.T, output.sample_rate)
    """

    def __init__(
        self,
        transformer: ACEStepTransformer,
        config: ACEStepConfig,
        text_encoder: Optional[Any] = None,
        dcae: Optional[Any] = None,
        vocoder: Optional[Any] = None,
        dtype: mx.Dtype = mx.bfloat16,
    ):
        self.transformer = transformer
        self.config = config
        self.text_encoder = text_encoder
        self.dcae = dcae
        self.vocoder = vocoder
        self.dtype = dtype

        # Default scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        dtype: mx.Dtype = mx.bfloat16,
        load_text_encoder: bool = True,
        load_dcae: bool = True,
        load_vocoder: bool = True,
    ) -> "ACEStep":
        """
        Load ACE-Step from pretrained weights.

        Args:
            model_path: Path to model directory or HuggingFace repo ID
            dtype: Data type for model weights
            load_text_encoder: Whether to load text encoder
            load_dcae: Whether to load DCAE (needed for full generation)
            load_vocoder: Whether to load vocoder (needed for audio output)

        Returns:
            ACEStep instance
        """
        model_path = Path(model_path)

        # Download from HuggingFace if needed
        if not model_path.exists():
            print(f"Downloading model from {model_path}...")
            model_path = download_model(str(model_path))

        # Load transformer weights and config
        print("Loading transformer...")
        weights, config_dict = load_ace_step_weights(
            model_path, component="transformer", dtype=dtype
        )

        # Create config
        config = ACEStepConfig.from_dict(config_dict)

        # Create transformer
        transformer = ACEStepTransformer(config)

        # Load weights into transformer
        transformer.load_weights(list(weights.items()))

        # Initialize additional components (placeholders for now)
        text_encoder = None
        dcae = None
        vocoder = None

        if load_text_encoder:
            print("Text encoder loading not yet implemented - using placeholder")
            # TODO: Load UMT5 encoder

        if load_dcae:
            print("DCAE loading not yet implemented - using placeholder")
            # TODO: Load DCAE

        if load_vocoder:
            print("Vocoder loading not yet implemented - using placeholder")
            # TODO: Load HiFi-GAN vocoder

        return cls(
            transformer=transformer,
            config=config,
            text_encoder=text_encoder,
            dcae=dcae,
            vocoder=vocoder,
            dtype=dtype,
        )

    def encode_text(
        self,
        prompt: str,
    ) -> Tuple[mx.array, mx.array]:
        """
        Encode text prompt to embeddings.

        Args:
            prompt: Text prompt

        Returns:
            Tuple of (embeddings, attention_mask)
        """
        if self.text_encoder is None:
            # Return placeholder embeddings
            # Shape: (batch, seq_len, dim)
            embeddings = mx.zeros((1, 64, self.config.text_embedding_dim))
            mask = mx.ones((1, 64))
            return embeddings, mask

        # TODO: Implement actual text encoding with UMT5
        raise NotImplementedError("Text encoding not yet implemented")

    def encode_lyrics(
        self,
        lyrics: str,
    ) -> Tuple[mx.array, mx.array]:
        """
        Encode lyrics to token indices.

        Args:
            lyrics: Lyrics text

        Returns:
            Tuple of (token_indices, attention_mask)
        """
        # Placeholder - return dummy tokens
        # In real implementation, use VoiceBpeTokenizer
        tokens = mx.zeros((1, 128), dtype=mx.int32)
        mask = mx.ones((1, 128))
        return tokens, mask

    def decode_latents(
        self,
        latents: mx.array,
    ) -> np.ndarray:
        """
        Decode latents to audio waveform.

        Args:
            latents: Audio latents from diffusion

        Returns:
            Audio waveform as numpy array
        """
        if self.dcae is None or self.vocoder is None:
            # Return placeholder audio (silence)
            duration = latents.shape[-1] * 512 * 8 / 44100  # Approximate
            samples = int(duration * 44100)
            return np.zeros((2, samples), dtype=np.float32)

        # TODO: Implement DCAE + vocoder decoding
        raise NotImplementedError("Audio decoding not yet implemented")

    @mx.compile
    def _transformer_forward(
        self,
        latents: mx.array,
        timestep: mx.array,
        text_embeds: mx.array,
        text_mask: mx.array,
        speaker_embeds: Optional[mx.array],
        lyric_tokens: Optional[mx.array],
        lyric_mask: Optional[mx.array],
    ) -> mx.array:
        """Compiled transformer forward pass."""
        return self.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=text_embeds,
            encoder_attention_mask=text_mask,
            speaker_embeds=speaker_embeds,
            lyric_token_idx=lyric_tokens,
            lyric_mask=lyric_mask,
        )

    def generate(
        self,
        prompt: str,
        lyrics: Optional[str] = None,
        duration: float = 30.0,
        num_inference_steps: int = 60,
        guidance_scale: float = 15.0,
        seed: Optional[int] = None,
        speaker_embeds: Optional[mx.array] = None,
        return_latents: bool = False,
        scheduler_type: str = "euler",
        callback: Optional[callable] = None,
    ) -> GenerationOutput:
        """
        Generate music from text prompt.

        Args:
            prompt: Text description of desired music
            lyrics: Optional lyrics for vocal generation
            duration: Duration in seconds (max ~4 minutes)
            num_inference_steps: Number of diffusion steps
            guidance_scale: CFG guidance scale
            seed: Random seed for reproducibility
            speaker_embeds: Optional speaker embedding for voice cloning
            return_latents: Whether to return intermediate latents
            scheduler_type: "euler" or "heun"
            callback: Optional callback(step, timestep, latents)

        Returns:
            GenerationOutput with audio and metadata
        """
        # Set seed if provided
        if seed is not None:
            mx.random.seed(seed)

        # Calculate latent dimensions
        # Audio: duration * 44100 samples
        # Latent: 8 channels, 16 height, variable width
        # Width = duration * 44100 / (512 * 8) ≈ duration * 10.77
        latent_width = int(duration * 44100 / (512 * 8))
        latent_shape = (1, self.config.in_channels, self.config.max_height, latent_width)

        # Initialize noise
        latents = mx.random.normal(latent_shape).astype(self.dtype)

        # Encode text
        text_embeds, text_mask = self.encode_text(prompt)

        # Encode lyrics if provided
        if lyrics is not None:
            lyric_tokens, lyric_mask = self.encode_lyrics(lyrics)
        else:
            lyric_tokens, lyric_mask = None, None

        # Get scheduler
        scheduler = get_scheduler(scheduler_type, shift=3.0)
        timesteps, _ = retrieve_timesteps(scheduler, num_inference_steps)

        # Diffusion loop
        for i, t in enumerate(timesteps):
            # Expand timestep for batch
            timestep = mx.array([t] * latents.shape[0])

            # Model prediction
            noise_pred = self._transformer_forward(
                latents,
                timestep,
                text_embeds,
                text_mask,
                speaker_embeds,
                lyric_tokens,
                lyric_mask,
            )

            # Classifier-free guidance
            if guidance_scale > 1.0:
                # Get unconditional prediction
                noise_pred_uncond = self._transformer_forward(
                    latents,
                    timestep,
                    mx.zeros_like(text_embeds),
                    mx.zeros_like(text_mask),
                    None,
                    None,
                    None,
                )
                # CFG combination
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # Scheduler step
            output = scheduler.step(noise_pred, t, latents)
            latents = output.prev_sample

            # Callback
            if callback is not None:
                callback(i, t, latents)

            # Evaluate for progress
            mx.eval(latents)

        # Decode to audio
        audio = self.decode_latents(latents)

        return GenerationOutput(
            audio=audio,
            sample_rate=44100,
            duration=duration,
            latents=latents if return_latents else None,
        )

    def __repr__(self) -> str:
        return (
            f"ACEStep(\n"
            f"  config={self.config},\n"
            f"  dtype={self.dtype},\n"
            f"  has_text_encoder={self.text_encoder is not None},\n"
            f"  has_dcae={self.dcae is not None},\n"
            f"  has_vocoder={self.vocoder is not None},\n"
            f")"
        )
