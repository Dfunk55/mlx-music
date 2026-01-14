"""
HiFi-GAN Vocoder for ACE-Step.

Converts mel-spectrograms to audio waveforms using the
ADaMoSHiFiGANV1 architecture.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class VocoderConfig:
    """Configuration for HiFi-GAN vocoder."""

    # Input
    input_channels: int = 128  # Mel bins
    sampling_rate: int = 44100
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    f_min: float = 40.0
    f_max: float = 16000.0

    # ConvNeXt Backbone
    depths: List[int] = field(default_factory=lambda: [3, 3, 9, 3])
    dims: List[int] = field(default_factory=lambda: [128, 256, 384, 512])
    kernel_size: int = 7

    # HiFi-GAN Generator
    upsample_rates: List[int] = field(default_factory=lambda: [4, 4, 2, 2, 2, 2, 2])
    upsample_kernel_sizes: List[int] = field(
        default_factory=lambda: [8, 8, 4, 4, 4, 4, 4]
    )
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11, 13])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    upsample_initial_channel: int = 1024
    pre_conv_kernel_size: int = 13
    post_conv_kernel_size: int = 13

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "VocoderConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


class LayerNorm1d(nn.Module):
    """Layer normalization for 1D sequences (channels last format)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.bias = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, channels, time) → normalize over channels
        mean = mx.mean(x, axis=1, keepdims=True)
        var = mx.var(x, axis=1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        return x * self.weight[None, :, None] + self.bias[None, :, None]


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block for mel-spectrogram processing.

    Depthwise conv → LayerNorm → Linear → GELU → Linear
    """

    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()

        # Depthwise convolution
        padding = kernel_size // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim
        )

        self.norm = LayerNorm1d(dim)
        self.pwconv1 = nn.Linear(dim, dim * 4)
        self.pwconv2 = nn.Linear(dim * 4, dim)

        # Layer scale
        self.gamma = mx.ones((dim,)) * 1e-6

    def __call__(self, x: mx.array) -> mx.array:
        residual = x

        # Depthwise conv
        x = self.dwconv(x)

        # Transpose for linear: (B, C, T) → (B, T, C)
        x = mx.transpose(x, axes=(0, 2, 1))

        # Norm and MLP
        x = self.norm(mx.transpose(x, axes=(0, 2, 1)))
        x = mx.transpose(x, axes=(0, 2, 1))

        x = self.pwconv1(x)
        x = mx.gelu(x)
        x = self.pwconv2(x)

        # Layer scale
        x = x * self.gamma[None, None, :]

        # Transpose back: (B, T, C) → (B, C, T)
        x = mx.transpose(x, axes=(0, 2, 1))

        return x + residual


class ConvNeXtEncoder(nn.Module):
    """
    ConvNeXt backbone for mel-spectrogram processing.

    4-stage encoder that processes mel-spectrograms into
    high-dimensional features for the HiFi-GAN generator.
    """

    def __init__(self, config: VocoderConfig):
        super().__init__()
        self.config = config

        # Stem
        self.stem = nn.Conv1d(
            config.input_channels,
            config.dims[0],
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2,
        )

        # Stages
        self.stages = []
        self.downsamples = []

        for i, (depth, dim) in enumerate(zip(config.depths, config.dims)):
            # Downsample between stages (not first)
            if i > 0:
                self.downsamples.append(
                    nn.Sequential(
                        LayerNorm1d(config.dims[i - 1]),
                        nn.Conv1d(config.dims[i - 1], dim, kernel_size=1),
                    )
                )
            else:
                self.downsamples.append(None)

            # ConvNeXt blocks
            blocks = [ConvNeXtBlock(dim, config.kernel_size) for _ in range(depth)]
            self.stages.append(blocks)

        # Final norm
        self.norm = LayerNorm1d(config.dims[-1])

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, n_mels, time)
        x = self.stem(x)

        for i, (downsample, stage) in enumerate(zip(self.downsamples, self.stages)):
            if downsample is not None:
                x = downsample(x)

            for block in stage:
                x = block(x)

        x = self.norm(x)
        return x


class ResBlock1(nn.Module):
    """
    HiFi-GAN residual block with multiple dilations.

    Each block has parallel paths with different dilations
    that are summed together.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: List[int] = [1, 3, 5],
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        self.convs = []
        for dilation in dilations:
            padding = (kernel_size * dilation - dilation) // 2
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
                    nn.LeakyReLU(negative_slope=0.1),
                    nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
                    nn.LeakyReLU(negative_slope=0.1),
                )
            )

    def __call__(self, x: mx.array) -> mx.array:
        for conv in self.convs:
            x = x + conv(x)
        return x


class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN generator head.

    Upsamples backbone features to audio waveform using
    transposed convolutions and multi-kernel residual blocks.
    """

    def __init__(self, config: VocoderConfig):
        super().__init__()
        self.config = config

        # Pre-conv from backbone output
        self.conv_pre = nn.Conv1d(
            config.dims[-1],
            config.upsample_initial_channel,
            kernel_size=config.pre_conv_kernel_size,
            padding=config.pre_conv_kernel_size // 2,
        )

        # Upsampling layers
        self.ups = []
        self.resblocks = []

        ch = config.upsample_initial_channel
        for i, (u_rate, u_kernel) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes)
        ):
            # Transposed convolution for upsampling
            ch_out = ch // 2
            self.ups.append(
                nn.ConvTranspose1d(
                    ch,
                    ch_out,
                    kernel_size=u_kernel,
                    stride=u_rate,
                    padding=(u_kernel - u_rate) // 2,
                )
            )

            # Multi-kernel residual blocks
            blocks = []
            for kernel, dilations in zip(
                config.resblock_kernel_sizes, config.resblock_dilation_sizes
            ):
                blocks.append(ResBlock1(ch_out, kernel, dilations))
            self.resblocks.append(blocks)

            ch = ch_out

        # Post-conv to audio
        self.conv_post = nn.Conv1d(
            ch,
            1,  # Mono output
            kernel_size=config.post_conv_kernel_size,
            padding=config.post_conv_kernel_size // 2,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_pre(x)
        x = mx.leaky_relu(x, negative_slope=0.1)

        for i, (up, resblock_group) in enumerate(zip(self.ups, self.resblocks)):
            x = up(x)
            x = mx.leaky_relu(x, negative_slope=0.1)

            # Apply all residual blocks and average
            xs = None
            for resblock in resblock_group:
                if xs is None:
                    xs = resblock(x)
                else:
                    xs = xs + resblock(x)
            x = xs / len(resblock_group)

        x = mx.leaky_relu(x, negative_slope=0.1)
        x = self.conv_post(x)
        x = mx.tanh(x)

        return x


class HiFiGANVocoder(nn.Module):
    """
    Full HiFi-GAN vocoder for mel-to-audio conversion.

    Combines ConvNeXt backbone with HiFi-GAN generator.
    """

    def __init__(self, config: VocoderConfig):
        super().__init__()
        self.config = config
        self.backbone = ConvNeXtEncoder(config)
        self.generator = HiFiGANGenerator(config)

    def __call__(self, mel: mx.array) -> mx.array:
        """
        Convert mel-spectrogram to audio.

        Args:
            mel: Log-mel spectrogram (batch, n_mels, time)

        Returns:
            Audio waveform (batch, 1, samples)
        """
        # Backbone: mel → features
        features = self.backbone(mel)

        # Generator: features → audio
        audio = self.generator(features)

        return audio

    def decode(self, mel: mx.array) -> mx.array:
        """Alias for __call__ for consistency."""
        return self(mel)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "HiFiGANVocoder":
        """Load vocoder from pretrained weights."""
        import json
        from pathlib import Path

        from mlx_music.weights.weight_loader import load_safetensors

        model_path = Path(model_path)

        # Load config
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = VocoderConfig.from_dict(config_dict)
        else:
            config = VocoderConfig()

        # Create model
        model = cls(config)

        # Load weights
        weight_file = model_path / "diffusion_pytorch_model.safetensors"
        if weight_file.exists():
            weights = load_safetensors(weight_file, dtype=dtype)
            model.load_weights(list(weights.items()))

        return model


class MusicDCAEPipeline:
    """
    Complete audio processing pipeline.

    Combines DCAE and HiFi-GAN for:
    - audio → mel → latent (encode)
    - latent → mel → audio (decode)
    """

    def __init__(
        self,
        dcae: "DCAE",
        vocoder: HiFiGANVocoder,
        sample_rate: int = 44100,
    ):
        from mlx_music.models.ace_step.dcae import DCAE
        from mlx_music.utils.mel import LogMelSpectrogram

        self.dcae = dcae
        self.vocoder = vocoder
        self.sample_rate = sample_rate

        # Mel transform
        self.mel_transform = LogMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            n_mels=128,
            f_min=40.0,
            f_max=16000.0,
        )

    def encode(self, audio: mx.array) -> mx.array:
        """
        Encode audio to latent space.

        Args:
            audio: Stereo audio (2, samples) or (batch, 2, samples)

        Returns:
            Latent (batch, 8, H, W)
        """
        # Add batch dim if needed
        if audio.ndim == 2:
            audio = audio[None, ...]

        # Extract mel
        # Process each channel separately
        mel_ch1 = self.mel_transform(audio[:, 0, :])
        mel_ch2 = self.mel_transform(audio[:, 1, :])

        # Stack channels: (batch, 2, n_mels, time)
        mel = mx.stack([mel_ch1, mel_ch2], axis=1)

        # Normalize mel
        mel = self.dcae.normalize_mel(mel)

        # Encode to latent
        latent = self.dcae.encode(mel)

        return latent

    def decode(self, latent: mx.array) -> mx.array:
        """
        Decode latent to audio.

        Args:
            latent: Latent (batch, 8, H, W)

        Returns:
            Stereo audio (batch, 2, samples)
        """
        # Decode to mel
        mel = self.dcae.decode(latent)

        # Denormalize mel
        mel = self.dcae.denormalize_mel(mel)

        # Vocoder: mel → audio (process channels separately)
        audio_ch1 = self.vocoder.decode(mel[:, 0:1, :, :].squeeze(1))
        audio_ch2 = self.vocoder.decode(mel[:, 1:2, :, :].squeeze(1))

        # Combine channels
        audio = mx.concatenate([audio_ch1, audio_ch2], axis=1)

        return audio.squeeze(0) if latent.ndim == 3 else audio

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "MusicDCAEPipeline":
        """Load complete pipeline from pretrained weights."""
        from pathlib import Path

        from mlx_music.models.ace_step.dcae import DCAE

        model_path = Path(model_path)

        # Load DCAE
        dcae_path = model_path / "music_dcae_f8c8"
        dcae = DCAE.from_pretrained(str(dcae_path), dtype=dtype)

        # Load vocoder
        vocoder_path = model_path / "music_vocoder"
        vocoder = HiFiGANVocoder.from_pretrained(str(vocoder_path), dtype=dtype)

        return cls(dcae, vocoder)
