"""
Quantization utilities for mlx-music.

Adapted from LTX-2 MLX quantization implementation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class QuantizationMode(Enum):
    """Quantization modes."""

    NONE = "none"
    INT4 = "int4"
    INT8 = "int8"
    MIXED = "mixed"  # INT8 attention + INT4 FFN


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    mode: QuantizationMode = QuantizationMode.NONE

    # Per-component settings
    attention_bits: int = 8
    ffn_bits: int = 4
    embedding_bits: int = 8

    # Group size for quantization
    group_size: int = 64

    # Layers to exclude from quantization
    exclude_layers: List[str] = field(default_factory=lambda: [
        "rotary_emb",
        "time_proj",
        "norm",
        "lyric_embs",
    ])

    @classmethod
    def for_quality(cls) -> "QuantizationConfig":
        """Config prioritizing output quality."""
        return cls(
            mode=QuantizationMode.INT8,
            attention_bits=8,
            ffn_bits=8,
            group_size=64,
        )

    @classmethod
    def for_speed(cls) -> "QuantizationConfig":
        """Config prioritizing inference speed."""
        return cls(
            mode=QuantizationMode.INT4,
            attention_bits=4,
            ffn_bits=4,
            group_size=64,
        )

    @classmethod
    def for_balanced(cls) -> "QuantizationConfig":
        """Balanced config (recommended)."""
        return cls(
            mode=QuantizationMode.MIXED,
            attention_bits=8,
            ffn_bits=4,
            group_size=64,
        )


def should_quantize_layer(name: str, config: QuantizationConfig) -> bool:
    """Check if a layer should be quantized."""
    for exclude in config.exclude_layers:
        if exclude in name:
            return False
    return True


def get_quantization_bits(name: str, config: QuantizationConfig) -> int:
    """Get quantization bits for a layer."""
    if config.mode == QuantizationMode.NONE:
        return 0

    if config.mode == QuantizationMode.INT4:
        return 4

    if config.mode == QuantizationMode.INT8:
        return 8

    # Mixed mode
    if "attn" in name or "attention" in name:
        return config.attention_bits
    elif "ff" in name or "mlp" in name:
        return config.ffn_bits
    else:
        return config.embedding_bits


def quantize_weights(
    weights: Dict[str, mx.array],
    config: QuantizationConfig,
) -> Dict[str, mx.array]:
    """
    Quantize model weights.

    Args:
        weights: Dictionary of weights to quantize
        config: Quantization configuration

    Returns:
        Dictionary of quantized weights
    """
    if config.mode == QuantizationMode.NONE:
        return weights

    quantized = {}

    for name, weight in weights.items():
        if not should_quantize_layer(name, config):
            quantized[name] = weight
            continue

        bits = get_quantization_bits(name, config)
        if bits == 0:
            quantized[name] = weight
            continue

        # Only quantize 2D weights (linear layers)
        if weight.ndim != 2:
            quantized[name] = weight
            continue

        # Apply quantization
        quantized[name] = quantize_tensor(weight, bits, config.group_size)

    return quantized


def quantize_tensor(
    tensor: mx.array,
    bits: int,
    group_size: int,
) -> mx.array:
    """
    Quantize a single tensor.

    Uses absmax quantization with grouping.
    """
    if bits >= 16:
        return tensor

    # Reshape for group quantization
    original_shape = tensor.shape
    if tensor.shape[-1] % group_size != 0:
        # Pad to group size
        pad_size = group_size - (tensor.shape[-1] % group_size)
        tensor = mx.pad(tensor, [(0, 0), (0, pad_size)])

    # Reshape to (*, group_size)
    tensor = tensor.reshape(-1, group_size)

    # Compute scale per group
    max_val = mx.max(mx.abs(tensor), axis=-1, keepdims=True)
    scale = max_val / (2 ** (bits - 1) - 1)
    scale = mx.where(scale == 0, mx.ones_like(scale), scale)

    # Quantize
    quantized = mx.round(tensor / scale)
    quantized = mx.clip(quantized, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)

    # Dequantize for inference (store as float with quantization error)
    dequantized = quantized * scale

    # Reshape back
    dequantized = dequantized.reshape(original_shape[0], -1)
    dequantized = dequantized[:, : original_shape[1]]

    return dequantized.astype(tensor.dtype)


def quantize_model(
    model: nn.Module,
    config: QuantizationConfig,
) -> nn.Module:
    """
    Quantize a model in-place.

    Args:
        model: Model to quantize
        config: Quantization configuration

    Returns:
        Quantized model
    """
    if config.mode == QuantizationMode.NONE:
        return model

    # Get all parameters
    params = dict(model.parameters())
    quantized_params = quantize_weights(params, config)

    # Update model with quantized weights
    model.load_weights(list(quantized_params.items()))

    return model
