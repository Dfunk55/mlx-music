"""
Attention mechanisms for ACE-Step.

Implements:
- Linear attention with ReLU kernel (O(n) complexity)
- Standard scaled dot-product attention
- Rotary Position Embeddings (RoPE)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class AttentionConfig:
    """Configuration for attention layers."""

    dim: int = 2560
    num_heads: int = 20
    head_dim: int = 128
    dropout: float = 0.0
    use_linear_attention: bool = True
    use_rope: bool = True
    rope_theta: float = 1000000.0
    max_position: int = 32768


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        # RMS normalization
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).

    Implements the rotary embeddings from the RoFormer paper,
    with support for 1D (sequence) positions.
    """

    def __init__(
        self,
        dim: int,
        max_position: int = 32768,
        theta: float = 1000000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.theta = theta

        # Compute inverse frequencies
        inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        self.inv_freq = inv_freq

        # Pre-compute cos/sin caches
        self._build_cache(max_position)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for positions up to seq_len."""
        positions = mx.arange(seq_len).astype(mx.float32)
        freqs = mx.outer(positions, self.inv_freq)

        # Duplicate for sin/cos pairing
        emb = mx.concatenate([freqs, freqs], axis=-1)

        self.cos_cached = mx.cos(emb)
        self.sin_cached = mx.sin(emb)

    def __call__(
        self,
        x: mx.array,
        position_ids: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Apply rotary embeddings.

        Args:
            x: Input tensor of shape (batch, seq_len, num_heads, head_dim)
            position_ids: Optional position indices

        Returns:
            Tuple of (cos, sin) embeddings
        """
        seq_len = x.shape[1]

        if position_ids is None:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            cos = mx.take(self.cos_cached, position_ids, axis=0)
            sin = mx.take(self.sin_cached, position_ids, axis=0)

        return cos, sin


def rotate_half(x: mx.array) -> mx.array:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> Tuple[mx.array, mx.array]:
    """
    Apply rotary embeddings to query and key tensors.

    Args:
        q: Query tensor (batch, seq, heads, head_dim)
        k: Key tensor (batch, seq, heads, head_dim)
        cos: Cosine embeddings (seq, head_dim) or (batch, seq, head_dim)
        sin: Sine embeddings (seq, head_dim) or (batch, seq, head_dim)

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # Expand dims for broadcasting
    if cos.ndim == 2:
        cos = cos[None, :, None, :]  # (1, seq, 1, dim)
        sin = sin[None, :, None, :]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class LinearAttention(nn.Module):
    """
    Linear attention mechanism using ReLU kernel function.

    Computes attention in O(n) time complexity by approximating
    softmax with a kernel function: K(q,k) = relu(q) * relu(k)^T

    This is the attention mechanism used in ACE-Step.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections
        self.to_q = nn.Linear(config.dim, config.num_heads * config.head_dim, bias=False)
        self.to_k = nn.Linear(config.dim, config.num_heads * config.head_dim, bias=False)
        self.to_v = nn.Linear(config.dim, config.num_heads * config.head_dim, bias=False)
        self.to_out = nn.Linear(config.num_heads * config.head_dim, config.dim)

        # Optional Q/K normalization
        self.norm_q = RMSNorm(config.head_dim)
        self.norm_k = RMSNorm(config.head_dim)

        # RoPE
        if config.use_rope:
            self.rotary_emb = RotaryEmbedding(
                dim=config.head_dim,
                max_position=config.max_position,
                theta=config.rope_theta,
            )
        else:
            self.rotary_emb = None

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Apply linear attention.

        Args:
            hidden_states: Input tensor (batch, seq, dim)
            attention_mask: Optional attention mask
            position_ids: Optional position indices for RoPE

        Returns:
            Output tensor (batch, seq, dim)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        # Reshape to (batch, seq, heads, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply Q/K normalization
        q = self.norm_q(q)
        k = self.norm_k(k)

        # Apply RoPE
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(q, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Linear attention with ReLU kernel
        # K(q, k) = relu(q) * relu(k)
        q = mx.maximum(q, 0)  # ReLU
        k = mx.maximum(k, 0)  # ReLU

        # Apply mask to keys if provided
        if attention_mask is not None:
            # Expand mask for heads: (batch, seq) -> (batch, seq, heads, 1)
            mask = attention_mask[:, :, None, None]
            k = k * mask
            v = v * mask

        # Transpose for attention computation
        # q, k, v: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))

        # Linear attention: O = (K^T @ V) @ Q^T normalized
        # First compute K^T @ V: (batch, heads, head_dim, head_dim)
        kv = mx.matmul(mx.transpose(k, axes=(0, 1, 3, 2)), v)

        # Then compute Q @ (K^T @ V): (batch, heads, seq, head_dim)
        out = mx.matmul(q, kv)

        # Normalize by sum of keys (for proper attention scaling)
        k_sum = mx.sum(k, axis=2, keepdims=True)  # (batch, heads, 1, head_dim)
        normalizer = mx.matmul(q, mx.transpose(k_sum, axes=(0, 1, 3, 2)))  # (batch, heads, seq, 1)
        normalizer = mx.maximum(normalizer, 1e-6)  # Avoid division by zero

        out = out / normalizer

        # Transpose back: (batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim)
        out = mx.transpose(out, axes=(0, 2, 1, 3))

        # Reshape: (batch, seq, heads * head_dim)
        out = out.reshape(batch_size, seq_len, -1)

        # Output projection
        out = self.to_out(out)

        return out


class CrossAttention(nn.Module):
    """
    Cross-attention for conditioning on text/lyric embeddings.

    Uses standard scaled dot-product attention.
    """

    def __init__(self, config: AttentionConfig, context_dim: int = 768):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Query projection from hidden states
        self.to_q = nn.Linear(config.dim, config.num_heads * config.head_dim, bias=False)

        # Key/Value projections from context
        self.to_k = nn.Linear(context_dim, config.num_heads * config.head_dim, bias=False)
        self.to_v = nn.Linear(context_dim, config.num_heads * config.head_dim, bias=False)

        self.to_out = nn.Linear(config.num_heads * config.head_dim, config.dim)

    def __call__(
        self,
        hidden_states: mx.array,
        context: mx.array,
        context_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Apply cross-attention.

        Args:
            hidden_states: Query tensor (batch, seq_q, dim)
            context: Key/Value tensor (batch, seq_kv, context_dim)
            context_mask: Optional mask for context (batch, seq_kv)

        Returns:
            Output tensor (batch, seq_q, dim)
        """
        batch_size, seq_q, _ = hidden_states.shape
        seq_kv = context.shape[1]

        # Project
        q = self.to_q(hidden_states)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape to multi-head
        q = q.reshape(batch_size, seq_q, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_kv, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_kv, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, heads, seq, head_dim)
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))

        # Scaled dot-product attention
        scores = mx.matmul(q, mx.transpose(k, axes=(0, 1, 3, 2))) * self.scale

        # Apply mask if provided
        if context_mask is not None:
            # Expand: (batch, seq_kv) -> (batch, 1, 1, seq_kv)
            mask = context_mask[:, None, None, :]
            scores = mx.where(mask > 0, scores, mx.array(-1e9))

        # Softmax
        weights = mx.softmax(scores, axis=-1)

        # Apply attention
        out = mx.matmul(weights, v)

        # Transpose back: (batch, seq_q, heads, head_dim)
        out = mx.transpose(out, axes=(0, 2, 1, 3))

        # Reshape and project
        out = out.reshape(batch_size, seq_q, -1)
        out = self.to_out(out)

        return out


class FeedForward(nn.Module):
    """
    Feed-forward network with GLU activation (GLUMBConv from ACE-Step).
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, mult: float = 2.5):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * mult)

        # GLU-style: project to 2x hidden, split, gate
        self.w1 = nn.Linear(dim, hidden_dim * 2)
        self.w2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        # Split and apply GLU
        h = self.w1(x)
        h, gate = mx.split(h, 2, axis=-1)
        h = h * mx.sigmoid(gate)  # SiLU gating
        return self.w2(h)


class AdaLNSingle(nn.Module):
    """
    Adaptive Layer Normalization with single conditioning.

    Takes a condition vector and produces scale/shift/gate parameters
    for modulating the hidden states.
    """

    def __init__(self, dim: int, num_params: int = 6):
        super().__init__()
        self.dim = dim
        self.num_params = num_params

        # Scale-shift table: learnable parameters
        self.scale_shift_table = mx.zeros((num_params, dim))

    def __call__(
        self,
        condition: mx.array,
    ) -> Tuple[mx.array, ...]:
        """
        Compute modulation parameters.

        Args:
            condition: Conditioning vector (batch, dim)

        Returns:
            Tuple of num_params tensors, each (batch, dim)
        """
        # Add scale_shift_table to condition
        params = condition[:, None, :] + self.scale_shift_table[None, :, :]

        # Split into individual parameters
        return tuple(params[:, i, :] for i in range(self.num_params))
