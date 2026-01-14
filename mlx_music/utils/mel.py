"""
Mel-spectrogram utilities for mlx-music.

Implements STFT, mel filterbank, and log-mel spectrogram computation.
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np


def hann_window(size: int) -> mx.array:
    """Create a Hann window."""
    n = mx.arange(size, dtype=mx.float32)
    return 0.5 - 0.5 * mx.cos(2 * math.pi * n / (size - 1))


def mel_filterbank(
    n_mels: int = 128,
    n_fft: int = 2048,
    sample_rate: int = 44100,
    f_min: float = 40.0,
    f_max: Optional[float] = 16000.0,
) -> mx.array:
    """
    Create a mel filterbank matrix.

    Args:
        n_mels: Number of mel bands
        n_fft: FFT size
        sample_rate: Audio sample rate
        f_min: Minimum frequency
        f_max: Maximum frequency

    Returns:
        Filterbank matrix of shape (n_mels, n_fft // 2 + 1)
    """
    if f_max is None:
        f_max = sample_rate / 2

    # Mel scale conversion
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    # Create mel points
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Create filterbank
    n_freqs = n_fft // 2 + 1
    freq_bins = np.linspace(0, sample_rate / 2, n_freqs)

    filterbank = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        # Left slope
        left_mask = (freq_bins >= left) & (freq_bins <= center)
        filterbank[i, left_mask] = (freq_bins[left_mask] - left) / (center - left + 1e-10)

        # Right slope
        right_mask = (freq_bins >= center) & (freq_bins <= right)
        filterbank[i, right_mask] = (right - freq_bins[right_mask]) / (right - center + 1e-10)

    return mx.array(filterbank, dtype=mx.float32)


def stft(
    waveform: mx.array,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: Optional[mx.array] = None,
    center: bool = False,
) -> mx.array:
    """
    Short-time Fourier Transform.

    Args:
        waveform: Input audio (samples,) or (channels, samples)
        n_fft: FFT size
        hop_length: Hop length between frames
        win_length: Window length (defaults to n_fft)
        window: Window function (defaults to Hann)
        center: Whether to pad input for centering

    Returns:
        Complex STFT of shape (..., n_fft // 2 + 1, n_frames)
    """
    if win_length is None:
        win_length = n_fft

    if window is None:
        window = hann_window(win_length)

    # Pad window if needed
    if win_length < n_fft:
        pad_amount = n_fft - win_length
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left
        window = mx.pad(window, [(pad_left, pad_right)])

    # Handle batched input
    input_dim = waveform.ndim
    if input_dim == 1:
        waveform = waveform[None, :]  # Add batch dim

    batch_size, n_samples = waveform.shape

    # Center padding if requested
    if center:
        pad_amount = n_fft // 2
        waveform = mx.pad(waveform, [(0, 0), (pad_amount, pad_amount)])
        n_samples = waveform.shape[1]

    # Calculate number of frames
    n_frames = 1 + (n_samples - n_fft) // hop_length

    # Create frame indices
    frame_indices = mx.arange(n_frames) * hop_length
    sample_indices = mx.arange(n_fft)

    # Gather frames: (batch, n_frames, n_fft)
    indices = frame_indices[:, None] + sample_indices[None, :]
    frames = mx.take(waveform, indices, axis=1)

    # Apply window
    frames = frames * window[None, None, :]

    # FFT (real input → complex output)
    # MLX doesn't have direct rfft, use full fft and take positive frequencies
    spectrum = mx.fft.fft(frames, axis=-1)

    # Take positive frequencies only
    spectrum = spectrum[..., : n_fft // 2 + 1]

    # Transpose to (..., freq, time)
    spectrum = mx.transpose(spectrum, axes=(0, 2, 1))

    if input_dim == 1:
        spectrum = spectrum[0]  # Remove batch dim

    return spectrum


def istft(
    stft_matrix: mx.array,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: Optional[mx.array] = None,
    n_fft: Optional[int] = None,
    length: Optional[int] = None,
) -> mx.array:
    """
    Inverse Short-time Fourier Transform.

    Args:
        stft_matrix: Complex STFT (..., freq, time)
        hop_length: Hop length
        win_length: Window length
        window: Window function
        n_fft: FFT size (inferred from stft_matrix if None)
        length: Output length (truncate/pad to this)

    Returns:
        Reconstructed waveform
    """
    if n_fft is None:
        n_fft = (stft_matrix.shape[-2] - 1) * 2

    if win_length is None:
        win_length = n_fft

    if window is None:
        window = hann_window(win_length)

    # Pad window if needed
    if win_length < n_fft:
        pad_amount = n_fft - win_length
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left
        window = mx.pad(window, [(pad_left, pad_right)])

    # Handle batched input
    input_dim = stft_matrix.ndim
    if input_dim == 2:
        stft_matrix = stft_matrix[None, ...]

    batch_size, n_freqs, n_frames = stft_matrix.shape

    # Reconstruct negative frequencies (conjugate symmetric)
    if n_freqs == n_fft // 2 + 1:
        # Append conjugate of positive frequencies (excluding DC and Nyquist)
        neg_freqs = mx.conj(mx.flip(stft_matrix[..., 1:-1, :], axis=-2))
        full_spectrum = mx.concatenate([stft_matrix, neg_freqs], axis=-2)
    else:
        full_spectrum = stft_matrix

    # Transpose to (batch, time, freq)
    full_spectrum = mx.transpose(full_spectrum, axes=(0, 2, 1))

    # Inverse FFT
    frames = mx.fft.ifft(full_spectrum, axis=-1).real

    # Apply window
    frames = frames * window[None, None, :]

    # Calculate output length
    expected_length = n_fft + (n_frames - 1) * hop_length
    output = mx.zeros((batch_size, expected_length))

    # Overlap-add
    for i in range(n_frames):
        start = i * hop_length
        output = output.at[:, start : start + n_fft].add(frames[:, i, :])

    # Normalize by window sum
    window_sum = mx.zeros((expected_length,))
    for i in range(n_frames):
        start = i * hop_length
        window_sum = window_sum.at[start : start + n_fft].add(window**2)

    window_sum = mx.maximum(window_sum, 1e-8)
    output = output / window_sum[None, :]

    # Trim or pad to length
    if length is not None:
        if length < output.shape[1]:
            output = output[:, :length]
        elif length > output.shape[1]:
            output = mx.pad(output, [(0, 0), (0, length - output.shape[1])])

    if input_dim == 2:
        output = output[0]

    return output


class LogMelSpectrogram:
    """
    Log-mel spectrogram extractor.

    Converts audio waveforms to log-mel spectrograms for
    input to the DCAE encoder.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 40.0,
        f_max: float = 16000.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        # Pre-compute filterbank and window
        self.mel_fb = mel_filterbank(
            n_mels=n_mels,
            n_fft=n_fft,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
        )
        self.window = hann_window(win_length)

    def __call__(self, waveform: mx.array) -> mx.array:
        """
        Compute log-mel spectrogram.

        Args:
            waveform: Audio waveform (..., samples)

        Returns:
            Log-mel spectrogram (..., n_mels, n_frames)
        """
        # Compute STFT
        spec = stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )

        # Magnitude spectrum
        mag = mx.sqrt(spec.real**2 + spec.imag**2 + 1e-10)

        # Apply mel filterbank
        # mag: (..., freq, time), mel_fb: (n_mels, freq)
        # Need: (..., n_mels, time)
        if mag.ndim == 2:
            mel = mx.matmul(self.mel_fb, mag)  # (n_mels, time)
        else:
            # Batched: (batch, freq, time) → (batch, n_mels, time)
            mel = mx.matmul(self.mel_fb[None, ...], mag)

        # Log compression with floor
        log_mel = mx.log(mx.maximum(mel, 1e-5))

        return log_mel
