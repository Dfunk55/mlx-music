"""Audio I/O utilities for mlx-music."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def load_audio(
    path: Union[str, Path],
    sample_rate: Optional[int] = None,
    mono: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Load audio from file.

    Args:
        path: Path to audio file
        sample_rate: Target sample rate (resamples if different)
        mono: Whether to convert to mono

    Returns:
        Tuple of (audio array, sample rate)
    """
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")

    # Handle mono/stereo
    if audio.ndim == 1:
        audio = audio[None, :]  # (1, samples)
    else:
        audio = audio.T  # (channels, samples)

    if mono and audio.shape[0] > 1:
        audio = np.mean(audio, axis=0, keepdims=True)

    # Resample if needed
    if sample_rate is not None and sr != sample_rate:
        try:
            import resampy
            audio = resampy.resample(audio, sr, sample_rate, axis=1)
            sr = sample_rate
        except ImportError:
            logger.warning(f"resampy not installed, keeping original sample rate {sr}")

    return audio, sr


def save_audio(
    audio: np.ndarray,
    path: Union[str, Path],
    sample_rate: int = 44100,
) -> None:
    """
    Save audio to file.

    Args:
        audio: Audio array (channels, samples) or (samples,)
        path: Output path
        sample_rate: Sample rate
    """
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure correct shape for soundfile
    if audio.ndim == 1:
        audio = audio[:, None]  # (samples, 1)
    elif audio.ndim == 2:
        if audio.shape[0] <= 2:  # Likely (channels, samples)
            audio = audio.T  # (samples, channels)

    sf.write(str(path), audio, sample_rate)
    logger.info(f"Saved audio to {path}")
