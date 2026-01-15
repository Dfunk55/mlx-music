#!/usr/bin/env python3
"""
Benchmark MusicGen quantization at 8-bit and 4-bit.

Compares generation performance across quantization levels:
- bfloat16 (baseline)
- INT8 (quality-focused quantization)
- INT4 (speed-focused quantization)

Usage:
    python scripts/benchmark_musicgen_quantization.py
    python scripts/benchmark_musicgen_quantization.py --models small melody
    python scripts/benchmark_musicgen_quantization.py --duration 5 --iterations 3
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_music import MusicGen
from mlx_music.weights.quantization import (
    QuantizationConfig,
    get_model_size,
    quantize_model,
)


# Model paths
MODEL_PATHS = {
    "small": "/Users/dustinpainter/Dev-Projects/audio-models/MusicGen-small",
    "melody": "/Users/dustinpainter/Dev-Projects/audio-models/MusicGen-melody",
    "large": "/Users/dustinpainter/Dev-Projects/audio-models/MusicGen-large",
}

# Model specs for display
MODEL_SPECS = {
    "small": {"hidden_size": 1024, "num_layers": 24, "num_heads": 16},
    "melody": {"hidden_size": 1536, "num_layers": 48, "num_heads": 24},
    "large": {"hidden_size": 2048, "num_layers": 48, "num_heads": 32},
}


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    prompt: str = "upbeat electronic dance music with synths"
    duration: float = 5.0  # seconds
    warmup_iterations: int = 1
    timed_iterations: int = 3
    seed: int = 42
    temperature: float = 1.0
    top_k: int = 250
    guidance_scale: float = 3.0


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    model_name: str
    quantization: str
    num_params: int
    size_mb: float
    size_reduction_pct: float
    load_time_s: float
    gen_time_s: float
    tokens_generated: int
    tokens_per_second: float
    realtime_factor: float
    quality_pass: bool
    quality_details: Dict


def clear_memory():
    """Clear MLX memory cache between runs."""
    gc.collect()
    if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()


def verify_audio_quality(audio: np.ndarray, sample_rate: int) -> Dict:
    """
    Verify generated audio has reasonable characteristics.

    Checks:
    - Not silent (has RMS energy)
    - Has dynamics (not constant)
    - Not clipped
    """
    # Flatten to 1D if needed
    if audio.ndim > 1:
        audio = audio.flatten()

    rms = float(np.sqrt(np.mean(audio**2)))
    std = float(np.std(audio))
    max_amp = float(np.max(np.abs(audio)))
    clip_ratio = float(np.mean(np.abs(audio) > 0.99))

    is_silent = rms < 1e-4
    has_dynamics = std > 1e-3
    is_clipped = clip_ratio > 0.1

    quality_pass = not is_silent and has_dynamics and not is_clipped

    return {
        "rms_energy": rms,
        "std": std,
        "max_amplitude": max_amp,
        "clip_ratio": clip_ratio,
        "is_silent": is_silent,
        "has_dynamics": has_dynamics,
        "is_clipped": is_clipped,
        "quality_pass": quality_pass,
    }


def load_and_quantize_model(
    model_path: str,
    quant_mode: Optional[str],
    dtype: mx.Dtype = mx.bfloat16,
) -> Tuple[MusicGen, float]:
    """
    Load model and optionally quantize it.

    Args:
        model_path: Path to model directory
        quant_mode: None (no quant), "int8", or "int4"
        dtype: Data type for loading

    Returns:
        Tuple of (model, load_time_seconds)
    """
    start_time = time.time()

    # Load model
    model = MusicGen.from_pretrained(
        model_path,
        dtype=dtype,
        load_text_encoder=True,
        load_encodec=True,
    )

    # Evaluate to ensure model is loaded
    mx.eval(model.decoder.parameters())

    # Apply quantization if requested
    if quant_mode == "int8":
        quantize_model(model.decoder, QuantizationConfig.for_quality())
        mx.eval(model.decoder.parameters())
    elif quant_mode == "int4":
        quantize_model(model.decoder, QuantizationConfig.for_speed())
        mx.eval(model.decoder.parameters())

    load_time = time.time() - start_time
    return model, load_time


def run_generation_benchmark(
    model: MusicGen,
    config: BenchmarkConfig,
) -> Tuple[float, np.ndarray, int]:
    """
    Run generation benchmark with warm-up and timing.

    Returns:
        Tuple of (avg_time_seconds, last_audio, tokens_generated)
    """
    frame_rate = model.config.frame_rate
    tokens_per_run = int(config.duration * frame_rate)

    # Warm-up runs
    for _ in range(config.warmup_iterations):
        mx.random.seed(config.seed)
        output = model.generate(
            prompt=config.prompt,
            duration=config.duration,
            temperature=config.temperature,
            top_k=config.top_k,
            guidance_scale=config.guidance_scale,
            seed=config.seed,
        )
        # Force evaluation
        mx.eval(mx.array(output.audio))
        clear_memory()

    # Timed runs
    times = []
    last_audio = None
    for _ in range(config.timed_iterations):
        mx.random.seed(config.seed)
        start_time = time.time()
        output = model.generate(
            prompt=config.prompt,
            duration=config.duration,
            temperature=config.temperature,
            top_k=config.top_k,
            guidance_scale=config.guidance_scale,
            seed=config.seed,
        )
        # Force evaluation
        mx.eval(mx.array(output.audio))
        elapsed = time.time() - start_time
        times.append(elapsed)
        last_audio = output.audio

    avg_time = sum(times) / len(times)
    return avg_time, last_audio, tokens_per_run


def benchmark_model(
    model_name: str,
    model_path: str,
    config: BenchmarkConfig,
) -> List[BenchmarkResult]:
    """
    Benchmark a single model at all quantization levels.

    Returns:
        List of BenchmarkResult for each quantization level
    """
    results = []
    baseline_size = None

    quant_modes = [
        (None, "bfloat16"),
        ("int8", "int8"),
        ("int4", "int4"),
    ]

    for quant_mode, quant_name in quant_modes:
        print(f"\n  Testing {quant_name}...")
        clear_memory()

        try:
            # Load and optionally quantize
            model, load_time = load_and_quantize_model(
                model_path, quant_mode, dtype=mx.bfloat16
            )

            # Get model size
            num_params, size_mb = get_model_size(model.decoder)

            # Set baseline for size reduction calculation
            if baseline_size is None:
                baseline_size = size_mb
                size_reduction = 0.0
            else:
                size_reduction = ((baseline_size - size_mb) / baseline_size) * 100

            # Run generation benchmark
            gen_time, audio, tokens = run_generation_benchmark(model, config)

            # Verify audio quality
            sample_rate = model.config.audio_encoder.sampling_rate
            quality = verify_audio_quality(audio, sample_rate)

            # Calculate metrics
            tokens_per_sec = tokens / gen_time
            realtime_factor = config.duration / gen_time

            result = BenchmarkResult(
                model_name=model_name,
                quantization=quant_name,
                num_params=num_params,
                size_mb=size_mb,
                size_reduction_pct=size_reduction,
                load_time_s=load_time,
                gen_time_s=gen_time,
                tokens_generated=tokens,
                tokens_per_second=tokens_per_sec,
                realtime_factor=realtime_factor,
                quality_pass=quality["quality_pass"],
                quality_details=quality,
            )
            results.append(result)

            print(f"    Size: {size_mb:.1f} MB ({size_reduction:+.0f}%)")
            print(f"    Gen time: {gen_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
            print(f"    Quality: {'PASS' if quality['quality_pass'] else 'FAIL'}")

            # Clean up
            del model
            clear_memory()

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    return results


def print_results_table(all_results: List[List[BenchmarkResult]], config: BenchmarkConfig):
    """Print formatted results table."""
    print("\n" + "=" * 80)
    print("MUSICGEN QUANTIZATION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Prompt: \"{config.prompt}\"")
    print(f"Duration: {config.duration}s | Warmup: {config.warmup_iterations} | Iterations: {config.timed_iterations}")
    print("=" * 80)

    for model_results in all_results:
        if not model_results:
            continue

        model_name = model_results[0].model_name
        specs = MODEL_SPECS.get(model_name, {})
        print(f"\nModel: MusicGen-{model_name} ({specs.get('hidden_size', '?')}d, {specs.get('num_layers', '?')}L)")
        print("-" * 80)
        print(f"{'Quantization':<12} {'Size (MB)':<15} {'Gen Time (s)':<14} {'Tok/s':<10} {'Speedup':<10} {'Quality':<8}")
        print("-" * 80)

        baseline_time = model_results[0].gen_time_s if model_results else 1.0

        for r in model_results:
            size_str = f"{r.size_mb:.1f}"
            if r.size_reduction_pct > 0:
                size_str += f" (-{r.size_reduction_pct:.0f}%)"

            speedup = baseline_time / r.gen_time_s
            quality_str = "PASS" if r.quality_pass else "FAIL"

            print(
                f"{r.quantization:<12} {size_str:<15} {r.gen_time_s:<14.2f} "
                f"{r.tokens_per_second:<10.1f} {speedup:<10.2f}x {quality_str:<8}"
            )

    print("\n" + "=" * 80)


def save_results_json(
    all_results: List[List[BenchmarkResult]],
    config: BenchmarkConfig,
    output_path: Path,
):
    """Save results to JSON file."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "results": [
            [asdict(r) for r in model_results]
            for model_results in all_results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MusicGen quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["small", "melody", "large"],
        default=["small", "melody"],
        help="Models to benchmark (default: small melody)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Generation duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of timed iterations (default: 3)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations (default: 1)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="upbeat electronic dance music with synths",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (optional)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        prompt=args.prompt,
        duration=args.duration,
        warmup_iterations=args.warmup,
        timed_iterations=args.iterations,
        seed=args.seed,
    )

    print("MusicGen Quantization Benchmark")
    print(f"Models: {', '.join(args.models)}")
    print(f"Duration: {config.duration}s")
    print(f"Iterations: {config.warmup_iterations} warmup + {config.timed_iterations} timed")

    all_results = []

    for model_name in args.models:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path or not Path(model_path).exists():
            print(f"\nSkipping {model_name}: model not found at {model_path}")
            continue

        print(f"\n{'='*40}")
        print(f"Benchmarking MusicGen-{model_name}")
        print(f"{'='*40}")

        results = benchmark_model(model_name, model_path, config)
        all_results.append(results)

    # Print summary table
    print_results_table(all_results, config)

    # Save JSON if requested
    if args.output:
        save_results_json(all_results, config, Path(args.output))


if __name__ == "__main__":
    main()
