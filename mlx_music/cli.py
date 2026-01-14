"""Command-line interface for mlx-music."""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MLX Music - Generate music on Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate music from text
  mlx-music generate --prompt "upbeat electronic dance music" --duration 30

  # Generate with lyrics
  mlx-music generate --prompt "pop ballad" --lyrics "Verse 1: ..." --duration 60

  # Convert and quantize model
  mlx-music convert --model ACE-Step/ACE-Step-v1-3.5B --quantize 4bit
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate music")
    gen_parser.add_argument(
        "--model",
        type=str,
        default="ACE-Step/ACE-Step-v1-3.5B",
        help="Model path or HuggingFace repo ID",
    )
    gen_parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text description of desired music",
    )
    gen_parser.add_argument(
        "--lyrics",
        type=str,
        default=None,
        help="Optional lyrics for vocal generation",
    )
    gen_parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration in seconds (default: 30)",
    )
    gen_parser.add_argument(
        "--steps",
        type=int,
        default=60,
        help="Number of diffusion steps (default: 60)",
    )
    gen_parser.add_argument(
        "--guidance",
        type=float,
        default=15.0,
        help="Guidance scale (default: 15.0)",
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    gen_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.wav",
        help="Output file path (default: output.wav)",
    )
    gen_parser.add_argument(
        "--scheduler",
        type=str,
        choices=["euler", "heun"],
        default="euler",
        help="Scheduler type (default: euler)",
    )

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert/quantize model")
    convert_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Source model path or HuggingFace repo ID",
    )
    convert_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output directory for converted model",
    )
    convert_parser.add_argument(
        "--quantize",
        type=str,
        choices=["none", "4bit", "8bit", "mixed"],
        default="none",
        help="Quantization mode",
    )
    convert_parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Weight dtype (default: bfloat16)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "generate":
        generate_command(args)
    elif args.command == "convert":
        convert_command(args)


def generate_command(args):
    """Handle generate command."""
    from tqdm import tqdm

    from mlx_music import ACEStep
    from mlx_music.utils.audio_io import save_audio

    print(f"Loading model: {args.model}")
    model = ACEStep.from_pretrained(args.model)

    print(f"Generating {args.duration}s of music...")
    print(f"  Prompt: {args.prompt}")
    if args.lyrics:
        print(f"  Lyrics: {args.lyrics[:50]}...")

    # Progress callback
    pbar = tqdm(total=args.steps, desc="Generating")

    def callback(step, timestep, latents):
        pbar.update(1)

    output = model.generate(
        prompt=args.prompt,
        lyrics=args.lyrics,
        duration=args.duration,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        scheduler_type=args.scheduler,
        callback=callback,
    )

    pbar.close()

    # Save output
    save_audio(output.audio, args.output, output.sample_rate)
    print(f"Generated {output.duration:.1f}s of audio at {output.sample_rate}Hz")


def convert_command(args):
    """Handle convert command."""
    print("Model conversion not yet implemented")
    print(f"Would convert {args.model} -> {args.output}")
    print(f"Quantization: {args.quantize}, dtype: {args.dtype}")


if __name__ == "__main__":
    main()
