# MLX Music

The first MLX-native music generation library for Apple Silicon.

Generate music from text descriptions and lyrics, optimized for M1/M2/M3/M4 Macs.

## Status: Early Development

This library is under active development. Currently implementing:
- [x] Core transformer architecture
- [x] Linear attention with RoPE
- [x] Flow matching scheduler
- [x] Weight loading from SafeTensors
- [x] Basic quantization support
- [ ] DCAE (audio encoder/decoder)
- [ ] HiFi-GAN vocoder
- [ ] UMT5 text encoder
- [ ] End-to-end generation
- [ ] Voice cloning support

## Supported Models

| Model | Status | Description |
|-------|--------|-------------|
| ACE-Step | In Progress | 3.5B param diffusion model for lyrics-to-music |

## Installation

```bash
# From source (recommended during development)
git clone https://github.com/Dfunk55/mlx-music.git
cd mlx-music
pip install -e ".[dev]"
```

## Quick Start

```python
from mlx_music import ACEStep

# Load model
model = ACEStep.from_pretrained("ACE-Step/ACE-Step-v1-3.5B")

# Generate music
output = model.generate(
    prompt="upbeat electronic dance music with heavy bass",
    lyrics="Verse 1: Dancing through the night...",
    duration=30.0,
    num_inference_steps=60,
    guidance_scale=15.0,
)

# Save audio
import soundfile as sf
sf.write("output.wav", output.audio.T, output.sample_rate)
```

## CLI Usage

```bash
# Generate music
mlx-music generate \
    --prompt "calm piano melody with soft strings" \
    --duration 30 \
    --output output.wav

# With lyrics
mlx-music generate \
    --prompt "pop ballad" \
    --lyrics "Verse 1: Under starlit skies we dance..." \
    --duration 60 \
    --output ballad.wav
```

## Architecture

MLX Music implements the ACE-Step architecture:

```
ACE-Step (3.5B parameters)
├── Linear Transformer (24 blocks, 2560 dim)
│   ├── Linear attention with ReLU kernel (O(n) complexity)
│   ├── Rotary Position Embeddings (RoPE)
│   ├── Cross-attention for text/lyric conditioning
│   └── AdaLN-single timestep conditioning
├── DCAE (Deep Compression AutoEncoder)
│   ├── Encoder: Audio → Latent space
│   └── Decoder: Latent → Mel-spectrogram
├── HiFi-GAN Vocoder
│   └── Mel-spectrogram → Audio waveform
└── UMT5 Text Encoder
    └── Text → Embeddings
```

## Quantization

Reduce memory usage and improve performance:

```python
from mlx_music import ACEStep
from mlx_music.weights import QuantizationConfig, quantize_model

model = ACEStep.from_pretrained("ACE-Step/ACE-Step-v1-3.5B")

# Quantize for speed (INT4)
config = QuantizationConfig.for_speed()
model = quantize_model(model.transformer, config)

# Or balanced (INT8 attention + INT4 FFN)
config = QuantizationConfig.for_balanced()
```

## Requirements

- Python 3.10+
- Apple Silicon Mac (M1/M2/M3/M4)
- MLX >= 0.25.0

## Acknowledgements

- [Apple MLX Team](https://github.com/ml-explore/mlx) - MLX framework
- [ACE-Step](https://github.com/ace-step/ACE-Step) - Original model
- [MFLUX](https://github.com/filipstrand/mflux) - Architecture patterns
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) - DSP utilities
- [LTX-2 MLX](https://github.com/Acelogic/LTX-2-MLX) - Reference implementation

## License

MIT License - See [LICENSE](LICENSE)

## Citation

```bibtex
@misc{mlx-music,
  author = {MLX Music Contributors},
  title = {MLX Music: Native Music Generation for Apple Silicon},
  year = {2025},
  howpublished = {\url{https://github.com/Dfunk55/mlx-music}},
}
```
