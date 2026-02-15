# Compression Agent — Agentic Model Quantization

An LLM-driven pipeline that automatically scans PyTorch vision models (including VLMs like CLIP), proposes quantization strategies using per-layer Beta distribution analysis, executes them, and finds Pareto-optimal trade-offs between model size, accuracy, and inference speed.

## Quick Start

```bash
pip install -r requirements.txt

# Best demo — CLIP ViT-L with zero-shot classification (428M params, real accuracy)
python -m src.main --model clip-vit-l --api-key $OPENAI_API_KEY --show-examples

# Faster demo — CLIP ViT-B (150M params)
python -m src.main --model clip-vit-b --api-key $OPENAI_API_KEY --show-examples

# Smaller models
python -m src.main --model resnet18 --api-key $OPENAI_API_KEY
python -m src.main --model mobilenet_v2 --provider anthropic --api-key $ANTHROPIC_API_KEY
```

## Supported Models

| Model | Params | Size | Type | Key |
|-------|--------|------|------|-----|
| CLIP ViT-L/14 | 428M | ~1.7 GB | VLM (zero-shot) | `clip-vit-l` |
| CLIP ViT-B/16 | 150M | ~571 MB | VLM (zero-shot) | `clip-vit-b` |
| ViT-B/16 | 86.6M | ~330 MB | Vision Transformer | `vit_b_16` |
| ResNet-18 | 11.7M | ~43 MB | CNN | `resnet18` |
| MobileNetV2 | 3.5M | ~14 MB | CNN | `mobilenet_v2` |

CLIP models use **zero-shot classification** on CIFAR-10 — no fine-tuning needed, real accuracy (~90%), and clear differences between baseline and quantized models.

## Quantization Methods

- **Dynamic INT8** — PyTorch `quantize_dynamic`, no calibration needed
- **Static INT8** — PyTorch static quantization with calibration (CNNs)
- **Float16** — Simple half-precision conversion
- **Mixed Precision** — Per-layer bit-widths guided by Beta distribution sensitivity analysis
- **bitsandbytes 8-bit** — Linear layer replacement with LLM.int8()
- **bitsandbytes 4-bit** — NF4/FP4 quantization

## How It Works

1. **Scan** — Analyze model architecture, fit Beta distributions per layer, compute quantization sensitivity scores
2. **Plan** — LLM agent uses sensitivity analysis to propose quantization configs (including mixed-precision)
3. **Quantize** — Execute each proposed configuration
4. **Evaluate** — Measure accuracy (CIFAR-10 zero-shot for CLIP), model size, inference latency on MPS/CUDA/CPU
5. **Analyze** — LLM agent reviews results, optionally proposes refined configs
6. **Report** — Pareto frontier visualization + prediction examples saved to `results/`

## Project Structure

```
src/
  main.py             # CLI entry point
  scanner.py          # Model scanner with Beta distribution fitting
  agent.py            # LLM agent (planning + analysis)
  clip_wrapper.py     # CLIP zero-shot classifier wrapper
  evaluator.py        # Accuracy, size, latency (MPS/CUDA/CPU)
  visualize.py        # Prediction examples + failure analysis
  pareto.py           # Pareto frontier analysis + plotting
  quantizers/
    base.py           # Abstract backend + registry
    pytorch_native.py # Dynamic, static, fp16, mixed precision
    bnb.py            # bitsandbytes 4-bit / 8-bit
```
