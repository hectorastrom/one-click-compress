"""
CLI entry point for the agentic quantization pipeline.

Usage:
    python -m src.main --model resnet18 --api-key <KEY>
    python -m src.main --model clip-vit-l --api-key <KEY> --show-examples
    python -m src.main --model mobilenet_v2 --provider anthropic --api-key <KEY>
"""

from __future__ import annotations

import argparse
import logging
import os
import ssl
import sys

# Fix SSL certificate issue for dataset downloads
if hasattr(ssl, "_create_unverified_context"):
    ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from rich.console import Console
from rich.panel import Panel

from .agent import LLMClient, QuantizationOrchestrator
from .quantizers.base import build_default_registry

console = Console()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

TORCHVISION_MODELS = {
    "resnet18": {
        "factory": torchvision.models.resnet18,
        "weights": torchvision.models.ResNet18_Weights.DEFAULT,
        "input_size": 224,
    },
    "mobilenet_v2": {
        "factory": torchvision.models.mobilenet_v2,
        "weights": torchvision.models.MobileNet_V2_Weights.DEFAULT,
        "input_size": 224,
    },
    "vit_b_16": {
        "factory": torchvision.models.vit_b_16,
        "weights": torchvision.models.ViT_B_16_Weights.DEFAULT,
        "input_size": 224,
    },
}

CLIP_MODELS = {
    "clip-vit-l": {
        "hf_name": "openai/clip-vit-large-patch14",
        "input_size": 224,
        "params": "428M",
    },
    "clip-vit-b": {
        "hf_name": "openai/clip-vit-base-patch16",
        "input_size": 224,
        "params": "150M",
    },
}

SUPPORTED_MODELS = list(TORCHVISION_MODELS.keys()) + list(CLIP_MODELS.keys())


def load_model(model_name: str) -> tuple:
    """Load a pretrained model.

    Returns (model, input_size).
    For CLIP models, returns the CLIPClassifier wrapper with pre-computed text embeddings.
    """
    if model_name in CLIP_MODELS:
        return _load_clip_model(model_name)
    elif model_name in TORCHVISION_MODELS:
        return _load_torchvision_model(model_name)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported: {SUPPORTED_MODELS}"
        )


def _load_clip_model(model_name: str) -> tuple:
    """Load a CLIP model as a zero-shot classifier."""
    from .clip_wrapper import load_clip_model

    info = CLIP_MODELS[model_name]
    console.print(
        f"Loading [bold]{model_name}[/bold] ({info['hf_name']}, ~{info['params']} params)..."
    )
    console.print("  This is a VLM — using zero-shot classification on CIFAR-10.")

    model, input_size, _ = load_clip_model(
        model_name=info["hf_name"],
    )
    model.eval()
    return model, input_size


def _load_torchvision_model(model_name: str) -> tuple:
    """Load a torchvision model with CIFAR-10 head replacement."""
    info = TORCHVISION_MODELS[model_name]
    console.print(f"Loading pretrained [bold]{model_name}[/bold]...")
    model = info["factory"](weights=info["weights"])
    model.eval()

    # For CIFAR-10 (10 classes), replace the final classifier head
    # since pretrained models are for ImageNet (1000 classes)
    num_classes = 10  # CIFAR-10
    if model_name == "resnet18":
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, num_classes
        )
    elif model_name == "vit_b_16":
        model.heads.head = torch.nn.Linear(
            model.heads.head.in_features, num_classes
        )

    return model, info["input_size"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_cifar10(
    input_size: int = 224,
    batch_size: int = 64,
    max_test_samples: int | None = 2000,
    max_calib_samples: int = 500,
    data_dir: str = "./data",
    use_clip_transform: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 test set and a calibration subset.

    Args:
        input_size: Resize images to this size.
        batch_size: Batch size for DataLoaders.
        max_test_samples: Limit test set size for faster eval. None = full set.
        max_calib_samples: Number of calibration samples.
        data_dir: Where to download/cache the dataset.
        use_clip_transform: If True, use CLIP-specific normalization.

    Returns:
        (test_loader, calibration_loader)
    """
    if use_clip_transform:
        from .clip_wrapper import get_clip_transform
        transform = get_clip_transform(input_size)
        console.print(f"Loading CIFAR-10 with [bold]CLIP transforms[/bold] ({input_size}x{input_size})...")
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        console.print(f"Loading CIFAR-10 dataset (resize to {input_size}x{input_size})...")

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    # Subset test set for speed
    if max_test_samples is not None and len(test_dataset) > max_test_samples:
        indices = list(range(max_test_samples))
        test_dataset = Subset(test_dataset, indices)

    # Calibration subset from training set
    calib_indices = list(range(min(max_calib_samples, len(train_dataset))))
    calib_dataset = Subset(train_dataset, calib_indices)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    calib_loader = DataLoader(
        calib_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    console.print(
        f"  Test samples: {len(test_dataset)}  |  "
        f"Calibration samples: {len(calib_dataset)}"
    )

    return test_loader, calib_loader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic Model Quantization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --model clip-vit-l --api-key $OPENAI_API_KEY --show-examples
  python -m src.main --model clip-vit-b --api-key $OPENAI_API_KEY --show-examples
  python -m src.main --model resnet18 --api-key $OPENAI_API_KEY
  python -m src.main --model mobilenet_v2 --provider anthropic --api-key $ANTHROPIC_API_KEY
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=SUPPORTED_MODELS,
        help="Model to quantize (default: resnet18). CLIP models use zero-shot classification.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the LLM provider. Also reads from OPENAI_API_KEY or ANTHROPIC_API_KEY env vars.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="",
        help="Specific LLM model name (e.g., gpt-4o, claude-sonnet-4-20250514). Uses provider default if not set.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Max agent iterations (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation (default: 64)",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=2000,
        help="Max test samples for evaluation (default: 2000, use -1 for full set)",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Max batches for accuracy evaluation (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect cuda/cpu)",
    )
    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="Generate visual prediction examples comparing baseline vs quantized models",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Banner
    console.print(Panel(
        "[bold blue]Compression Agent[/bold blue]\n"
        "Agentic Model Quantization Pipeline",
        border_style="blue",
    ))

    # Resolve API key
    api_key = args.api_key
    if api_key is None:
        env_var = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
        api_key = os.environ.get(env_var, "")
        if not api_key:
            console.print(
                f"[red]No API key provided. Set --api-key or {env_var} environment variable.[/red]"
            )
            sys.exit(1)

    # Resolve device — prefer CUDA > MPS > CPU
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    console.print(f"Using device: [bold]{device}[/bold]")

    # Load model
    model, input_size = load_model(args.model)

    # Load dataset
    is_clip = args.model in CLIP_MODELS
    max_test = None if args.max_test_samples == -1 else args.max_test_samples
    # Use smaller batch size for large models to avoid OOM
    batch_size = min(args.batch_size, 16) if is_clip else args.batch_size
    test_loader, calib_loader = load_cifar10(
        input_size=input_size,
        batch_size=batch_size,
        max_test_samples=max_test,
        use_clip_transform=is_clip,
    )

    # Setup LLM client
    llm_client = LLMClient(
        provider=args.provider,
        api_key=api_key,
        model=args.llm_model,
    )

    # Build quantizer registry
    registry = build_default_registry()
    console.print(f"Available quantization methods: {registry.available_methods()}")

    # Run orchestrator
    orchestrator = QuantizationOrchestrator(
        llm_client=llm_client,
        registry=registry,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
        show_examples=args.show_examples,
    )

    input_shape = (1, 3, input_size, input_size)

    results = orchestrator.run(
        model=model,
        model_name=args.model,
        test_loader=test_loader,
        calibration_loader=calib_loader,
        device=device,
        input_shape=input_shape,
        max_eval_batches=args.max_eval_batches,
    )

    # Summary
    console.print(f"\n[bold]Total configurations tried: {len(results)}[/bold]")
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    console.print(f"  Successful: {len(successful)}  |  Failed: {len(failed)}")
    console.print(f"\nResults saved to [bold]{args.output_dir}/[/bold]")


if __name__ == "__main__":
    main()
