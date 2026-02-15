"""
Evaluator — measure quantized model quality, size, and speed.

Provides a unified evaluation interface that measures:
- Top-1 accuracy on a test dataset
- Model size in MB (serialized)
- Inference latency (average ms per sample)
- Compression ratio vs. the original model

Supports CPU, CUDA, and MPS (Apple Silicon Metal) devices.
PyTorch INT8 quantized models are automatically evaluated on CPU since
quantized ops are not supported on MPS/CUDA.
"""

from __future__ import annotations

import io
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Evaluation metrics for a single quantized model."""
    config_name: str
    accuracy_pct: float
    size_mb: float
    latency_ms: float
    compression_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _has_quantized_layers(model: nn.Module) -> bool:
    """Check if a model contains PyTorch quantized layers."""
    for module in model.modules():
        module_type = type(module).__name__
        if "Quantized" in module_type or "Dynamic" in module_type:
            return True
        # Check for quantized weight attributes
        for attr_name in ("weight", "_packed_params"):
            attr = getattr(module, attr_name, None)
            if attr is not None and hasattr(attr, "dtype"):
                if attr.dtype in (torch.qint8, torch.quint8, torch.qint32):
                    return True
    return False


def _get_eval_device(model: nn.Module, preferred_device: torch.device) -> torch.device:
    """
    Get the appropriate device for evaluating a model.

    Quantized INT8 models can only run on CPU. FP32/FP16 models can run on
    the preferred device (MPS, CUDA, or CPU).
    """
    if _has_quantized_layers(model):
        if preferred_device.type != "cpu":
            logger.info(
                f"Model has quantized layers — falling back to CPU "
                f"(quantized ops not supported on {preferred_device})"
            )
        return torch.device("cpu")
    return preferred_device


def _sync_device(device: torch.device) -> None:
    """Synchronize the device for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_model_size_mb(model: nn.Module) -> float:
    """Measure model size by serializing state_dict to buffer."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 * 1024)


def measure_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """
    Measure top-1 accuracy on a test dataset.

    Args:
        model: The model to evaluate.
        test_loader: DataLoader yielding (images, labels) batches.
        device: Device to run inference on.
        max_batches: If set, only evaluate this many batches (for speed).

    Returns:
        Top-1 accuracy as a percentage (0-100).
    """
    # Use appropriate device (quantized models must stay on CPU)
    eval_device = _get_eval_device(model, device)
    model = model.to(eval_device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            try:
                images = images.to(eval_device)
                # Handle dtype mismatch (e.g., fp16 model with fp32 input)
                first_param = next(model.parameters(), None)
                if first_param is not None and first_param.dtype == torch.float16:
                    images = images.half()

                labels = labels.to(eval_device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            except Exception as e:
                logger.warning(f"Error during accuracy eval on batch {batch_idx}: {e}")
                continue

    if total == 0:
        return 0.0

    return 100.0 * correct / total


def measure_latency(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 32, 32),
    device: torch.device = torch.device("cpu"),
    n_runs: int = 100,
    warmup_runs: int = 10,
) -> float:
    """
    Measure average inference latency in milliseconds.

    Args:
        model: The model to benchmark.
        input_shape: Shape of dummy input tensor.
        device: Device to run on.
        n_runs: Number of timed forward passes.
        warmup_runs: Number of warmup passes (not timed).

    Returns:
        Average latency in milliseconds per forward pass.
    """
    # Use appropriate device (quantized models must stay on CPU)
    eval_device = _get_eval_device(model, device)
    model = model.to(eval_device)
    model.eval()

    # Determine input dtype
    first_param = next(model.parameters(), None)
    dtype = torch.float32
    if first_param is not None and first_param.dtype == torch.float16:
        dtype = torch.float16

    dummy_input = torch.randn(*input_shape, device=eval_device, dtype=dtype)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            try:
                model(dummy_input)
            except Exception:
                break
    _sync_device(eval_device)

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            _sync_device(eval_device)
            start = time.perf_counter()
            try:
                model(dummy_input)
            except Exception:
                continue
            _sync_device(eval_device)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # convert to ms

    if not latencies:
        return float("inf")

    return sum(latencies) / len(latencies)


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    original_size_mb: float,
    config_name: str = "unknown",
    input_shape: tuple[int, ...] = (1, 3, 32, 32),
    max_eval_batches: int | None = None,
    latency_runs: int = 50,
    metadata: dict[str, Any] | None = None,
) -> EvalResult:
    """
    Full evaluation of a quantized model.

    Automatically selects the right device — quantized INT8 models are
    evaluated on CPU; FP32/FP16 models use the preferred device (MPS/CUDA/CPU).

    Args:
        model: Quantized model to evaluate.
        test_loader: Test DataLoader for accuracy.
        device: Preferred compute device.
        original_size_mb: Size of the original (unquantized) model for compression ratio.
        config_name: Human-readable name for this configuration.
        input_shape: Shape of inputs for latency measurement.
        max_eval_batches: Limit accuracy evaluation batches.
        latency_runs: Number of runs for latency measurement.
        metadata: Extra metadata to include.

    Returns:
        EvalResult with all metrics.
    """
    try:
        # Determine actual evaluation device
        eval_device = _get_eval_device(model, device)
        model = model.to(eval_device)

        # Measure size (device-independent)
        size_mb = measure_model_size_mb(model)

        # Measure accuracy
        accuracy = measure_accuracy(model, test_loader, eval_device, max_eval_batches)

        # Measure latency
        latency = measure_latency(
            model,
            input_shape=input_shape,
            device=eval_device,
            n_runs=latency_runs,
        )

        # Compression ratio
        compression_ratio = original_size_mb / size_mb if size_mb > 0 else 0.0

        result_metadata = metadata or {}
        if eval_device != device:
            result_metadata["eval_device_override"] = str(eval_device)

        return EvalResult(
            config_name=config_name,
            accuracy_pct=round(accuracy, 2),
            size_mb=round(size_mb, 2),
            latency_ms=round(latency, 3),
            compression_ratio=round(compression_ratio, 2),
            metadata=result_metadata,
        )

    except Exception as e:
        logger.error(f"Evaluation failed for {config_name}: {e}")
        return EvalResult(
            config_name=config_name,
            accuracy_pct=0.0,
            size_mb=0.0,
            latency_ms=float("inf"),
            compression_ratio=0.0,
            error=str(e),
        )
