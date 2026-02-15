# @Time    : 2026-02-14
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : argmax_accuracy.py

"""
Argmax top-1 accuracy benchmark for categorical classifiers.

This script is intended for classification models (for example ResNet on
ImageNette/ImageNet-style labels). It compares FP32 vs quantized models and
reports:
  - Top-1 accuracy for each model
  - Prediction agreement between models
  - Delta in top-1 accuracy

Note: Argmax accuracy is NOT meaningful for detection/segmentation outputs
such as YOLO boxes. Use this benchmark only for categorical logits.

Usage:
  uv run -m eval.argmax_accuracy \
      --fp32 weights/resnet50.pt2 \
      --quantized out/resnet50_int8.pt2 \
      --dataset data/imagenette_calibration.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Register quantized ops so INT8 .pt2 files can be loaded
from torchao.quantization.pt2e.quantize_pt2e import (  # noqa: F401
    prepare_pt2e,
    convert_pt2e,
)

from compression.utils import SavedDataset


def _first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Model output has no tensor: {type(output).__name__}")


def _load_predict_fn(model_path: str):
    suffix = Path(model_path).suffix
    if suffix == ".pt2":
        ep = torch.export.load(model_path)
        model = ep.module()

        @torch.no_grad()
        def predict(x: torch.Tensor) -> torch.Tensor:
            return _first_tensor(model(x))

        return predict, "pt2"

    if suffix == ".pte":
        from executorch.runtime import Runtime

        runtime = Runtime.get()
        program = runtime.load_program(model_path)
        method = program.load_method("forward")

        def predict(x: torch.Tensor) -> torch.Tensor:
            output = method.execute([x.contiguous()])
            return _first_tensor(output)

        return predict, "pte"

    raise ValueError(
        f"Unsupported model format '{suffix}'. Expected .pt2 or .pte"
    )


def _labels_to_class_ids(y: torch.Tensor) -> torch.Tensor:
    # Scalar class ids: [B] or [B, 1]
    if y.ndim == 1:
        return y.long()
    if y.ndim == 2 and y.shape[1] == 1:
        return y.squeeze(1).long()

    # One-hot/multi-logit labels: [B, C]
    if y.ndim == 2 and y.shape[1] > 1:
        return y.argmax(dim=1).long()

    raise ValueError(
        "Unsupported label shape for categorical benchmark: "
        f"{tuple(y.shape)}"
    )


def _logits_to_class_ids(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(
            "Model output is not categorical logits [B, C]. "
            f"Got shape {tuple(logits.shape)}"
        )
    return logits.argmax(dim=1).long()


def benchmark_argmax_accuracy(
    fp32_path: str,
    quantized_path: str,
    dataset_path: str,
    max_samples: int = 0,
    batch_size: int = 1,
) -> dict:
    dataset = SavedDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    fp32_predict, fp32_fmt = _load_predict_fn(fp32_path)
    q_predict, q_fmt = _load_predict_fn(quantized_path)

    print("\nArgmax Accuracy Benchmark (categorical models only)")
    print("-" * 60)
    print(f"FP32 model:      {fp32_path} ({fp32_fmt})")
    print(f"Quantized model: {quantized_path} ({q_fmt})")
    print(f"Dataset:         {dataset_path}")
    print(f"Batch size:      {batch_size}")
    print("-" * 60)

    total = 0
    fp32_correct = 0
    q_correct = 0
    agree = 0

    with torch.no_grad():
        for x, y in loader:
            if max_samples > 0 and total >= max_samples:
                break

            if max_samples > 0:
                remaining = max_samples - total
                if x.shape[0] > remaining:
                    x = x[:remaining]
                    y = y[:remaining]

            y_true = _labels_to_class_ids(y)

            fp32_logits = fp32_predict(x.float())
            q_logits = q_predict(x.float())

            fp32_pred = _logits_to_class_ids(fp32_logits)
            q_pred = _logits_to_class_ids(q_logits)

            fp32_correct += int((fp32_pred == y_true).sum().item())
            q_correct += int((q_pred == y_true).sum().item())
            agree += int((fp32_pred == q_pred).sum().item())
            total += int(y_true.numel())

    if total == 0:
        raise RuntimeError("No samples evaluated. Check dataset and max_samples.")

    fp32_acc = fp32_correct / total
    q_acc = q_correct / total
    agreement = agree / total
    acc_delta = q_acc - fp32_acc

    print(f"Samples evaluated:      {total}")
    print(f"FP32 top-1 accuracy:    {fp32_acc:.4f}")
    print(f"INT8 top-1 accuracy:    {q_acc:.4f}")
    print(f"Prediction agreement:   {agreement:.4f}")
    print(f"Accuracy delta (INT8):  {acc_delta:+.4f}")

    return {
        "samples": total,
        "fp32_top1": fp32_acc,
        "quantized_top1": q_acc,
        "agreement": agreement,
        "delta": acc_delta,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Argmax top-1 benchmark for categorical classifiers",
    )
    parser.add_argument("--fp32", required=True, help="Path to FP32 model (.pt2/.pte)")
    parser.add_argument(
        "--quantized",
        required=True,
        help="Path to quantized model (.pt2/.pte)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to saved dataset .pt (classification labels)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit evaluated samples (0 = all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation batch size (default: 1 for exported model guards)",
    )
    args = parser.parse_args()

    for path in [args.fp32, args.quantized, args.dataset]:
        if not Path(path).exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    try:
        benchmark_argmax_accuracy(
            fp32_path=args.fp32,
            quantized_path=args.quantized,
            dataset_path=args.dataset,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
        )
    except ValueError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        print(
            "This benchmark is for categorical classifiers only (logits [B, C]).",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
