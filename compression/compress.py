# @Time    : 2026-02-14
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : compress.py

"""
One-shot compression pipeline: INT8 quantize -> XNNPACK .pte -> evaluate.

Takes any .pt2 exported model and a saved dataset (.pt from save_dataset),
runs post-training quantization with calibration, lowers to a deployable
.pte file, and runs the evaluation suite (cosine + argmax by default).

Usage:
    python -m compression.compress model.pt2 calibration.pt
    python -m compression.compress model.pt2 calibration.pt --output-dir out/
    python -m compression.compress model.pt2 calibration.pt --eval-samples 64
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure quantized_decomposed ops are registered before loading INT8 .pt2
from torchao.quantization.pt2e.quantize_pt2e import (  # noqa: F401
    prepare_pt2e,
    convert_pt2e,
)

from compression.utils import SavedDataset
from compression.quantize import universal_compress
from compression.lower_to_pte import lower_to_pte
from eval.eval_suite import run_eval_suite


W = 60  # column width for output formatting


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    """Print a prominent section banner."""
    print()
    print("+" + "-" * (W - 2) + "+")
    print("|" + title.center(W - 2) + "|")
    print("+" + "-" * (W - 2) + "+")


def _step(number: int, total: int, title: str) -> None:
    """Print a step header."""
    print()
    tag = f"  [{number}/{total}] {title}"
    print("-" * W)
    print(tag)
    print("-" * W)


def _kv(key: str, value: str, indent: int = 2) -> None:
    """Print a key-value line with aligned columns."""
    pad = " " * indent
    print(f"{pad}{key:<24s}{value}")


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _first_tensor(output):
    """Return the first tensor from model output."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Model output has no tensor: {type(output).__name__}")


def _tensor_stats(tensor: torch.Tensor) -> dict:
    x = tensor.detach().float()
    return {
        "shape": tuple(x.shape),
        "dtype": str(tensor.dtype),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "nonzero_frac": float((x != 0).float().mean()),
    }


def _yolo_head_stats(tensor: torch.Tensor) -> dict:
    """Return YOLO-specific channel stats when output looks like (1,84,N)."""
    x = tensor.detach().float()
    if x.ndim != 3 or x.shape[1] < 84:
        return {"available": False}

    pred = x.squeeze(0).T
    class_scores = pred[:, 4:]
    class_max, _ = class_scores.max(dim=1)

    return {
        "available": True,
        "class_nonzero_frac": float((class_scores != 0).float().mean()),
        "class_abs_max": float(class_scores.abs().max()),
        "class_q50": float(torch.quantile(class_max, 0.50)),
        "class_q90": float(torch.quantile(class_max, 0.90)),
        "class_q99": float(torch.quantile(class_max, 0.99)),
        "class_gt_005": int((class_max > 0.05).sum().item()),
        "class_gt_025": int((class_max > 0.25).sum().item()),
        "anchors": int(class_max.numel()),
    }


def _print_output_debug(label: str, tensor: torch.Tensor) -> dict:
    tstats = _tensor_stats(tensor)
    ystats = _yolo_head_stats(tensor)
    print(f"\n  Debug output: {label}")
    _kv("shape", str(tstats["shape"]), indent=4)
    _kv("dtype", str(tstats["dtype"]), indent=4)
    _kv(
        "min/max/mean/std",
        (
            f"{tstats['min']:.4g} / {tstats['max']:.4g} / "
            f"{tstats['mean']:.4g} / {tstats['std']:.4g}"
        ),
        indent=4,
    )
    _kv("nonzero fraction", f"{tstats['nonzero_frac']:.6f}", indent=4)

    if ystats["available"]:
        _kv("class nonzero frac", f"{ystats['class_nonzero_frac']:.6f}", indent=4)
        _kv("class abs max", f"{ystats['class_abs_max']:.4g}", indent=4)
        _kv(
            "class max q50/q90/q99",
            (
                f"{ystats['class_q50']:.4g} / "
                f"{ystats['class_q90']:.4g} / "
                f"{ystats['class_q99']:.4g}"
            ),
            indent=4,
        )
        _kv(
            "anchors >0.05 / >0.25",
            f"{ystats['class_gt_005']} / {ystats['class_gt_025']} of {ystats['anchors']}",
            indent=4,
        )

    return {"tensor": tstats, "yolo": ystats}


def _print_weight_zero_stats(pt2_path: str, label: str) -> None:
    ep = torch.export.load(pt2_path)
    model = ep.module()
    state = model.state_dict()
    total = 0
    zeros = 0
    for value in state.values():
        if torch.is_tensor(value):
            total += value.numel()
            zeros += int((value == 0).sum().item())
    frac = (zeros / total) if total > 0 else 0.0
    print(f"\n  Debug weights: {label}")
    _kv("tensor count", str(len(state)), indent=4)
    _kv("total elements", str(total), indent=4)
    _kv("exact zeros", f"{zeros} ({frac:.6f})", indent=4)


def _run_pte_once(pte_path: str, x: torch.Tensor) -> torch.Tensor:
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")
    output = method.execute([x.contiguous()])
    return _first_tensor(output)


def _debug_pipeline_outputs(
    fp32_pt2_path: str,
    int8_pt2_path: str,
    pte_path: str,
    sample_x: torch.Tensor,
) -> dict:
    """Run one sample through all artifacts and print detailed diagnostics."""
    print("\n" + "-" * W)
    print("  [debug] Stage-by-stage output checks")
    print("-" * W)

    with torch.no_grad():
        fp32_model = torch.export.load(fp32_pt2_path).module()
        fp32_out = _first_tensor(fp32_model(sample_x))
        fp32_stats = _print_output_debug("fp32 .pt2", fp32_out)

        int8_model = torch.export.load(int8_pt2_path).module()
        int8_out = _first_tensor(int8_model(sample_x))
        int8_stats = _print_output_debug("int8 .pt2", int8_out)

    pte_out = _run_pte_once(pte_path, sample_x)
    pte_stats = _print_output_debug("xnnpack .pte", pte_out)

    fp32_flat = fp32_out.flatten().float()
    int8_flat = int8_out.flatten().float()
    pte_flat = pte_out.flatten().float()

    print("\n  Debug similarity")
    _kv(
        "cos(fp32,int8)",
        f"{F.cosine_similarity(fp32_flat.unsqueeze(0), int8_flat.unsqueeze(0)).item():.6f}",
        indent=4,
    )
    _kv("mse(fp32,int8)", f"{F.mse_loss(fp32_flat, int8_flat).item():.6f}", indent=4)
    _kv(
        "cos(fp32,pte)",
        f"{F.cosine_similarity(fp32_flat.unsqueeze(0), pte_flat.unsqueeze(0)).item():.6f}",
        indent=4,
    )
    _kv("mse(fp32,pte)", f"{F.mse_loss(fp32_flat, pte_flat).item():.6f}", indent=4)

    pte_yolo = pte_stats["yolo"]
    if pte_yolo["available"] and pte_yolo["class_abs_max"] == 0.0:
        print("\n  [debug warning] .pte class logits are all zero.")
        print("  This indicates a lowering/runtime issue, not zeroed model weights.")

    return {
        "fp32": fp32_stats,
        "int8": int8_stats,
        "pte": pte_stats,
    }


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(model_path: str, dataset_path: str) -> None:
    """Raise early with a clear message if inputs are invalid."""
    mp = Path(model_path)
    dp = Path(dataset_path)

    if not mp.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if mp.suffix != ".pt2":
        raise ValueError(
            f"Expected a .pt2 exported program, got '{mp.suffix}' file: "
            f"{model_path}\n"
            "  Hint: export your model first with "
            "compression.utils.save_torch_export(), then pass the .pt2 file."
        )

    if not dp.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if dp.suffix != ".pt":
        raise ValueError(
            f"Expected a .pt dataset file, got '{dp.suffix}' file: "
            f"{dataset_path}\n"
            "  Hint: save your dataset first with "
            "compression.utils.save_dataset()."
        )


def _size_mb(path: str) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def compress_and_evaluate(
    model_path: str,
    dataset_path: str,
    output_dir: str = ".",
    batch_size: int = 1,
    num_calibration_batches: int = 10,
    num_eval_samples: int = 32,
    debug: bool = False,
) -> dict:
    """Quantize a .pt2 model, lower to .pte, and evaluate quality.

    This is the single entry-point that chains quantization, lowering,
    and evaluation in one call.

    Args:
        model_path: Path to the FP32 .pt2 exported program.
        dataset_path: Path to a .pt dataset file (from save_dataset).
        output_dir: Directory for output artifacts.
        batch_size: Batch size used during calibration.
        num_calibration_batches: Batches fed through the model for
            observer calibration.
        num_eval_samples: Samples used for FP32-vs-PTE evaluation.

    Returns:
        Dict containing output paths and all evaluation metrics.
    """
    _validate_inputs(model_path, dataset_path)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stem = Path(model_path).stem
    int8_pt2 = str(out / f"{stem}_int8.pt2")
    pte = str(out / f"{stem}_int8_xnnpack.pte")

    # -- Header -------------------------------------------------------------
    _banner("one-click-compress")
    _kv("Model:", model_path)
    _kv("Dataset:", dataset_path)
    _kv("Output dir:", str(out.resolve()))
    t0 = time.monotonic()

    # -- Load dataset -------------------------------------------------------
    dataset = SavedDataset(dataset_path)
    sample_x, _sample_y = dataset[0]
    if sample_x.ndim == 3:
        sample_x = sample_x.unsqueeze(0)
    sample_x = sample_x.float()

    if debug:
        _print_weight_zero_stats(model_path, "fp32 .pt2 (input)")

    # -- Step 1: INT8 Post-Training Quantization ----------------------------
    _step(1, 3, "INT8 Post-Training Quantization")
    universal_compress(
        model_path=model_path,
        calibration_dataset=dataset,
        output_path=int8_pt2,
        batch_size=batch_size,
        num_calibration_batches=num_calibration_batches,
    )
    if debug:
        _print_weight_zero_stats(int8_pt2, "int8 .pt2 (after quantization)")

    # -- Step 2: Lower to XNNPACK .pte -------------------------------------
    _step(2, 3, "Lower to XNNPACK .pte")
    lower_to_pte(int8_pt2, pte)

    debug_info = None
    if debug:
        debug_info = _debug_pipeline_outputs(
            fp32_pt2_path=model_path,
            int8_pt2_path=int8_pt2,
            pte_path=pte,
            sample_x=sample_x,
        )

    # -- Step 3: Evaluation suite -------------------------------------------
    _step(3, 3, "Run evaluation suite")
    metrics = run_eval_suite(
        fp32_model_path=model_path,
        candidate_model_path=pte,
        dataset=dataset,
        num_eval_samples=num_eval_samples,
        eval_cosine=True,
        eval_argmax=True,
    )
    if debug and debug_info is not None:
        pte_yolo = debug_info["pte"]["yolo"]
        if pte_yolo["available"]:
            metrics["pte_class_abs_max"] = pte_yolo["class_abs_max"]
            metrics["pte_class_nonzero_frac"] = pte_yolo["class_nonzero_frac"]

    fp32_mb = _size_mb(model_path)
    pte_mb = _size_mb(pte)
    size_delta_mb = pte_mb - fp32_mb
    size_delta_pct = ((size_delta_mb / fp32_mb) * 100.0) if fp32_mb > 0 else 0.0

    elapsed = time.monotonic() - t0

    # -- Final summary ------------------------------------------------------
    _banner("RESULTS")
    print()
    print("  Artifacts")
    _kv("INT8 .pt2:", int8_pt2, indent=4)
    _kv("XNNPACK .pte:", pte, indent=4)

    print()
    print("  Size")
    _kv(
        "Change (.pt2 -> .pte):",
        f"{size_delta_mb:+.2f} MB ({size_delta_pct:+.1f}%)",
        indent=4,
    )

    print()
    print("  Evaluation")
    _kv("Cosine sim (mean):", f"{metrics['cos_mean']:.6f}", indent=4)
    _kv("Cosine sim (min):", f"{metrics['cos_min']:.6f}", indent=4)
    _kv(
        "Latency mean (fp32):",
        f"{metrics['fp32_latency_ms_mean']:.3f} ms",
        indent=4,
    )
    _kv(
        "Latency mean (pte):",
        f"{metrics['candidate_latency_ms_mean']:.3f} ms",
        indent=4,
    )
    _kv(
        "Latency delta (pte):",
        f"{metrics['latency_delta_ms']:+.3f} ms",
        indent=4,
    )
    _kv(
        "Latency ratio (pte):",
        f"{metrics['latency_ratio']:.3f}x",
        indent=4,
    )
    if metrics.get("argmax_supported", False):
        _kv(
            "Argmax top-1 (fp32):",
            f"{metrics['fp32_top1']:.4f}",
            indent=4,
        )
        _kv(
            "Argmax top-1 (pte):",
            f"{metrics['candidate_top1']:.4f}",
            indent=4,
        )
        _kv(
            "Argmax agreement:",
            f"{metrics['argmax_agreement']:.4f}",
            indent=4,
        )
        _kv(
            "Argmax delta (pte):",
            f"{metrics['argmax_delta']:+.4f}",
            indent=4,
        )
    else:
        _kv(
            "Argmax:",
            f"skipped ({metrics.get('argmax_reason', 'unsupported output/labels')})",
            indent=4,
        )
    if "pte_class_abs_max" in metrics:
        _kv("Debug pte class abs max:", f"{metrics['pte_class_abs_max']:.6g}", indent=4)
        _kv(
            "Debug pte class nonzero:",
            f"{metrics['pte_class_nonzero_frac']:.6f}",
            indent=4,
        )

    print()
    _kv("Elapsed:", f"{elapsed:.1f}s")
    print("+" + "-" * (W - 2) + "+")

    return {
        "int8_pt2": int8_pt2,
        "pte": pte,
        "size_delta_mb": size_delta_mb,
        "size_delta_pct": size_delta_pct,
        **metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compress a .pt2 model: "
            "INT8 quantize -> XNNPACK .pte -> evaluate"
        ),
    )
    parser.add_argument(
        "model",
        help="Path to the FP32 .pt2 exported program",
    )
    parser.add_argument(
        "dataset",
        help="Path to the .pt calibration dataset (from save_dataset)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output artifacts (default: cwd)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Calibration batch size (default: 1)",
    )
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=10,
        help="Number of calibration batches (default: 10)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=32,
        help="Number of evaluation samples (default: 32)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print stage-by-stage output and weight diagnostics",
    )
    args = parser.parse_args()

    try:
        compress_and_evaluate(
            model_path=args.model,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_calibration_batches=args.calibration_batches,
            num_eval_samples=args.eval_samples,
            debug=args.debug,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
