# @Time    : 2026-02-14 23:48
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : eval_suite.py

"""
Shared evaluation suite for compressed model artifacts.

Default metrics:
  - Cosine similarity between reference and candidate outputs
  - Argmax/top-1 metrics on the provided dataset labels
  - Inference latency for reference and candidate models
"""

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Register quantized ops so INT8 .pt2 files can be loaded
from torchao.quantization.pt2e.quantize_pt2e import (  # noqa: F401
    prepare_pt2e,
    convert_pt2e,
)


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
        exported = torch.export.load(model_path)
        model = exported.module()

        @torch.no_grad()
        def predict(x: torch.Tensor) -> torch.Tensor:
            return _first_tensor(model(x))

        return predict, "pt2"

    if suffix == ".pte":
        from executorch.runtime import Runtime

        runtime = Runtime.get()
        program = runtime.load_program(model_path)
        method = program.load_method("forward")

        @torch.no_grad()
        def predict(x: torch.Tensor) -> torch.Tensor:
            output = method.execute([x.contiguous()])
            return _first_tensor(output)

        return predict, "pte"

    raise ValueError(
        f"Unsupported model format '{suffix}'. Expected .pt2 or .pte"
    )


def _labels_to_class_ids(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 1:
        return y.long()
    if y.ndim == 2 and y.shape[1] == 1:
        return y.squeeze(1).long()
    if y.ndim == 2 and y.shape[1] > 1:
        return y.argmax(dim=1).long()
    raise ValueError(
        "Unsupported label shape for argmax benchmark: "
        f"{tuple(y.shape)}"
    )


def _logits_to_class_ids(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(
            "Output is not categorical logits [B, C]. "
            f"Got shape {tuple(logits.shape)}"
        )
    return logits.argmax(dim=1).long()


def run_eval_suite(
    fp32_model_path: str,
    candidate_model_path: str,
    dataset: Dataset,
    num_eval_samples: int = 32,
    batch_size: int = 1,
    eval_cosine: bool = True,
    eval_argmax: bool = True,
) -> dict:
    """Run default eval metrics between FP32 model and candidate artifact."""
    if not eval_cosine and not eval_argmax:
        raise ValueError("At least one eval must be enabled.")

    fp32_predict, _fp32_fmt = _load_predict_fn(fp32_model_path)
    candidate_predict, _candidate_fmt = _load_predict_fn(candidate_model_path)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    max_samples = min(num_eval_samples, len(dataset))

    cosine_scores: list[float] = []
    fp32_latencies_ms: list[float] = []
    candidate_latencies_ms: list[float] = []
    argmax_supported = True
    argmax_reason = ""
    fp32_correct = 0
    candidate_correct = 0
    prediction_agreement = 0
    argmax_count = 0
    sample_count = 0

    with torch.no_grad():
        for x, y in loader:
            if sample_count >= max_samples:
                break

            remaining = max_samples - sample_count
            if x.shape[0] > remaining:
                x = x[:remaining]
                y = y[:remaining]

            x = x.float()
            t0 = time.perf_counter()
            fp32_out = fp32_predict(x)
            fp32_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            t1 = time.perf_counter()
            candidate_out = candidate_predict(x)
            candidate_latencies_ms.append((time.perf_counter() - t1) * 1000.0)

            batch_n = int(x.shape[0])
            sample_count += batch_n

            if eval_cosine:
                for row in range(batch_n):
                    fp32_flat = fp32_out[row].flatten().float()
                    candidate_flat = candidate_out[row].flatten().float()
                    cosine_scores.append(
                        F.cosine_similarity(
                            fp32_flat.unsqueeze(0),
                            candidate_flat.unsqueeze(0),
                        ).item()
                    )

            if eval_argmax and argmax_supported:
                try:
                    y_true = _labels_to_class_ids(y)
                    fp32_pred = _logits_to_class_ids(fp32_out)
                    candidate_pred = _logits_to_class_ids(candidate_out)
                except ValueError as exc:
                    argmax_supported = False
                    argmax_reason = str(exc)
                    continue

                fp32_correct += int((fp32_pred == y_true).sum().item())
                candidate_correct += int((candidate_pred == y_true).sum().item())
                prediction_agreement += int((fp32_pred == candidate_pred).sum().item())
                argmax_count += int(y_true.numel())

    if sample_count == 0:
        raise RuntimeError("No samples were evaluated.")

    result: dict = {
        "samples_evaluated": sample_count,
    }

    if not fp32_latencies_ms or not candidate_latencies_ms:
        raise RuntimeError("Latency eval failed: no latency samples collected.")

    fp32_latency_mean = sum(fp32_latencies_ms) / len(fp32_latencies_ms)
    candidate_latency_mean = (
        sum(candidate_latencies_ms) / len(candidate_latencies_ms)
    )
    latency_delta_ms = candidate_latency_mean - fp32_latency_mean
    latency_ratio = (
        candidate_latency_mean / fp32_latency_mean
        if fp32_latency_mean > 0
        else 0.0
    )
    result["fp32_latency_ms_mean"] = fp32_latency_mean
    result["candidate_latency_ms_mean"] = candidate_latency_mean
    result["latency_delta_ms"] = latency_delta_ms
    result["latency_ratio"] = latency_ratio

    if eval_cosine:
        if not cosine_scores:
            raise RuntimeError("Cosine eval enabled but no cosine scores computed.")
        result["cos_mean"] = sum(cosine_scores) / len(cosine_scores)
        result["cos_min"] = min(cosine_scores)

    if eval_argmax:
        result["argmax_supported"] = argmax_supported
        if argmax_supported:
            if argmax_count == 0:
                raise RuntimeError("Argmax eval enabled but no samples were scored.")
            fp32_top1 = fp32_correct / argmax_count
            candidate_top1 = candidate_correct / argmax_count
            agreement = prediction_agreement / argmax_count
            result["argmax_samples"] = argmax_count
            result["fp32_top1"] = fp32_top1
            result["candidate_top1"] = candidate_top1
            result["argmax_agreement"] = agreement
            result["argmax_delta"] = candidate_top1 - fp32_top1
        else:
            result["argmax_reason"] = argmax_reason

    return result
