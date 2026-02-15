"""
Model Scanner — Introspect any PyTorch nn.Module and produce a structured report.

Analyzes architecture, layer types, parameter counts, weight distributions,
fits a Beta distribution per layer to guide quantization sensitivity decisions,
and determines which quantization methods are applicable.
"""

from __future__ import annotations

import io
import logging
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Try to import scipy for Beta fitting; fall back to method-of-moments
try:
    from scipy.stats import beta as beta_dist
    from scipy.stats import kurtosis as scipy_kurtosis
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.info("scipy not installed — using method-of-moments for Beta distribution fitting.")


@dataclass
class LayerInfo:
    """Information about a single layer in the model."""
    name: str
    layer_type: str
    param_count: int
    dtype: str
    has_weights: bool
    weight_stats: dict[str, float] | None = None  # mean, std, min, max, sparsity


@dataclass
class ModelReport:
    """Full scan report for a model."""
    model_name: str
    total_params: int
    total_size_mb: float
    architecture_hint: str  # "cnn", "transformer", "mixed", "unknown"
    layers: list[LayerInfo] = field(default_factory=list)
    layer_type_counts: dict[str, int] = field(default_factory=dict)
    applicable_methods: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict for the LLM agent."""
        d = asdict(self)
        # Trim full layer details for the LLM — only include top 20 largest layers
        sorted_layers = sorted(d["layers"], key=lambda l: l["param_count"], reverse=True)
        d["layers"] = sorted_layers[:20]
        d["total_layer_count"] = len(self.layers)

        # Build a dedicated per-layer sensitivity table for the LLM
        # This is the key data for mixed_precision decisions
        sensitivity_table = []
        for layer in self.layers:
            if layer.weight_stats and "quantization_sensitivity" in layer.weight_stats:
                s = layer.weight_stats
                sensitivity_table.append({
                    "name": layer.name,
                    "type": layer.layer_type,
                    "params": layer.param_count,
                    "sensitivity": s["quantization_sensitivity"],
                    "beta_alpha": s.get("beta_alpha", 1.0),
                    "beta_beta": s.get("beta_beta", 1.0),
                    "kurtosis": s.get("kurtosis", 0.0),
                })

        # Sort by sensitivity descending — most sensitive first
        sensitivity_table.sort(key=lambda x: x["sensitivity"], reverse=True)
        d["sensitivity_table"] = sensitivity_table

        # Summary stats for the LLM
        if sensitivity_table:
            sens_values = [e["sensitivity"] for e in sensitivity_table]
            d["sensitivity_summary"] = {
                "total_analyzed_layers": len(sensitivity_table),
                "avg_sensitivity": round(sum(sens_values) / len(sens_values), 3),
                "max_sensitivity": round(max(sens_values), 3),
                "min_sensitivity": round(min(sens_values), 3),
                "high_sensitivity_layers": [
                    e["name"] for e in sensitivity_table if e["sensitivity"] > 0.5
                ],
                "low_sensitivity_layers": [
                    e["name"] for e in sensitivity_table if e["sensitivity"] < 0.1
                ][:10],  # cap for brevity
            }

        return d

    def to_summary_string(self) -> str:
        """Produce a concise human-readable summary."""
        lines = [
            f"Model: {self.model_name}",
            f"Total parameters: {self.total_params:,}",
            f"Model size: {self.total_size_mb:.2f} MB",
            f"Architecture: {self.architecture_hint}",
            f"Total layers: {len(self.layers)}",
            f"Layer types: {self.layer_type_counts}",
            f"Applicable quantization methods: {self.applicable_methods}",
        ]
        # Show per-layer sensitivity highlights
        layers_with_sens = [
            l for l in self.layers
            if l.weight_stats and "quantization_sensitivity" in l.weight_stats
        ]
        if layers_with_sens:
            sorted_by_sens = sorted(
                layers_with_sens,
                key=lambda l: l.weight_stats["quantization_sensitivity"],
                reverse=True,
            )
            lines.append("Top sensitive layers (quantization_sensitivity):")
            for l in sorted_by_sens[:5]:
                s = l.weight_stats
                lines.append(
                    f"  {l.name}: sens={s['quantization_sensitivity']:.3f}, "
                    f"beta(a={s['beta_alpha']:.2f}, b={s['beta_beta']:.2f}), "
                    f"kurtosis={s['kurtosis']:.2f}"
                )
        if self.summary:
            lines.append(f"Notes: {self.summary}")
        return "\n".join(lines)


def _fit_beta_scipy(normalized: np.ndarray) -> tuple[float, float, float]:
    """Fit Beta distribution using scipy MLE. Returns (alpha, beta, ks_error)."""
    # Clamp to (0, 1) open interval for Beta fitting
    eps = 1e-6
    clamped = np.clip(normalized, eps, 1.0 - eps)
    try:
        a, b, loc, scale = beta_dist.fit(clamped, floc=0, fscale=1)
        # KS goodness-of-fit: lower is better
        from scipy.stats import kstest
        ks_stat, _ = kstest(clamped, "beta", args=(a, b))
        return float(a), float(b), float(ks_stat)
    except Exception:
        return _fit_beta_moments(normalized)


def _fit_beta_moments(normalized: np.ndarray) -> tuple[float, float, float]:
    """Fit Beta distribution via method of moments. Returns (alpha, beta, error)."""
    eps = 1e-6
    clamped = np.clip(normalized, eps, 1.0 - eps)
    mean = float(np.mean(clamped))
    var = float(np.var(clamped))
    if var <= 0 or mean <= 0 or mean >= 1:
        return 1.0, 1.0, 1.0  # uniform fallback

    common = (mean * (1 - mean) / max(var, eps)) - 1
    common = max(common, eps)
    alpha = mean * common
    beta = (1 - mean) * common

    # Clamp to reasonable range
    alpha = max(min(alpha, 100.0), 0.01)
    beta = max(min(beta, 100.0), 0.01)

    # Rough error estimate: difference between empirical and fitted variance
    fitted_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    error = abs(var - fitted_var) / max(var, eps)
    return float(alpha), float(beta), float(min(error, 1.0))


def _compute_quantization_sensitivity(
    beta_alpha: float,
    beta_beta: float,
    kurt: float,
    sparsity: float,
) -> float:
    """
    Compute a quantization sensitivity score (0-1).

    Higher score = layer is more sensitive to quantization = needs higher precision.

    Factors:
    - High kurtosis (heavy tails / outliers) -> more sensitive
    - Low alpha or beta (skewed / U-shaped distribution) -> more sensitive
    - High sparsity -> less sensitive (many zeros compress well)
    """
    import math

    # Handle NaN/inf values (e.g., from constant or degenerate weight tensors)
    if math.isnan(kurt) or math.isinf(kurt):
        kurt = 0.0
    if math.isnan(beta_alpha) or math.isinf(beta_alpha):
        beta_alpha = 1.0
    if math.isnan(beta_beta) or math.isinf(beta_beta):
        beta_beta = 1.0
    if math.isnan(sparsity):
        sparsity = 0.0

    # Kurtosis factor: excess kurtosis > 3 is heavy-tailed
    kurt_score = min(max(kurt, 0) / 10.0, 1.0)

    # Beta shape factor: both alpha,beta > 2 means well-behaved bell shape
    # alpha,beta < 1 means U-shape or extreme skew
    min_ab = min(beta_alpha, beta_beta)
    if min_ab >= 2.0:
        shape_score = 0.0  # well-behaved
    elif min_ab >= 1.0:
        shape_score = 0.3  # moderate
    elif min_ab >= 0.5:
        shape_score = 0.6  # concerning
    else:
        shape_score = 1.0  # very sensitive (U-shape / extreme)

    # Sparsity factor: more sparse = less sensitive
    sparsity_bonus = sparsity * 0.3

    sensitivity = 0.4 * kurt_score + 0.5 * shape_score - sparsity_bonus
    return round(max(min(sensitivity, 1.0), 0.0), 3)


def _compute_weight_stats(param: torch.Tensor) -> dict[str, Any]:
    """
    Compute statistics on a weight tensor including Beta distribution fit.

    Returns a dict with basic stats, Beta parameters, kurtosis,
    and a quantization_sensitivity score.
    """
    with torch.no_grad():
        flat = param.float().flatten()
        flat_np = flat.cpu().numpy()

        # Basic stats
        w_mean = flat.mean().item()
        w_std = flat.std().item() if flat.numel() > 1 else 0.0
        w_min = flat.min().item()
        w_max = flat.max().item()
        sparsity = (flat == 0).float().mean().item()

        # Normalize to [0, 1] for Beta fitting
        w_range = w_max - w_min
        if w_range > 1e-10 and len(flat_np) > 10:
            normalized = (flat_np - w_min) / w_range

            # Fit Beta distribution
            if HAS_SCIPY:
                beta_alpha, beta_beta, fit_error = _fit_beta_scipy(normalized)
            else:
                beta_alpha, beta_beta, fit_error = _fit_beta_moments(normalized)
        else:
            # Degenerate case (constant or near-constant weights, e.g., BatchNorm init)
            beta_alpha, beta_beta, fit_error = 1.0, 1.0, 1.0

        # Kurtosis (excess kurtosis: normal = 0)
        import math
        if HAS_SCIPY and len(flat_np) > 10 and w_std > 1e-10:
            try:
                kurt = float(scipy_kurtosis(flat_np, fisher=True))
                if math.isnan(kurt) or math.isinf(kurt):
                    kurt = 0.0
            except Exception:
                kurt = 0.0
        else:
            n = len(flat_np)
            if n > 3 and w_std > 1e-10:
                m4 = np.mean((flat_np - w_mean) ** 4)
                kurt = float(m4 / (w_std ** 4) - 3.0)
                if math.isnan(kurt) or math.isinf(kurt):
                    kurt = 0.0
            else:
                kurt = 0.0

        # Quantization sensitivity score
        sensitivity = _compute_quantization_sensitivity(
            beta_alpha, beta_beta, kurt, sparsity
        )

        return {
            "mean": w_mean,
            "std": w_std,
            "min": w_min,
            "max": w_max,
            "sparsity": sparsity,
            "shape": list(param.shape),
            "beta_alpha": round(beta_alpha, 4),
            "beta_beta": round(beta_beta, 4),
            "beta_fit_error": round(fit_error, 4),
            "kurtosis": round(kurt, 4),
            "quantization_sensitivity": sensitivity,
        }


def _classify_architecture(type_counts: dict[str, int]) -> str:
    """Heuristic to classify model architecture based on layer types."""
    conv_layers = sum(v for k, v in type_counts.items() if "Conv" in k)
    linear_layers = type_counts.get("Linear", 0)
    attention_layers = sum(
        v for k, v in type_counts.items()
        if "Attention" in k or "MultiheadAttention" in k
    )
    norm_layers = sum(v for k, v in type_counts.items() if "Norm" in k)

    if attention_layers > 0 or (linear_layers > conv_layers and norm_layers > 2):
        return "transformer"
    elif conv_layers > 0 and conv_layers >= linear_layers:
        return "cnn"
    elif conv_layers > 0 and linear_layers > 0:
        return "mixed"
    else:
        return "unknown"


def _determine_applicable_methods(
    arch_hint: str, type_counts: dict[str, int]
) -> list[str]:
    """Determine which quantization methods can be applied to this model."""
    methods = []

    # Dynamic INT8 — works on models with Linear and/or Conv layers
    has_linear = type_counts.get("Linear", 0) > 0
    has_conv = any("Conv" in k for k in type_counts)
    if has_linear or has_conv:
        methods.append("dynamic_int8")

    # Static INT8 — works on most models but needs calibration and fusion support
    if arch_hint == "cnn":
        methods.append("static_int8")

    # FP16 — always applicable
    methods.append("fp16")

    # bitsandbytes — replaces Linear layers
    if has_linear:
        methods.append("bnb_int8")
        methods.append("bnb_int4")

    # Mixed precision — available when there are multiple quantizable layers
    quantizable_layer_count = sum(
        type_counts.get(t, 0) for t in ("Linear", "Conv2d", "Conv1d")
    )
    if quantizable_layer_count >= 2:
        methods.append("mixed_precision")

    return methods


def _estimate_model_size_mb(model: nn.Module) -> float:
    """Estimate model size by saving to an in-memory buffer."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = buffer.tell()
    return size_bytes / (1024 * 1024)


def scan_model(model: nn.Module, model_name: str = "unknown") -> ModelReport:
    """
    Scan a PyTorch model and produce a structured report.

    Args:
        model: The PyTorch model to scan.
        model_name: A human-readable name for the model.

    Returns:
        A ModelReport with full analysis.
    """
    model.eval()

    layers: list[LayerInfo] = []
    type_counter: Counter = Counter()
    total_params = 0

    for name, module in model.named_modules():
        # Skip the root module itself
        if name == "":
            continue

        # Get the class name (e.g., "Conv2d", "Linear", "BatchNorm2d")
        layer_type = module.__class__.__name__
        type_counter[layer_type] += 1

        # Count parameters directly owned by this module (not children)
        own_params = sum(p.numel() for p in module.parameters(recurse=False))
        total_params_check = own_params  # for this layer

        # Determine dtype from first parameter
        first_param = next(module.parameters(recurse=False), None)
        dtype_str = str(first_param.dtype) if first_param is not None else "none"
        has_weights = first_param is not None

        # Weight stats for layers that have parameters
        weight_stats = None
        if has_weights and own_params > 0:
            # Use the first parameter (typically the weight matrix)
            weight_stats = _compute_weight_stats(first_param)

        layers.append(LayerInfo(
            name=name,
            layer_type=layer_type,
            param_count=own_params,
            dtype=dtype_str,
            has_weights=has_weights,
            weight_stats=weight_stats,
        ))

    # Total param count from model directly
    total_params = sum(p.numel() for p in model.parameters())
    total_size_mb = _estimate_model_size_mb(model)

    type_counts = dict(type_counter)
    arch_hint = _classify_architecture(type_counts)
    applicable = _determine_applicable_methods(arch_hint, type_counts)

    # Build summary with sensitivity info
    top_layers = sorted(layers, key=lambda l: l.param_count, reverse=True)[:5]
    top_summary = ", ".join(
        f"{l.name} ({l.layer_type}, {l.param_count:,} params)" for l in top_layers
    )

    # Sensitivity summary
    layers_with_sens = [
        l for l in layers
        if l.weight_stats and "quantization_sensitivity" in l.weight_stats
    ]
    if layers_with_sens:
        avg_sens = sum(
            l.weight_stats["quantization_sensitivity"] for l in layers_with_sens
        ) / len(layers_with_sens)
        high_sens = [
            l for l in layers_with_sens
            if l.weight_stats["quantization_sensitivity"] > 0.5
        ]
        summary = (
            f"Largest layers: {top_summary}. "
            f"Avg quantization sensitivity: {avg_sens:.3f}. "
            f"High-sensitivity layers ({len(high_sens)}/{len(layers_with_sens)}): "
            + ", ".join(l.name for l in high_sens[:5])
        )
    else:
        summary = f"Largest layers: {top_summary}"

    report = ModelReport(
        model_name=model_name,
        total_params=total_params,
        total_size_mb=total_size_mb,
        architecture_hint=arch_hint,
        layers=layers,
        layer_type_counts=type_counts,
        applicable_methods=applicable,
        summary=summary,
    )

    return report
