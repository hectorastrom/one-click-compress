"""
bitsandbytes quantization backends.

Replaces nn.Linear layers with bitsandbytes quantized versions:
- Linear8bitLt for 8-bit quantization
- Linear4bit for 4-bit quantization (NF4 or FP4)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from .base import QuantizerBackend, QuantizationConfig, QuantizationResult

logger = logging.getLogger(__name__)


def _check_bnb_available() -> bool:
    """Check if bitsandbytes is installed and functional."""
    try:
        import bitsandbytes as bnb  # noqa: F401
        return True
    except ImportError:
        return False


def _replace_linear_with_8bit(model: nn.Module) -> nn.Module:
    """Replace all nn.Linear layers with bitsandbytes Linear8bitLt."""
    import bitsandbytes as bnb

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            has_bias = module.bias is not None
            new_layer = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=has_bias,
                has_fp16_weights=False,
                threshold=6.0,
            )
            # Copy weights
            new_layer.weight = nn.Parameter(module.weight.data.clone())
            if has_bias:
                new_layer.bias = nn.Parameter(module.bias.data.clone())
            setattr(model, name, new_layer)
        else:
            _replace_linear_with_8bit(module)

    return model


def _replace_linear_with_4bit(
    model: nn.Module,
    quant_type: str = "nf4",
    compute_dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """Replace all nn.Linear layers with bitsandbytes Linear4bit."""
    import bitsandbytes as bnb

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            has_bias = module.bias is not None
            new_layer = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=has_bias,
                quant_type=quant_type,
                compute_dtype=compute_dtype,
            )
            new_layer.weight = nn.Parameter(module.weight.data.clone())
            if has_bias:
                new_layer.bias = nn.Parameter(module.bias.data.clone())
            setattr(model, name, new_layer)
        else:
            _replace_linear_with_4bit(module, quant_type, compute_dtype)

    return model


class BitsAndBytes8bitQuantizer(QuantizerBackend):
    """bitsandbytes 8-bit quantization (LLM.int8())."""

    name = "bnb_int8"

    def quantize(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        calibration_data: Any | None = None,
    ) -> QuantizationResult:
        if not _check_bnb_available():
            return QuantizationResult(
                config=config,
                model=model,
                size_mb=self._measure_size_mb(model),
                error="bitsandbytes is not installed. Install with: pip install bitsandbytes",
            )

        try:
            model_copy = self._deep_copy_model(model)
            model_copy.eval()

            threshold = config.extra_params.get("threshold", 6.0)
            model_copy = _replace_linear_with_8bit(model_copy)

            size_mb = self._measure_size_mb(model_copy)

            return QuantizationResult(
                config=config,
                model=model_copy,
                size_mb=size_mb,
                metadata={
                    "method": "bnb_int8",
                    "threshold": threshold,
                },
            )
        except Exception as e:
            logger.error(f"bitsandbytes 8-bit quantization failed: {e}")
            return QuantizationResult(
                config=config,
                model=model,
                size_mb=self._measure_size_mb(model),
                error=str(e),
            )

    @classmethod
    def is_applicable(cls, model_report: dict) -> bool:
        if not _check_bnb_available():
            return False
        type_counts = model_report.get("layer_type_counts", {})
        return type_counts.get("Linear", 0) > 0


class BitsAndBytes4bitQuantizer(QuantizerBackend):
    """bitsandbytes 4-bit quantization (NF4/FP4)."""

    name = "bnb_int4"

    def quantize(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        calibration_data: Any | None = None,
    ) -> QuantizationResult:
        if not _check_bnb_available():
            return QuantizationResult(
                config=config,
                model=model,
                size_mb=self._measure_size_mb(model),
                error="bitsandbytes is not installed. Install with: pip install bitsandbytes",
            )

        try:
            model_copy = self._deep_copy_model(model)
            model_copy.eval()

            quant_type = config.extra_params.get("quant_type", "nf4")
            compute_dtype_str = config.extra_params.get("compute_dtype", "float16")
            compute_dtype = getattr(torch, compute_dtype_str, torch.float16)

            model_copy = _replace_linear_with_4bit(
                model_copy,
                quant_type=quant_type,
                compute_dtype=compute_dtype,
            )

            size_mb = self._measure_size_mb(model_copy)

            return QuantizationResult(
                config=config,
                model=model_copy,
                size_mb=size_mb,
                metadata={
                    "method": "bnb_int4",
                    "quant_type": quant_type,
                    "compute_dtype": compute_dtype_str,
                },
            )
        except Exception as e:
            logger.error(f"bitsandbytes 4-bit quantization failed: {e}")
            return QuantizationResult(
                config=config,
                model=model,
                size_mb=self._measure_size_mb(model),
                error=str(e),
            )

    @classmethod
    def is_applicable(cls, model_report: dict) -> bool:
        if not _check_bnb_available():
            return False
        type_counts = model_report.get("layer_type_counts", {})
        return type_counts.get("Linear", 0) > 0
