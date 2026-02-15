"""
Quantizer base class and registry.

All quantization backends implement QuantizerBackend.
The QuantizerRegistry maps method names to their implementations.
"""

from __future__ import annotations

import copy
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class QuantizationConfig:
    """Configuration for a single quantization run."""
    method: str                          # e.g. "dynamic_int8", "static_int8", "fp16", "bnb_int8", "bnb_int4", "mixed_precision"
    target_layers: str = "all"           # "all" or comma-separated layer names
    extra_params: dict[str, Any] = field(default_factory=dict)
    per_layer_config: dict[str, dict[str, Any]] | None = None
    # per_layer_config maps layer names to per-layer overrides, e.g.:
    # {"layer4.0.conv2": {"bits": 8, "method": "dynamic_int8"},
    #  "fc": {"bits": 4, "method": "fp16"}}

    def to_dict(self) -> dict[str, Any]:
        d = {
            "method": self.method,
            "target_layers": self.target_layers,
            "extra_params": self.extra_params,
        }
        if self.per_layer_config:
            d["per_layer_config"] = self.per_layer_config
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QuantizationConfig:
        return cls(
            method=d.get("method", ""),
            target_layers=d.get("target_layers", "all"),
            extra_params=d.get("extra_params", {}),
            per_layer_config=d.get("per_layer_config"),
        )


@dataclass
class QuantizationResult:
    """Result of applying a quantization config."""
    config: QuantizationConfig
    model: nn.Module
    size_mb: float
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


class QuantizerBackend(ABC):
    """Abstract base class for quantization backends."""

    # Human-readable name for this backend
    name: str = "base"

    @abstractmethod
    def quantize(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        calibration_data: Any | None = None,
    ) -> QuantizationResult:
        """
        Apply quantization to a model.

        Args:
            model: The original PyTorch model (will be deep-copied internally).
            config: Quantization configuration.
            calibration_data: Optional calibration DataLoader or tensor.

        Returns:
            QuantizationResult with the quantized model and metadata.
        """
        ...

    @classmethod
    @abstractmethod
    def is_applicable(cls, model_report: dict) -> bool:
        """
        Check if this backend can handle the model described by the report.

        Args:
            model_report: The ModelReport as a dict.

        Returns:
            True if this backend is applicable.
        """
        ...

    @staticmethod
    def _deep_copy_model(model: nn.Module) -> nn.Module:
        """Safely deep-copy a model on CPU (quantization ops require CPU)."""
        # Remember original device so we don't permanently move the source model
        original_device = next(model.parameters()).device
        model_cpu = model.cpu()
        copied = copy.deepcopy(model_cpu)
        # Move the original back to its device
        model.to(original_device)
        return copied

    @staticmethod
    def _measure_size_mb(model: nn.Module) -> float:
        """Measure serialized model size in MB."""
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.tell() / (1024 * 1024)


class QuantizerRegistry:
    """Registry mapping method names to QuantizerBackend instances."""

    def __init__(self) -> None:
        self._backends: dict[str, QuantizerBackend] = {}

    def register(self, method_name: str, backend: QuantizerBackend) -> None:
        """Register a backend for a given method name."""
        self._backends[method_name] = backend

    def get(self, method_name: str) -> QuantizerBackend | None:
        """Get the backend for a method name."""
        return self._backends.get(method_name)

    def available_methods(self) -> list[str]:
        """List all registered method names."""
        return list(self._backends.keys())

    def get_applicable(self, model_report: dict) -> dict[str, QuantizerBackend]:
        """Return only backends that are applicable to the given model."""
        return {
            name: backend
            for name, backend in self._backends.items()
            if backend.is_applicable(model_report)
        }


def build_default_registry() -> QuantizerRegistry:
    """Build a registry with all available quantization backends."""
    from .pytorch_native import (
        DynamicInt8Quantizer, StaticInt8Quantizer, Float16Quantizer,
        MixedPrecisionQuantizer,
    )
    from .bnb import BitsAndBytes8bitQuantizer, BitsAndBytes4bitQuantizer

    registry = QuantizerRegistry()
    registry.register("dynamic_int8", DynamicInt8Quantizer())
    registry.register("static_int8", StaticInt8Quantizer())
    registry.register("fp16", Float16Quantizer())
    registry.register("bnb_int8", BitsAndBytes8bitQuantizer())
    registry.register("bnb_int4", BitsAndBytes4bitQuantizer())
    registry.register("mixed_precision", MixedPrecisionQuantizer())
    return registry
