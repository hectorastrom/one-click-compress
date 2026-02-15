"""
PyTorch native quantization backends.

Implements three quantization modes:
- Dynamic INT8: torch.quantization.quantize_dynamic (no calibration needed)
- Static INT8: Fuse + prepare + calibrate + convert (needs calibration data)
- Float16: Simple .half() conversion
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.quantization import (
    quantize_dynamic,
    fuse_modules,
    prepare,
    convert,
    get_default_qconfig,
)

from .base import QuantizerBackend, QuantizationConfig, QuantizationResult

logger = logging.getLogger(__name__)


def _ensure_qengine() -> str:
    """Ensure a valid quantization engine is set. Returns the engine name."""
    supported = torch.backends.quantized.supported_engines
    current = torch.backends.quantized.engine
    if current != "none":
        return current
    # Prefer x86 (better for server), fall back to qnnpack (ARM/Mac)
    for engine in ("x86", "qnnpack"):
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    raise RuntimeError(f"No suitable quantization engine found. Supported: {supported}")


class DynamicInt8Quantizer(QuantizerBackend):
    """Dynamic INT8 quantization — no calibration needed."""

    name = "dynamic_int8"

    def quantize(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        calibration_data: Any | None = None,
    ) -> QuantizationResult:
        try:
            engine = _ensure_qengine()
            logger.info(f"Using quantization engine: {engine}")

            model_copy = self._deep_copy_model(model)
            model_copy.eval()

            # Determine which layer types to quantize
            layer_types = {nn.Linear}
            if config.extra_params.get("include_conv", False):
                layer_types.add(nn.Conv2d)

            quantized = quantize_dynamic(
                model_copy,
                qconfig_spec=layer_types,
                dtype=torch.qint8,
            )

            size_mb = self._measure_size_mb(quantized)

            return QuantizationResult(
                config=config,
                model=quantized,
                size_mb=size_mb,
                metadata={
                    "method": "dynamic_int8",
                    "quantized_layer_types": [t.__name__ for t in layer_types],
                },
            )
        except Exception as e:
            logger.error(f"Dynamic INT8 quantization failed: {e}")
            return QuantizationResult(
                config=config,
                model=model,
                size_mb=self._measure_size_mb(model),
                error=str(e),
            )

    @classmethod
    def is_applicable(cls, model_report: dict) -> bool:
        type_counts = model_report.get("layer_type_counts", {})
        has_linear = type_counts.get("Linear", 0) > 0
        has_conv = any("Conv" in k for k in type_counts)
        return has_linear or has_conv


class StaticInt8Quantizer(QuantizerBackend):
    """Static INT8 quantization with calibration — best for CNN models."""

    name = "static_int8"

    def quantize(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        calibration_data: Any | None = None,
    ) -> QuantizationResult:
        try:
            engine = _ensure_qengine()
            logger.info(f"Using quantization engine: {engine}")

            model_copy = self._deep_copy_model(model)
            model_copy.eval()

            # Attempt layer fusion for common CNN patterns
            model_copy = self._try_fuse(model_copy)

            # Set quantization config — use the active engine
            backend = config.extra_params.get("backend", engine)
            model_copy.qconfig = get_default_qconfig(backend)

            # Prepare for calibration
            prepared = prepare(model_copy)

            # Run calibration
            if calibration_data is not None:
                self._calibrate(prepared, calibration_data)
            else:
                logger.warning(
                    "No calibration data provided for static quantization. "
                    "Using random input for calibration (results may be suboptimal)."
                )
                self._calibrate_random(prepared)

            # Convert to quantized model
            quantized = convert(prepared)

            size_mb = self._measure_size_mb(quantized)

            return QuantizationResult(
                config=config,
                model=quantized,
                size_mb=size_mb,
                metadata={
                    "method": "static_int8",
                    "backend": backend,
                    "calibrated": calibration_data is not None,
                },
            )
        except Exception as e:
            logger.error(f"Static INT8 quantization failed: {e}")
            return QuantizationResult(
                config=config,
                model=model,
                size_mb=self._measure_size_mb(model),
                error=str(e),
            )

    @classmethod
    def is_applicable(cls, model_report: dict) -> bool:
        # Static quantization works best on CNN models
        arch = model_report.get("architecture_hint", "")
        return arch in ("cnn", "mixed")

    def _try_fuse(self, model: nn.Module) -> nn.Module:
        """Try to fuse Conv-BN-ReLU sequences. Fails gracefully."""
        try:
            # Attempt common ResNet-style fusion patterns
            for name, module in model.named_modules():
                if isinstance(module, nn.Sequential):
                    # Look for Conv2d -> BatchNorm2d -> ReLU patterns
                    children = list(module.named_children())
                    fuse_list = []
                    i = 0
                    while i < len(children):
                        group = [children[i][0]]
                        if (
                            isinstance(children[i][1], nn.Conv2d)
                            and i + 1 < len(children)
                            and isinstance(children[i + 1][1], nn.BatchNorm2d)
                        ):
                            group.append(children[i + 1][0])
                            if (
                                i + 2 < len(children)
                                and isinstance(children[i + 2][1], nn.ReLU)
                            ):
                                group.append(children[i + 2][0])
                                i += 3
                            else:
                                i += 2
                            if len(group) >= 2:
                                fuse_list.append(group)
                        else:
                            i += 1

                    if fuse_list:
                        fuse_modules(module, fuse_list, inplace=True)
        except Exception as e:
            logger.debug(f"Layer fusion failed (non-critical): {e}")

        return model

    def _calibrate(self, model: nn.Module, data_loader: Any) -> None:
        """Run calibration using provided data loader."""
        model.eval()
        n_batches = 0
        max_batches = 50  # limit calibration for speed
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                model(inputs)
                n_batches += 1
                if n_batches >= max_batches:
                    break

    def _calibrate_random(self, model: nn.Module) -> None:
        """Calibrate with random data as a fallback."""
        model.eval()
        with torch.no_grad():
            for _ in range(20):
                dummy = torch.randn(1, 3, 224, 224)
                try:
                    model(dummy)
                except Exception:
                    # Try smaller input
                    dummy = torch.randn(1, 3, 32, 32)
                    model(dummy)


class Float16Quantizer(QuantizerBackend):
    """Float16 (half-precision) conversion."""

    name = "fp16"

    def quantize(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        calibration_data: Any | None = None,
    ) -> QuantizationResult:
        try:
            model_copy = self._deep_copy_model(model)
            model_copy.eval()

            # Convert to half precision
            model_copy = model_copy.half()

            size_mb = self._measure_size_mb(model_copy)

            return QuantizationResult(
                config=config,
                model=model_copy,
                size_mb=size_mb,
                metadata={"method": "fp16"},
            )
        except Exception as e:
            logger.error(f"FP16 conversion failed: {e}")
            return QuantizationResult(
                config=config,
                model=model,
                size_mb=self._measure_size_mb(model),
                error=str(e),
            )

    @classmethod
    def is_applicable(cls, model_report: dict) -> bool:
        # FP16 is always applicable
        return True


class MixedPrecisionQuantizer(QuantizerBackend):
    """
    Mixed-precision quantizer — keeps sensitive layers at FP32, quantizes the rest to INT8.

    Uses per_layer_config to protect sensitive layers:
    - "fp32" / "fp16": keep at full/half precision (sensitive layers)
    - "int8": apply dynamic INT8 quantization (robust layers)

    Since mixing fp16 and int8 causes dtype conflicts in forward pass,
    this quantizer uses fp32 for protected layers and int8 for the rest.
    The "fp16" directive is treated as "fp32" to avoid mixed-dtype issues
    (the FP16 uniform quantizer handles the pure fp16 case).
    """

    name = "mixed_precision"

    def quantize(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        calibration_data: Any | None = None,
    ) -> QuantizationResult:
        try:
            engine = _ensure_qengine()
            logger.info(f"Mixed precision using quantization engine: {engine}")

            model_copy = self._deep_copy_model(model)
            model_copy.eval()

            per_layer = config.per_layer_config or {}
            default_precision = config.extra_params.get("default_precision", "int8")

            # Categorize layers: protected (fp32) vs quantized (int8)
            protected_layers = set()  # keep at fp32
            int8_layers = []

            for mod_name, module in model_copy.named_modules():
                if mod_name == "":
                    continue
                has_params = any(True for _ in module.parameters(recurse=False))
                if not has_params:
                    continue

                layer_cfg = per_layer.get(mod_name, {})
                precision = layer_cfg.get("precision", layer_cfg.get("bits", default_precision))

                # Normalize
                if isinstance(precision, int):
                    precision = "int8" if precision <= 8 else "fp32"
                if precision in ("fp32", "fp16", "half", "16", "32", "skip"):
                    protected_layers.add(mod_name)
                else:
                    int8_layers.append(mod_name)

            # Build a selective qconfig_spec: quantize only nn.Linear layers
            # that are NOT in the protected set
            # We use quantize_dynamic with a custom set of modules
            modules_to_quantize = set()
            modules_dict = dict(model_copy.named_modules())
            for layer_name in int8_layers:
                module = modules_dict.get(layer_name)
                if module is not None and isinstance(module, nn.Linear):
                    modules_to_quantize.add(nn.Linear)

            if modules_to_quantize:
                # To selectively quantize, we temporarily replace protected layers
                # with a wrapper that prevents quantization, then restore them.
                # Simpler approach: quantize all Linear, then replace protected back.

                # Save protected layer state dicts
                protected_states = {}
                protected_modules = {}
                for pname in protected_layers:
                    mod = modules_dict.get(pname)
                    if mod is not None and isinstance(mod, nn.Linear):
                        protected_states[pname] = {
                            "in_features": mod.in_features,
                            "out_features": mod.out_features,
                            "bias": mod.bias is not None,
                            "weight": mod.weight.data.clone(),
                            "bias_data": mod.bias.data.clone() if mod.bias is not None else None,
                        }

                # Quantize all Linear layers
                model_copy = quantize_dynamic(
                    model_copy,
                    qconfig_spec={nn.Linear},
                    dtype=torch.qint8,
                )

                # Restore protected layers back to fp32 Linear
                for pname, state in protected_states.items():
                    # Navigate to parent and replace the quantized module
                    parts = pname.rsplit(".", 1)
                    if len(parts) == 2:
                        parent_name, child_name = parts
                        parent = dict(model_copy.named_modules()).get(parent_name)
                    else:
                        parent = model_copy
                        child_name = pname

                    if parent is not None:
                        restored = nn.Linear(
                            state["in_features"],
                            state["out_features"],
                            bias=state["bias"],
                        )
                        restored.weight.data = state["weight"]
                        if state["bias_data"] is not None:
                            restored.bias.data = state["bias_data"]
                        setattr(parent, child_name, restored)

            size_mb = self._measure_size_mb(model_copy)

            return QuantizationResult(
                config=config,
                model=model_copy,
                size_mb=size_mb,
                metadata={
                    "method": "mixed_precision",
                    "protected_layers": list(protected_layers)[:20],
                    "quantized_layers_count": len(int8_layers),
                    "protected_layers_count": len(protected_layers),
                    "per_layer_config": per_layer,
                    "default_precision": default_precision,
                },
            )
        except Exception as e:
            logger.error(f"Mixed precision quantization failed: {e}")
            return QuantizationResult(
                config=config,
                model=model,
                size_mb=self._measure_size_mb(model),
                error=str(e),
            )

    @classmethod
    def is_applicable(cls, model_report: dict) -> bool:
        type_counts = model_report.get("layer_type_counts", {})
        quantizable = sum(
            type_counts.get(t, 0) for t in ("Linear", "Conv2d", "Conv1d")
        )
        return quantizable >= 2
