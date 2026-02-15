"""
Visualization — show per-image prediction examples comparing models.

Creates a grid of sample images with predictions from the baseline and
each quantized model, highlighting correct (green) and incorrect (red).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ImageNet normalization (used in our transforms)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized image tensor back to displayable [0,1] numpy array."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return img


def _predict_batch(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    """Run a batch through a model. Returns (predicted_classes, confidences)."""
    from .evaluator import _get_eval_device

    # Use appropriate device (quantized models must stay on CPU)
    eval_device = _get_eval_device(model, device)
    model = model.to(eval_device)
    model.eval()
    with torch.no_grad():
        inputs = images.to(eval_device)
        # Handle dtype mismatch
        first_param = next(model.parameters(), None)
        if first_param is not None and first_param.dtype == torch.float16:
            inputs = inputs.half()
        outputs = model(inputs)
        probs = torch.softmax(outputs.float(), dim=1)
        confidences, predictions = probs.max(dim=1)
    return predictions.cpu().tolist(), confidences.cpu().tolist()


def generate_prediction_examples(
    models: dict[str, nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
    n_samples: int = 16,
    save_path: str = "results/examples.png",
) -> None:
    """
    Generate a visual grid comparing predictions across models.

    Rows = sample images, Columns = models.
    Each cell shows the predicted class and confidence.
    Green border = correct, Red border = incorrect.

    Args:
        models: Dict mapping config_name -> nn.Module.
        test_loader: DataLoader yielding (images, labels).
        device: Device to run inference on.
        class_names: List of class names (index -> name). Defaults to CIFAR-10.
        n_samples: Number of sample images to show.
        save_path: Where to save the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if class_names is None:
        class_names = CIFAR10_CLASSES

    # Collect sample images and labels
    all_images, all_labels = [], []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)
        if sum(img.shape[0] for img in all_images) >= n_samples:
            break

    all_images = torch.cat(all_images, dim=0)[:n_samples]
    all_labels = torch.cat(all_labels, dim=0)[:n_samples].tolist()

    # Get predictions from each model
    model_names = list(models.keys())
    model_preds: dict[str, tuple[list[int], list[float]]] = {}
    for name, model in models.items():
        try:
            model = model.to(device)
            preds, confs = _predict_batch(model, all_images, device)
            model_preds[name] = (preds, confs)
        except Exception as e:
            logger.warning(f"Could not get predictions from {name}: {e}")
            model_preds[name] = ([-1] * n_samples, [0.0] * n_samples)

    n_models = len(model_names)

    # Create figure: n_samples rows x (1 + n_models) columns
    # First column = original image + true label, remaining = model predictions
    fig_width = 2.5 * (1 + n_models)
    fig_height = 2.2 * n_samples
    fig, axes = plt.subplots(
        n_samples, 1 + n_models,
        figsize=(fig_width, fig_height),
    )

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    # Column headers
    headers = ["Image (true label)"] + model_names
    for col, header in enumerate(headers):
        axes[0, col].set_title(header, fontsize=10, fontweight="bold", pad=8)

    for row in range(n_samples):
        img_np = _denormalize(all_images[row])
        true_label = all_labels[row]
        true_name = class_names[true_label] if true_label < len(class_names) else f"class_{true_label}"

        # Column 0: original image with true label
        ax = axes[row, 0]
        ax.imshow(img_np)
        ax.set_ylabel(f"#{row}", fontsize=8, rotation=0, labelpad=20)
        ax.set_xlabel(f"True: {true_name}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        # Blue border for ground truth
        for spine in ax.spines.values():
            spine.set_edgecolor("#3498db")
            spine.set_linewidth(2)

        # Remaining columns: each model's prediction
        for col, mname in enumerate(model_names, start=1):
            ax = axes[row, col]
            ax.imshow(img_np)
            ax.set_xticks([])
            ax.set_yticks([])

            preds, confs = model_preds[mname]
            pred_idx = preds[row]
            conf = confs[row]

            if pred_idx < 0:
                pred_name = "ERROR"
                correct = False
            else:
                pred_name = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"
                correct = (pred_idx == true_label)

            # Color border based on correct/incorrect
            color = "#2ecc71" if correct else "#e74c3c"
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

            # Label
            symbol = "✓" if correct else "✗"
            ax.set_xlabel(
                f"{symbol} {pred_name} ({conf:.1%})",
                fontsize=8,
                color=color,
                fontweight="bold",
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Prediction examples saved to {save_path}")


def generate_failure_analysis(
    models: dict[str, nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
    n_correct: int = 8,
    n_incorrect: int = 8,
    save_path: str = "results/failure_analysis.png",
) -> None:
    """
    Generate a focused comparison showing samples where quantization
    changed the prediction — cases where baseline is correct but
    quantized is wrong, and vice versa.

    Args:
        models: Dict mapping config_name -> nn.Module.
                Must include a 'baseline_fp32' entry.
        test_loader: DataLoader yielding (images, labels).
        device: Device.
        class_names: Class name list.
        n_correct: Show this many images where baseline+quantized agree (correct).
        n_incorrect: Show this many images where quantization caused an error.
        save_path: Where to save.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if class_names is None:
        class_names = CIFAR10_CLASSES

    # Find baseline model
    baseline_key = None
    for k in models:
        if "baseline" in k.lower():
            baseline_key = k
            break
    if baseline_key is None:
        baseline_key = list(models.keys())[0]

    quantized_models = {k: v for k, v in models.items() if k != baseline_key}
    if not quantized_models:
        logger.warning("No quantized models to compare against baseline.")
        return

    # Collect predictions on all test images
    all_images, all_labels = [], []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)
    all_images_t = torch.cat(all_images, dim=0)
    all_labels_list = torch.cat(all_labels, dim=0).tolist()

    # Baseline predictions
    baseline_model = models[baseline_key].to(device)
    base_preds, base_confs = _predict_batch(baseline_model, all_images_t, device)

    # Find interesting cases for each quantized model
    for qname, qmodel in quantized_models.items():
        try:
            qmodel = qmodel.to(device)
            q_preds, q_confs = _predict_batch(qmodel, all_images_t, device)
        except Exception as e:
            logger.warning(f"Skipping failure analysis for {qname}: {e}")
            continue

        # Categories:
        # 1. Both correct
        # 2. Baseline correct, quantized wrong (quantization hurt)
        # 3. Baseline wrong, quantized correct (quantization helped — rare)
        # 4. Both wrong
        both_correct, quant_broke, quant_fixed, both_wrong = [], [], [], []

        for i in range(len(all_labels_list)):
            true_l = all_labels_list[i]
            b_correct = (base_preds[i] == true_l)
            q_correct = (q_preds[i] == true_l)

            entry = (i, true_l, base_preds[i], base_confs[i], q_preds[i], q_confs[i])
            if b_correct and q_correct:
                both_correct.append(entry)
            elif b_correct and not q_correct:
                quant_broke.append(entry)
            elif not b_correct and q_correct:
                quant_fixed.append(entry)
            else:
                both_wrong.append(entry)

        # Build the figure
        show_broke = quant_broke[:n_incorrect]
        show_correct = both_correct[:n_correct]
        sections = []
        if show_broke:
            sections.append(("Quantization BROKE prediction", show_broke))
        if show_correct:
            sections.append(("Both models CORRECT", show_correct))

        total_rows = sum(len(s[1]) for s in sections)
        if total_rows == 0:
            continue

        fig, axes = plt.subplots(total_rows, 3, figsize=(9, 2.2 * total_rows))
        if total_rows == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(
            f"Failure Analysis: {baseline_key} vs {qname}",
            fontsize=13, fontweight="bold", y=1.02,
        )

        row_idx = 0
        for section_title, entries in sections:
            for j, (img_idx, true_l, b_pred, b_conf, q_pred, q_conf) in enumerate(entries):
                img_np = _denormalize(all_images_t[img_idx])
                true_name = class_names[true_l] if true_l < len(class_names) else str(true_l)

                # Column 0: image + true label
                ax = axes[row_idx, 0]
                ax.imshow(img_np)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel(f"True: {true_name}", fontsize=8, fontweight="bold")
                if j == 0:
                    ax.set_title(section_title, fontsize=9, fontweight="bold", color="#8e44ad", pad=6)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#3498db")
                    spine.set_linewidth(2)

                # Column 1: baseline prediction
                ax = axes[row_idx, 1]
                ax.imshow(img_np)
                ax.set_xticks([]); ax.set_yticks([])
                b_name = class_names[b_pred] if b_pred < len(class_names) else str(b_pred)
                b_ok = (b_pred == true_l)
                b_color = "#2ecc71" if b_ok else "#e74c3c"
                ax.set_xlabel(f"{'✓' if b_ok else '✗'} {b_name} ({b_conf:.1%})", fontsize=8, color=b_color, fontweight="bold")
                for spine in ax.spines.values():
                    spine.set_edgecolor(b_color)
                    spine.set_linewidth(2)
                if j == 0:
                    ax.set_title(baseline_key, fontsize=9, fontweight="bold", pad=6)

                # Column 2: quantized prediction
                ax = axes[row_idx, 2]
                ax.imshow(img_np)
                ax.set_xticks([]); ax.set_yticks([])
                q_name = class_names[q_pred] if q_pred < len(class_names) else str(q_pred)
                q_ok = (q_pred == true_l)
                q_color = "#2ecc71" if q_ok else "#e74c3c"
                ax.set_xlabel(f"{'✓' if q_ok else '✗'} {q_name} ({q_conf:.1%})", fontsize=8, color=q_color, fontweight="bold")
                for spine in ax.spines.values():
                    spine.set_edgecolor(q_color)
                    spine.set_linewidth(2)
                if j == 0:
                    ax.set_title(qname, fontsize=9, fontweight="bold", pad=6)

                row_idx += 1

        plt.tight_layout()
        fname = save_path.replace(".png", f"_{qname}.png")
        os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info(f"Failure analysis saved to {fname}")
