"""
Pareto frontier analysis and visualization.

Computes Pareto-optimal configurations from evaluation results
and produces matplotlib scatter plots with the frontier highlighted.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def find_pareto_optimal(
    results: Sequence,  # Sequence[EvalResult]
    minimize_size: bool = True,
    maximize_accuracy: bool = True,
) -> list:
    """
    Find Pareto-optimal points from evaluation results.

    A point is Pareto-optimal if no other point is strictly better in
    all objectives simultaneously.

    We optimize for:
    - Minimize size_mb
    - Maximize accuracy_pct

    Args:
        results: List of EvalResult objects.
        minimize_size: Whether to minimize size (True) or maximize.
        maximize_accuracy: Whether to maximize accuracy (True) or minimize.

    Returns:
        List of EvalResult objects on the Pareto frontier.
    """
    if not results:
        return []

    # Build array of objectives: [size, accuracy] for each result
    # Negate objectives we want to maximize so we can uniformly minimize
    points = []
    for r in results:
        size = r.size_mb if minimize_size else -r.size_mb
        acc = -r.accuracy_pct if maximize_accuracy else r.accuracy_pct
        points.append((size, acc))

    points_arr = np.array(points)
    n = len(points_arr)

    # A point is dominated if another point is <= in all objectives and < in at least one
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # Check if j dominates i
            if np.all(points_arr[j] <= points_arr[i]) and np.any(points_arr[j] < points_arr[i]):
                is_pareto[i] = False
                break

    return [results[i] for i in range(n) if is_pareto[i]]


def plot_pareto(
    results: Sequence,  # Sequence[EvalResult]
    save_path: str = "results/pareto.png",
    title: str = "Quantization Pareto Frontier",
) -> None:
    """
    Generate Pareto frontier plots and save to file.

    Creates a 2-panel figure:
    - Left: Size (MB) vs Accuracy (%)
    - Right: Latency (ms) vs Accuracy (%)

    Pareto-optimal points are highlighted in red.

    Args:
        results: List of EvalResult objects.
        save_path: Path to save the figure.
        title: Figure title.
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    valid = [r for r in results if r.error is None]
    if len(valid) < 2:
        logger.warning("Not enough valid results to plot Pareto frontier.")
        return

    pareto_set = set(id(r) for r in find_pareto_optimal(valid))

    sizes = [r.size_mb for r in valid]
    accuracies = [r.accuracy_pct for r in valid]
    latencies = [r.latency_ms for r in valid]
    names = [r.config_name for r in valid]
    is_pareto = [id(r) in pareto_set for r in valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ---- Left panel: Size vs Accuracy ----
    for i, (s, a, name, par) in enumerate(zip(sizes, accuracies, names, is_pareto)):
        color = "#e74c3c" if par else "#3498db"
        marker = "*" if par else "o"
        markersize = 15 if par else 8
        ax1.scatter(s, a, c=color, marker=marker, s=markersize**2, zorder=3)
        ax1.annotate(
            name,
            (s, a),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
            alpha=0.8,
        )

    # Draw Pareto frontier line
    pareto_results = [(s, a) for s, a, p in zip(sizes, accuracies, is_pareto) if p]
    if pareto_results:
        pareto_results.sort(key=lambda x: x[0])
        ps, pa = zip(*pareto_results)
        ax1.plot(ps, pa, "r--", alpha=0.5, linewidth=1.5, label="Pareto frontier")
        ax1.legend()

    ax1.set_xlabel("Model Size (MB)", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_title("Size vs. Accuracy")
    ax1.grid(True, alpha=0.3)

    # ---- Right panel: Latency vs Accuracy ----
    for i, (l, a, name, par) in enumerate(zip(latencies, accuracies, names, is_pareto)):
        color = "#e74c3c" if par else "#3498db"
        marker = "*" if par else "o"
        markersize = 15 if par else 8
        ax2.scatter(l, a, c=color, marker=marker, s=markersize**2, zorder=3)
        ax2.annotate(
            name,
            (l, a),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
            alpha=0.8,
        )

    # Draw Pareto frontier line
    pareto_results_lat = [(l, a) for l, a, p in zip(latencies, accuracies, is_pareto) if p]
    if pareto_results_lat:
        pareto_results_lat.sort(key=lambda x: x[0])
        pl, pa = zip(*pareto_results_lat)
        ax2.plot(pl, pa, "r--", alpha=0.5, linewidth=1.5, label="Pareto frontier")
        ax2.legend()

    ax2.set_xlabel("Latency (ms)", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("Latency vs. Accuracy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Pareto plot saved to {save_path}")
