#!/usr/bin/env python3
"""Visualization: loss curves and comparison bar charts."""

import os
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]


def plot_comparison(results: dict, output_dir: str):
    """Generate comparison plots from results dict."""
    os.makedirs(output_dir, exist_ok=True)
    models = list(results.keys())
    colors = COLORS[:len(models)]

    # 1. Loss curves (only for models with training history)
    models_with_history = [m for m in models if "train_losses" in results[m]]
    if models_with_history:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for name, color in zip(models_with_history, colors):
            data = results[name]
            epochs = range(1, len(data["train_losses"]) + 1)
            axes[0].plot(epochs, data["train_losses"], label=f"{name} (train)", linewidth=2, color=color)
            axes[0].plot(epochs, data["valid_losses"], label=f"{name} (valid)", linewidth=2, linestyle="--", color=color)

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (Cross Entropy)")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        for name, color in zip(models_with_history, colors):
            data = results[name]
            epochs = range(1, len(data["valid_losses"]) + 1)
            ppls = [math.exp(l) if l < 20 else float("inf") for l in data["valid_losses"]]
            axes[1].plot(epochs, ppls, label=name, linewidth=2, color=color)

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Perplexity")
        axes[1].set_title("Validation Perplexity")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
        plt.close()

    # 2. Comparison bar charts
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Test Perplexity
    ax = axes[0, 0]
    ppls = [results[m]["test_ppl"] for m in models]
    bars = ax.bar(models, ppls, color=colors)
    ax.set_title("Test Perplexity (lower is better)")
    ax.set_ylabel("Perplexity")
    for bar, v in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(ppls) * 0.01,
                f"{v:.1f}", ha="center", fontweight="bold", fontsize=10)

    # Parameters
    ax = axes[0, 1]
    params = [results[m]["params"]["total"] / 1e6 for m in models]
    no_embed = [results[m]["params"]["no_embed"] / 1e6 for m in models]
    x = range(len(models))
    width = 0.35
    ax.bar([i - width/2 for i in x], params, width, label="Total", color=colors)
    ax.bar([i + width/2 for i in x], no_embed, width, label="No Embed",
           color=[c + "80" for c in colors])
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_title("Parameter Count")
    ax.set_ylabel("Parameters (M)")
    ax.legend()

    # Inference Speed
    ax = axes[1, 0]
    speeds = [results[m]["inference"]["tokens_per_second"] / 1000 for m in models]
    bars = ax.bar(models, speeds, color=colors)
    ax.set_title("Inference Speed (higher is better)")
    ax.set_ylabel("K tokens/second")
    for bar, v in zip(bars, speeds):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(speeds) * 0.01,
                f"{v:.0f}K", ha="center", fontweight="bold", fontsize=10)

    # Memory
    ax = axes[1, 1]
    mem = [results[m]["memory"]["peak_memory_mb"] for m in models]
    bars = ax.bar(models, mem, color=colors)
    ax.set_title("Peak GPU Memory (lower is better)")
    ax.set_ylabel("MB")
    for bar, v in zip(bars, mem):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(mem) * 0.01,
                f"{v:.0f}", ha="center", fontweight="bold", fontsize=10)

    plt.suptitle("Tensor Network LM vs Transformer LM", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}/")


def main():
    results_path = os.path.join(os.path.dirname(__file__), "..", "results", "comparison.json")
    if not os.path.exists(results_path):
        print("No results found. Run compare.py first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    plot_comparison(results, output_dir)


if __name__ == "__main__":
    main()
