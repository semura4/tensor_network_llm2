#!/usr/bin/env python3
"""Train all models (MPS, Transformer, Hybrid) and compare them."""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml

from src.data.dataset import get_dataloaders
from src.models.mps_lm import MPSLanguageModel
from src.models.transformer_lm import TransformerLanguageModel
from src.models.hybrid_lm import HybridMPSTransformerLM
from src.models.mera_lm import MERALanguageModel
from src.training.trainer import Trainer
from src.evaluation.metrics import count_parameters, compute_perplexity
from src.evaluation.benchmark import (
    measure_inference_speed,
    measure_memory,
    measure_training_throughput,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(config: dict):
    mc = config["model"]
    if mc["type"] == "mps":
        return MPSLanguageModel(
            vocab_size=mc["vocab_size"], d_model=mc["d_model"],
            d_local=mc.get("d_local", 32), bond_dim=mc["bond_dim"],
            num_layers=mc["num_layers"], max_seq_len=mc["max_seq_len"],
            dropout=mc["dropout"], tie_weights=mc["tie_weights"],
            d_ff=mc.get("d_ff"),
        )
    elif mc["type"] == "transformer":
        return TransformerLanguageModel(
            vocab_size=mc["vocab_size"], d_model=mc["d_model"],
            num_heads=mc["num_heads"], d_ff=mc["d_ff"],
            num_layers=mc["num_layers"], max_seq_len=mc["max_seq_len"],
            dropout=mc["dropout"], tie_weights=mc["tie_weights"],
        )
    elif mc["type"] == "hybrid":
        return HybridMPSTransformerLM(
            vocab_size=mc["vocab_size"], d_model=mc["d_model"],
            num_heads=mc["num_heads"], d_ff=mc["d_ff"],
            bond_dim=mc["bond_dim"],
            num_layers=mc["num_layers"], max_seq_len=mc["max_seq_len"],
            dropout=mc["dropout"], tie_weights=mc["tie_weights"],
        )
    elif mc["type"] == "mera":
        return MERALanguageModel(
            vocab_size=mc["vocab_size"], d_model=mc["d_model"],
            bond_dim=mc["bond_dim"], num_heads=mc["num_heads"],
            num_layers=mc["num_layers"], num_scales=mc["num_scales"],
            max_seq_len=mc["max_seq_len"], dropout=mc["dropout"],
            tie_weights=mc["tie_weights"], d_ff=mc.get("d_ff"),
        )
    else:
        raise ValueError(f"Unknown model type: {mc['type']}")


def print_table(results: dict):
    """Print comparison table for any number of models."""
    models = list(results.keys())
    col_width = 18
    total_width = 30 + col_width * len(models)

    print("\n" + "=" * total_width)
    print("COMPARISON RESULTS")
    print("=" * total_width)
    header = f"{'Metric':<30}" + "".join(f"{m:>{col_width}}" for m in models)
    print(header)
    print("-" * total_width)

    metric_fns = [
        ("Parameters (total)", lambda r: f"{r['params']['total']:,}"),
        ("Parameters (no embed)", lambda r: f"{r['params']['no_embed']:,}"),
        ("Best Valid Perplexity", lambda r: f"{r['best_valid_ppl']:.1f}"),
        ("Test Perplexity", lambda r: f"{r['test_ppl']:.1f}"),
        ("Inference (tok/s)", lambda r: f"{r['inference']['tokens_per_second']:,.0f}"),
        ("Inference (ms/batch)", lambda r: f"{r['inference']['ms_per_batch']:.1f}"),
        ("Peak Memory (MB)", lambda r: f"{r['memory']['peak_memory_mb']:.1f}"),
        ("Train Throughput (step/s)", lambda r: f"{r['train_speed']['steps_per_second']:.1f}"),
        ("Total Train Time (s)", lambda r: f"{r['total_time']:.0f}"),
    ]

    for name, fmt in metric_fns:
        row = f"{name:<30}" + "".join(f"{fmt(results[m]):>{col_width}}" for m in models)
        print(row)
    print("=" * total_width)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config_dir = os.path.join(os.path.dirname(__file__), "..", "config")

    configs = [
        ("MPS", load_config(os.path.join(config_dir, "mps_small.yaml"))),
        ("Transformer", load_config(os.path.join(config_dir, "transformer_small.yaml"))),
        ("Hybrid", load_config(os.path.join(config_dir, "hybrid_small.yaml"))),
        ("MERA", load_config(os.path.join(config_dir, "mera_small.yaml"))),
    ]

    # Data (shared across all models)
    dc = configs[0][1]["data"]
    train_loader, valid_loader, test_loader = get_dataloaders(
        batch_size=configs[0][1]["training"]["batch_size"],
        seq_len=dc["seq_len"],
        vocab_size=dc["tokenizer_vocab_size"],
    )

    results = {}

    for name, config in configs:
        print(f"\n{'='*70}")
        print(f"Training {name} model")
        print(f"{'='*70}")

        model = build_model(config).to(device)
        param_info = count_parameters(model)
        embed_params = param_info["breakdown"].get("embedding", 0)

        print(f"Total parameters: {param_info['total']:,}")
        for comp, cnt in param_info["breakdown"].items():
            print(f"  {comp}: {cnt:,}")

        tc = config["training"]
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            lr=tc["lr"],
            weight_decay=tc["weight_decay"],
            warmup_fraction=tc["warmup_fraction"],
            grad_clip=tc["grad_clip"],
            mixed_precision=tc["mixed_precision"],
            device=device,
        )

        history = trainer.train(epochs=tc["epochs"])

        test_ppl = compute_perplexity(model, test_loader, device=device)
        print(f"Test Perplexity: {test_ppl:.1f}")

        seq_len = config["data"]["seq_len"]
        print("Running benchmarks...")
        inference = measure_inference_speed(model, seq_len=seq_len, device=device)
        memory = measure_memory(model, seq_len=seq_len, device=device)
        train_speed = measure_training_throughput(model, seq_len=seq_len, device=device, num_steps=20)

        results[name] = {
            "params": {
                "total": param_info["total"],
                "no_embed": param_info["total"] - embed_params,
            },
            "best_valid_ppl": history["best_valid_ppl"],
            "test_ppl": test_ppl,
            "train_losses": history["train_losses"],
            "valid_losses": history["valid_losses"],
            "inference": inference,
            "memory": memory,
            "train_speed": train_speed,
            "total_time": history["total_time"],
        }

        del model, trainer
        if device == "cuda":
            torch.cuda.empty_cache()

    print_table(results)

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "comparison.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to results/comparison.json")

    # Generate plots
    try:
        from scripts.visualize import plot_comparison
        plot_comparison(results, results_dir)
        print("Plots saved to results/")
    except Exception as e:
        print(f"Could not generate plots: {e}")


if __name__ == "__main__":
    main()
