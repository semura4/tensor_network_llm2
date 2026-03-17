#!/usr/bin/env python3
"""Train GPT-scale models (Standard Transformer or Tensor Network).

Usage:
    python scripts/train_gpt.py --config config/gpt_tn_350m.yaml
    python scripts/train_gpt.py --config config/gpt2_medium_baseline.yaml
    python scripts/train_gpt.py --config config/gpt_tn_125m.yaml --resume checkpoints/step_5000.pt
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml

from src.models.gpt import GPTModel, build_gpt_model
from src.data.openwebtext import get_owt_dataloaders
from src.training.trainer_large import LargeTrainer
from src.evaluation.metrics import count_parameters


def main():
    parser = argparse.ArgumentParser(description="Train GPT-scale model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Build model
    model = build_gpt_model(config)
    param_info = count_parameters(model)
    print(f"\nModel: {config['model']['type']} | {param_info['total']:,} parameters")
    for comp, cnt in param_info["breakdown"].items():
        print(f"  {comp}: {cnt:,}")

    # Data
    dc = config["data"]
    tc = config["training"]
    train_loader, val_loader = get_owt_dataloaders(
        batch_size=tc["micro_batch_size"],
        seq_len=dc["seq_len"],
    )

    # Trainer
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    save_dir = os.path.join("checkpoints", config_name)

    trainer = LargeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=tc["lr"],
        min_lr_ratio=tc.get("min_lr_ratio", 0.1),
        weight_decay=tc["weight_decay"],
        beta1=tc.get("beta1", 0.9),
        beta2=tc.get("beta2", 0.95),
        warmup_steps=tc["warmup_steps"],
        max_steps=tc["max_steps"],
        grad_clip=tc["grad_clip"],
        grad_accumulation_steps=tc["grad_accumulation_steps"],
        mixed_precision=tc["mixed_precision"],
        eval_interval=tc.get("eval_interval", 1000),
        eval_steps=tc.get("eval_steps", 50),
        save_interval=tc.get("save_interval", 5000),
        save_dir=save_dir,
        device=device,
        log_interval=tc.get("log_interval", 100),
    )

    print(f"\nTraining config:")
    print(f"  Effective batch size: {tc['micro_batch_size'] * tc['grad_accumulation_steps']}")
    print(f"  Sequence length: {dc['seq_len']}")
    print(f"  Max steps: {tc['max_steps']}")
    print(f"  Tokens per step: {tc['micro_batch_size'] * tc['grad_accumulation_steps'] * dc['seq_len']:,}")
    print(f"  Total tokens: {tc['micro_batch_size'] * tc['grad_accumulation_steps'] * dc['seq_len'] * tc['max_steps']:,}")
    print(f"  Save dir: {save_dir}")
    print()

    history = trainer.train(resume_from=args.resume)

    print(f"\nTraining complete!")
    print(f"  Best val loss: {history['best_val_loss']:.4f}")
    print(f"  Best val PPL: {history['best_val_ppl']:.1f}")
    print(f"  Total time: {history['total_time']/3600:.1f} hours")


if __name__ == "__main__":
    main()
