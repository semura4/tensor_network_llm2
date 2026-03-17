"""Evaluation metrics for language models."""

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast


def count_parameters(model: nn.Module) -> dict:
    """Count parameters with breakdown by component."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    breakdown = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        breakdown[name] = params

    return {
        "total": total,
        "trainable": trainable,
        "breakdown": breakdown,
    }


@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    use_amp: bool = True,
) -> float:
    """Compute perplexity on a dataset."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with autocast("cuda", enabled=use_amp and device == "cuda"):
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)
