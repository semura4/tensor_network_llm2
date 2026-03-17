"""Benchmarking: inference speed, memory, training throughput."""

import time

import torch
import torch.nn as nn
from torch.amp import autocast


def measure_inference_speed(
    model: nn.Module,
    seq_len: int = 256,
    batch_size: int = 32,
    num_runs: int = 100,
    device: str = "cuda",
    use_amp: bool = True,
) -> dict:
    """Measure inference speed in tokens/second."""
    model.eval()
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad(), autocast("cuda", enabled=use_amp and device == "cuda"):
            model(input_ids)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad(), autocast("cuda", enabled=use_amp and device == "cuda"):
            model(input_ids)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = batch_size * seq_len * num_runs
    return {
        "tokens_per_second": total_tokens / elapsed,
        "ms_per_batch": (elapsed / num_runs) * 1000,
        "total_time": elapsed,
    }


def measure_memory(
    model: nn.Module,
    seq_len: int = 256,
    batch_size: int = 32,
    device: str = "cuda",
    use_amp: bool = True,
) -> dict:
    """Measure peak GPU memory during forward pass."""
    if device != "cuda":
        return {"peak_memory_mb": 0, "note": "Memory measurement only supported on CUDA"}

    model.eval()
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    with torch.no_grad(), autocast("cuda", enabled=use_amp):
        _ = model(input_ids)

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {"peak_memory_mb": peak}


def measure_training_throughput(
    model: nn.Module,
    seq_len: int = 256,
    batch_size: int = 32,
    num_steps: int = 50,
    device: str = "cuda",
    use_amp: bool = True,
) -> dict:
    """Measure training throughput (steps/second)."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=use_amp and device == "cuda")

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        with autocast("cuda", enabled=use_amp and device == "cuda"):
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_steps):
        optimizer.zero_grad()
        with autocast("cuda", enabled=use_amp and device == "cuda"):
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "steps_per_second": num_steps / elapsed,
        "ms_per_step": (elapsed / num_steps) * 1000,
        "tokens_per_second": (batch_size * seq_len * num_steps) / elapsed,
    }
