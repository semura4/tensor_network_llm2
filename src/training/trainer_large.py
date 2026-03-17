"""Step-based trainer for large-scale GPT training.

Features:
- Gradient accumulation for large effective batch sizes
- Step-based training (not epoch-based)
- Periodic evaluation and checkpointing
- Mixed precision (AMP) with bf16/fp16
- Gradient clipping and learning rate scheduling
"""

import os
import time
import math
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path


class LargeTrainer:
    """Step-based trainer for GPT-scale models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 6e-4,
        min_lr_ratio: float = 0.1,
        weight_decay: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.95,
        warmup_steps: int = 2000,
        max_steps: int = 76000,
        grad_clip: float = 1.0,
        grad_accumulation_steps: int = 32,
        mixed_precision: bool = True,
        eval_interval: int = 1000,
        eval_steps: int = 50,
        save_interval: int = 5000,
        save_dir: str = "checkpoints",
        device: str = "cuda",
        log_interval: int = 100,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.grad_clip = grad_clip
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.eval_interval = eval_interval
        self.eval_steps = eval_steps
        self.save_interval = save_interval
        self.save_dir = Path(save_dir)
        self.log_interval = log_interval
        self.lr = lr
        self.min_lr = lr * min_lr_ratio

        self.use_amp = mixed_precision and device == "cuda"
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Optimizer
        param_groups = self._get_param_groups(weight_decay)
        self.optimizer = torch.optim.AdamW(
            param_groups, lr=lr, betas=(beta1, beta2), eps=1e-8,
        )
        self.scaler = GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))
        self.criterion = nn.CrossEntropyLoss()

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _get_param_groups(self, weight_decay: float):
        """Separate parameters into decay and no-decay groups."""
        decay = set()
        no_decay = set()
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or "bias" in name or "norm" in name or "emb" in name:
                no_decay.add(name)
            else:
                decay.add(name)

        param_dict = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        return [
            {"params": [param_dict[n] for n in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[n] for n in sorted(no_decay)], "weight_decay": 0.0},
        ]

    def _get_lr(self, step: int) -> float:
        """Cosine schedule with linear warmup."""
        if step < self.warmup_steps:
            return self.lr * step / max(self.warmup_steps, 1)
        if step >= self.max_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + coeff * (self.lr - self.min_lr)

    def _set_lr(self, step: int):
        lr = self._get_lr(step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set for eval_steps batches."""
        self.model.eval()
        total_loss = 0.0
        count = 0
        val_iter = iter(self.val_loader)

        for _ in range(self.eval_steps):
            try:
                input_ids, targets = next(val_iter)
            except StopIteration:
                break
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            with autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                logits = self.model(input_ids)
                loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            total_loss += loss.item()
            count += 1

        self.model.train()
        return total_loss / max(count, 1)

    def save_checkpoint(self, step: int, val_loss: float):
        """Save model checkpoint."""
        path = self.save_dir / f"step_{step}.pt"
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
        }, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and return the step number."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        print(f"  Resumed from {path} at step {ckpt['step']}")
        return ckpt["step"]

    def train(self, resume_from: str = None) -> dict:
        """Run step-based training loop."""
        start_step = 0
        if resume_from and os.path.exists(resume_from):
            start_step = self.load_checkpoint(resume_from)

        self.model.train()
        train_iter = iter(self.train_loader)

        total_loss = 0.0
        total_tokens = 0
        best_val_loss = float("inf")
        start_time = time.time()
        log_start = time.time()
        history = {"steps": [], "train_losses": [], "val_losses": [], "lrs": []}

        for step in range(start_step, self.max_steps):
            lr = self._set_lr(step)
            self.optimizer.zero_grad()

            # Gradient accumulation
            accum_loss = 0.0
            for micro_step in range(self.grad_accumulation_steps):
                try:
                    input_ids, targets = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    input_ids, targets = next(train_iter)

                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                with autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    logits = self.model(input_ids)
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)), targets.view(-1)
                    )
                    loss = loss / self.grad_accumulation_steps

                self.scaler.scale(loss).backward()
                accum_loss += loss.item()
                total_tokens += targets.numel()

            # Gradient clipping and optimizer step
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += accum_loss

            # Logging
            if (step + 1) % self.log_interval == 0:
                avg_loss = total_loss / self.log_interval
                elapsed = time.time() - log_start
                tokens_per_sec = total_tokens / elapsed
                ppl = math.exp(min(avg_loss, 20))  # cap to avoid overflow
                print(
                    f"Step {step+1:>6d}/{self.max_steps} | "
                    f"loss {avg_loss:.4f} | ppl {ppl:.1f} | "
                    f"lr {lr:.2e} | grad_norm {grad_norm:.2f} | "
                    f"tok/s {tokens_per_sec:,.0f} | "
                    f"elapsed {time.time()-start_time:.0f}s"
                )
                history["steps"].append(step + 1)
                history["train_losses"].append(avg_loss)
                history["lrs"].append(lr)
                total_loss = 0.0
                total_tokens = 0
                log_start = time.time()

            # Evaluation
            if (step + 1) % self.eval_interval == 0:
                val_loss = self.evaluate()
                val_ppl = math.exp(min(val_loss, 20))
                print(f"  Val loss: {val_loss:.4f} | Val PPL: {val_ppl:.1f}")
                history["val_losses"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(step + 1, val_loss)

            # Periodic saving
            elif (step + 1) % self.save_interval == 0:
                self.save_checkpoint(step + 1, float("inf"))

        # Final evaluation and save
        val_loss = self.evaluate()
        self.save_checkpoint(self.max_steps, val_loss)

        history["best_val_loss"] = best_val_loss
        history["best_val_ppl"] = math.exp(min(best_val_loss, 20))
        history["total_time"] = time.time() - start_time

        # Save history
        with open(self.save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        return history
