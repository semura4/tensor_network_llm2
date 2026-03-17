"""Unified training loop for both MPS and Transformer models."""

import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm


class Trainer:
    """Trains a language model with standard practices."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_fraction: float = 0.05,
        grad_clip: float = 1.0,
        mixed_precision: bool = True,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.grad_clip = grad_clip
        self.use_amp = mixed_precision and device == "cuda"

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler(enabled=self.use_amp)

        # Compute total steps for scheduler
        self.total_steps = 0  # set in train()
        self.warmup_fraction = warmup_fraction
        self.base_lr = lr

        self.train_losses = []
        self.valid_losses = []
        self.best_valid_loss = float("inf")

    def _get_lr(self, step: int) -> float:
        """Cosine schedule with linear warmup."""
        warmup_steps = int(self.total_steps * self.warmup_fraction)
        if step < warmup_steps:
            return self.base_lr * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(self.total_steps - warmup_steps, 1)
        return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _set_lr(self, step: int):
        lr = self._get_lr(step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set, return average loss."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for input_ids, targets in self.valid_loader:
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            with autocast("cuda", enabled=self.use_amp):
                logits = self.model(input_ids)
                loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

        self.model.train()
        return total_loss / total_tokens

    def train(self, epochs: int = 30, log_interval: int = 50) -> dict:
        """Run training loop.

        Returns:
            dict with training history and best validation perplexity.
        """
        self.total_steps = len(self.train_loader) * epochs
        global_step = 0
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_tokens = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}")
            for input_ids, targets in pbar:
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                self._set_lr(global_step)

                self.optimizer.zero_grad()
                with autocast("cuda", enabled=self.use_amp):
                    logits = self.model(input_ids)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                batch_tokens = targets.numel()
                epoch_loss += loss.item() * batch_tokens
                epoch_tokens += batch_tokens
                global_step += 1

                if global_step % log_interval == 0:
                    avg = epoch_loss / epoch_tokens
                    pbar.set_postfix(loss=f"{avg:.4f}", ppl=f"{math.exp(avg):.1f}")

            train_loss = epoch_loss / epoch_tokens
            valid_loss = self.evaluate()
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)

            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss

            train_ppl = math.exp(train_loss)
            valid_ppl = math.exp(valid_loss)
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} PPL: {train_ppl:.1f} | "
                f"Valid Loss: {valid_loss:.4f} PPL: {valid_ppl:.1f} | "
                f"Time: {elapsed:.0f}s"
            )

        return {
            "train_losses": self.train_losses,
            "valid_losses": self.valid_losses,
            "best_valid_loss": self.best_valid_loss,
            "best_valid_ppl": math.exp(self.best_valid_loss),
            "total_time": time.time() - start_time,
        }
