"""OpenWebText dataset with GPT-2 tokenizer and memory-mapped storage.

Usage:
    1. Pre-tokenize: python scripts/prepare_openwebtext.py
    2. Train: uses OWTDataset to load from memmap files
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "owt"


class OWTDataset(Dataset):
    """Memory-mapped OpenWebText dataset for language modeling."""

    def __init__(self, split: str, seq_len: int = 1024):
        self.seq_len = seq_len
        data_path = CACHE_DIR / f"{split}.bin"
        if not data_path.exists():
            raise FileNotFoundError(
                f"{data_path} not found. Run: python scripts/prepare_openwebtext.py"
            )
        # Memory-map the tokenized data (uint16)
        self.data = np.memmap(str(data_path), dtype=np.uint16, mode="r")
        self.num_tokens = len(self.data)
        self.num_sequences = (self.num_tokens - 1) // seq_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = torch.from_numpy(self.data[start : start + self.seq_len + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


def get_owt_dataloaders(
    batch_size: int = 4,
    seq_len: int = 1024,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Get train and validation dataloaders for OpenWebText."""
    train_ds = OWTDataset("train", seq_len)
    val_ds = OWTDataset("val", seq_len)

    print(f"OWT Dataset - Train: {len(train_ds):,} sequences ({train_ds.num_tokens:,} tokens)")
    print(f"OWT Dataset - Val: {len(val_ds):,} sequences ({val_ds.num_tokens:,} tokens)")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    return train_loader, val_loader
