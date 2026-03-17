"""WikiText-2 dataset loading and tokenization."""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors


CACHE_DIR = Path(__file__).parent.parent.parent / "cache"


def train_bpe_tokenizer(texts: list[str], vocab_size: int = 8192) -> Tokenizer:
    """Train a BPE tokenizer on the given texts."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer


def get_tokenizer(vocab_size: int = 8192) -> Tokenizer:
    """Get or train the BPE tokenizer."""
    tok_path = CACHE_DIR / f"bpe_tokenizer_{vocab_size}.json"
    if tok_path.exists():
        return Tokenizer.from_file(str(tok_path))

    print("Training BPE tokenizer...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=str(CACHE_DIR))
    texts = [t for t in ds["train"]["text"] if t.strip()]
    tokenizer = train_bpe_tokenizer(texts, vocab_size)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tok_path))
    print(f"Tokenizer saved to {tok_path}")
    return tokenizer


class LMDataset(Dataset):
    """Language modeling dataset that produces fixed-length token chunks."""

    def __init__(self, split: str, seq_len: int = 256, vocab_size: int = 8192):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=str(CACHE_DIR))
        tokenizer = get_tokenizer(vocab_size)

        texts = [t for t in ds[split]["text"] if t.strip()]
        all_text = "\n".join(texts)

        encoded = tokenizer.encode(all_text)
        token_ids = encoded.ids

        # Clamp any OOV tokens to vocab range
        token_ids = [min(t, vocab_size - 1) for t in token_ids]

        # Split into fixed-length chunks (seq_len + 1 for input/target shift)
        chunk_size = seq_len + 1
        num_chunks = len(token_ids) // chunk_size
        token_ids = token_ids[: num_chunks * chunk_size]

        self.chunks = torch.tensor(token_ids, dtype=torch.long).view(num_chunks, chunk_size)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]  # input_ids, target_ids


def get_dataloaders(
    batch_size: int = 64,
    seq_len: int = 256,
    vocab_size: int = 8192,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get train, validation, and test dataloaders."""
    train_ds = LMDataset("train", seq_len, vocab_size)
    valid_ds = LMDataset("validation", seq_len, vocab_size)
    test_ds = LMDataset("test", seq_len, vocab_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Dataset sizes - Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")
    return train_loader, valid_loader, test_loader
