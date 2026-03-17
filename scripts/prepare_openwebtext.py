#!/usr/bin/env python3
"""Pre-tokenize OpenWebText using GPT-2 tokenizer and save as memmap.

This downloads OpenWebText (~6GB compressed), tokenizes it with the GPT-2
tokenizer, and saves the result as memory-mapped numpy arrays for fast
training access.

Output:
    cache/owt/train.bin  (~18GB, ~9B tokens as uint16)
    cache/owt/val.bin    (~9MB, ~4.5M tokens as uint16)
"""

import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CACHE_DIR = Path(__file__).parent.parent / "cache" / "owt"


def main():
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    print("Loading OpenWebText dataset...")
    dataset = load_dataset("openwebtext", trust_remote_code=True)

    # Split: use last 0.5% as validation
    split = dataset["train"].train_test_split(test_size=0.005, seed=42)
    splits = {"train": split["train"], "val": split["test"]}

    for split_name, ds in splits.items():
        print(f"\nTokenizing {split_name} ({len(ds):,} documents)...")
        out_path = CACHE_DIR / f"{split_name}.bin"

        if out_path.exists():
            print(f"  {out_path} already exists, skipping.")
            continue

        # First pass: count total tokens
        all_tokens = []
        batch_size = 1000

        for i in tqdm(range(0, len(ds), batch_size), desc=f"Tokenizing {split_name}"):
            batch = ds[i : i + batch_size]["text"]
            encoded = tokenizer(batch, return_attention_mask=False)
            for ids in encoded["input_ids"]:
                all_tokens.extend(ids)

        total_tokens = len(all_tokens)
        print(f"  Total tokens: {total_tokens:,}")

        # Save as uint16 memmap (GPT-2 vocab_size=50257 fits in uint16)
        arr = np.array(all_tokens, dtype=np.uint16)
        mm = np.memmap(str(out_path), dtype=np.uint16, mode="w+", shape=arr.shape)
        mm[:] = arr[:]
        mm.flush()
        del mm, arr, all_tokens

        print(f"  Saved to {out_path} ({total_tokens:,} tokens, {out_path.stat().st_size / 1e9:.2f} GB)")

    print("\nDone! Files saved to", CACHE_DIR)


if __name__ == "__main__":
    main()
