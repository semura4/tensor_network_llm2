# Tensor Network Language Model

テンソルネットワーク（Matrix Product State / Tensor Train）を用いた言語モデルの実装と、Transformerとの比較実験。

MPS再帰を **parallel prefix scan** で O(log T) に並列化し、Transformerに匹敵する訓練・推論速度を実現しています。

## Architecture

3つのモデルアーキテクチャを実装・比較:

### 1. MPS Language Model
Matrix Product State に着想を得た線形再帰モデル。各位置で対角状態遷移行列とゲート付き入力で隠れ状態を更新する。

```
Token + Position Embedding → N × [LayerNorm → Multi-Head MPS Recurrence → Residual → LayerNorm → FFN → Residual] → LayerNorm → LM Head
```

### 2. Transformer Language Model
標準的な decoder-only Transformer (GPT-style)。

```
Token + Position Embedding → N × [LayerNorm → Causal Self-Attention → Residual → LayerNorm → FFN → Residual] → LayerNorm → LM Head
```

### 3. Hybrid (MPS-Attention Transformer)
Transformer の Self-Attention を MPS 再帰に置換したハイブリッドモデル。FFN やLayerNorm 等の Transformer 構造は維持。

## Key Ideas

### MPS から並列スキャンへの道筋

1. **MPS/Tensor Train の再帰的収縮** — 左から右へ TT-core を順次収縮する操作は、本質的に線形再帰 (Linear RNN) と等価
2. **対角化** — 状態遷移行列を対角行列に制限: `h_t = a ⊙ (1-g_t) ⊙ h_{t-1} + g_t ⊙ W_ih(x_t)`
3. **Parallel Prefix Scan** — element-wise 線形再帰 `h_t = α_t h_{t-1} + β_t` は結合律を満たすため、Blelloch scan で O(log T) ステップに並列化可能

この過程は S4 → S4D → Mamba に至る State Space Model の設計思想と同じ帰結に独立に到達したものです。

### 並列スキャンの仕組み

線形再帰 `h_t = a_t · h_{t-1} + b_t` の合成は結合的:

```
(a₁, b₁) ⊗ (a₂, b₂) = (a₁·a₂, a₂·b₁ + b₂)
```

これにより、T 要素の再帰を log₂(T) 回の並列ステップで計算できます（Hillis-Steele 方式）。

## Results

WikiText-2, vocab=8192, d_model=256, 4 layers, 5 epochs:

| Metric | MPS | Transformer | Hybrid |
|---|---|---|---|
| Parameters | 4.24M | 4.24M | 4.24M |
| **Test Perplexity** | 679.9 | 813.2 | **656.6** |
| Inference (tok/s) | 779K | **2,281K** | 700K |
| Inference (ms/batch) | 5.3 | **1.8** | 5.9 |
| Train throughput (step/s) | 59.3 | **135.0** | 55.6 |
| Training time (5 epochs) | — | **19s** | 32s |

- **Hybrid が最良の Perplexity** — Transformer より 19% 低い
- Transformer は推論・訓練速度で優位（attention の GPU 並列性）
- MPS/Hybrid は並列スキャンにより実用的な速度を達成

## Project Structure

```
tensor-network-llm/
├── config/                  # Model & training configs (YAML)
│   ├── mps_small.yaml
│   ├── transformer_small.yaml
│   └── hybrid_small.yaml
├── src/
│   ├── models/
│   │   ├── parallel_scan.py # Parallel prefix scan (Blelloch/Hillis-Steele)
│   │   ├── mps_core.py      # MPSRecurrence with diagonal transition + scan
│   │   ├── mps_layer.py     # MPSSequenceLayer (pre-norm residual block)
│   │   ├── mps_lm.py        # Full MPS Language Model
│   │   ├── transformer_lm.py# Decoder-only Transformer (GPT-style)
│   │   ├── hybrid_lm.py     # Hybrid: Transformer + MPS attention
│   │   └── embedding.py     # Shared token + position embedding
│   ├── data/
│   │   └── dataset.py       # WikiText-2 loading + BPE tokenizer
│   ├── training/
│   │   └── trainer.py       # Unified trainer (AdamW, cosine LR, AMP)
│   └── evaluation/
│       ├── metrics.py        # Perplexity, parameter counting
│       └── benchmark.py      # Inference speed, memory, throughput
├── scripts/
│   ├── compare.py            # Train & compare MPS vs Transformer
│   ├── train_hybrid.py       # Train Hybrid + 3-way comparison
│   └── visualize.py          # Loss curves & bar charts
└── results/                  # Saved comparison results (JSON)
```

## Quick Start

```bash
pip install -r requirements.txt

# Train & compare all three models
python scripts/train_hybrid.py

# Or train MPS vs Transformer only
python scripts/compare.py

# Generate plots from saved results
python scripts/visualize.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (recommended, CPU works but slower)

## References

- Oseledets, I. (2011). *Tensor-Train Decomposition*. SIAM Journal on Scientific Computing.
- Gu, A., Goel, K., & Ré, C. (2022). *Efficiently Modeling Long Sequences with Structured State Spaces* (S4).
- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*.
- Blelloch, G. (1990). *Prefix Sums and Their Applications*. Technical Report CMU-CS-90-190.

## License

MIT
