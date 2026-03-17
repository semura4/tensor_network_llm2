"""Text generation with sampling strategies (top-k, top-p, temperature)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast


class TextGenerator:
    """Generates text from a trained language model."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: GPT2TokenizerFast,
        device: str = "cuda",
    ):
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate text continuation from a prompt.

        Args:
            prompt: input text
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature (lower = more deterministic)
            top_k: keep only top-k logits (0 = disabled)
            top_p: nucleus sampling threshold (1.0 = disabled)
            repetition_penalty: penalize repeated tokens (1.0 = disabled)

        Returns:
            Generated text including the prompt.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = input_ids[0].tolist()
        max_seq_len = self.model.embedding.pos_emb.num_embeddings

        for _ in range(max_new_tokens):
            # Truncate to max sequence length
            context = torch.tensor(
                [generated[-max_seq_len:]], device=self.device
            )

            logits = self.model(context)
            next_logits = logits[0, -1, :].float()  # last position

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated):
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= repetition_penalty
                    else:
                        next_logits[token_id] *= repetition_penalty

            # Temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
                next_logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                next_logits = sorted_logits.scatter(0, sorted_indices, sorted_logits)

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            # Stop on EOS
            if next_token == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    @torch.no_grad()
    def get_perplexity(self, text: str) -> float:
        """Compute perplexity of a text string."""
        import math
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        if input_ids.shape[1] < 2:
            return float("inf")

        logits = self.model(input_ids[:, :-1])
        targets = input_ids[:, 1:]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1)
        )
        return math.exp(loss.item())
