import torch
import torch.nn.functional as F


class Sampler:
    """Token sampling utilities for autoregressive decoding."""

    @staticmethod
    def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Zero out all logits except the top-k."""
        if top_k <= 0:
            return logits
        values, _ = torch.topk(logits, top_k)
        min_val = values[..., -1].unsqueeze(-1)
        return logits.masked_fill(logits < min_val, float('-inf'))

    @staticmethod
    def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Nucleus (top-p) filtering."""
        if top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        # Restore original ordering
        return sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    @staticmethod
    def sample(logits: torch.Tensor, temperature: float = 1.0,
               top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
        """Sample next token from logits.

        Args:
            logits: shape (batch, vocab)
            temperature: scaling factor (1.0 = no change)
            top_k: keep only top-k logits
            top_p: nucleus sampling threshold

        Returns:
            token indices, shape (batch, 1)
        """
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature
        logits = Sampler.top_k_filter(logits, top_k)
        logits = Sampler.top_p_filter(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
