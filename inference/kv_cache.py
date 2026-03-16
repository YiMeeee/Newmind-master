from typing import List, Optional, Tuple
import torch


class KVCache:
    """Manager for past_key_values used in autoregressive decoding."""

    def __init__(self):
        self._cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

    @property
    def cache(self):
        return self._cache

    def reset(self):
        """Clear all cached key-value pairs."""
        self._cache = None

    def update(self, new_past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Replace the internal cache with new past_key_values returned by the model."""
        self._cache = new_past_key_values

    def to_device(self, device: torch.device):
        """Move all cached tensors to the target device."""
        if self._cache is None:
            return
        self._cache = [
            (k.to(device), v.to(device))
            for k, v in self._cache
        ]
