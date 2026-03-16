from typing import Optional, Iterator
import torch

from inference.kv_cache import KVCache
from inference.sampler import Sampler


def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    stream: bool = False,
) -> torch.Tensor:
    """Generate tokens autoregressively.

    Args:
        model: RNewMindForCausalLM (or any model returning .logits and .past_key_values).
        input_ids: prompt token ids, shape (batch, seq_len).
        max_new_tokens: maximum tokens to generate.
        temperature: sampling temperature.
        top_k: top-k sampling filter (0 = disabled).
        top_p: nucleus sampling threshold (1.0 = disabled).
        eos_token_id: stop generation when this token is produced.
        stream: if True, return a generator yielding one token at a time.

    Returns:
        Generated token tensor shape (batch, max_new_tokens) when stream=False,
        or a generator of (batch, 1) tensors when stream=True.
    """
    if stream:
        return _stream(model, input_ids, max_new_tokens, temperature, top_k, top_p, eos_token_id)

    kv = KVCache()
    generated = []
    current_ids = input_ids

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(current_ids, past_key_values=kv.cache, use_cache=True)
        logits = out.logits[:, -1, :]
        next_token = Sampler.sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        kv.update(out.past_key_values)
        generated.append(next_token)
        current_ids = next_token
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return torch.cat(generated, dim=1)


def _stream(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_token_id: Optional[int],
) -> Iterator[torch.Tensor]:
    kv = KVCache()
    current_ids = input_ids

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(current_ids, past_key_values=kv.cache, use_cache=True)
        logits = out.logits[:, -1, :]
        next_token = Sampler.sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        kv.update(out.past_key_values)
        yield next_token
        current_ids = next_token
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
