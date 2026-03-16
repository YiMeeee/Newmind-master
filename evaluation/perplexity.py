from typing import List
import torch
import torch.nn.functional as F


def calc_perplexity(model, tokenizer, texts: List[str], device: str = "cpu",
                    max_length: int = 512) -> float:
    """Compute average perplexity over a list of texts.

    Args:
        model: RNewMindForCausalLM (or any causal LM returning .logits).
        tokenizer: tokenizer with __call__ and eos_token_id.
        texts: list of text strings to evaluate.
        device: torch device string.
        max_length: truncation length for each text.

    Returns:
        Average perplexity (float).
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] < 2:
                continue

            out = model(input_ids)
            logits = out.logits  # (1, seq, vocab)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_nll += loss.item()
            total_tokens += shift_labels.numel()

    return float(torch.exp(torch.tensor(total_nll / total_tokens)))
