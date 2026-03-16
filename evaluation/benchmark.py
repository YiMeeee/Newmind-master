from typing import List, Dict, Any
import torch

from inference.generate import generate


def run_benchmark(model, tokenizer, prompts: List[str], device: str = "cpu",
                  max_new_tokens: int = 128, temperature: float = 1.0,
                  top_k: int = 0, top_p: float = 1.0) -> List[Dict[str, Any]]:
    """Run generation quality benchmark over a list of prompts.

    Args:
        model: RNewMindForCausalLM (or any causal LM).
        tokenizer: tokenizer with __call__ and decode methods.
        prompts: list of prompt strings.
        device: torch device string.
        max_new_tokens: maximum tokens to generate per prompt.
        temperature: sampling temperature.
        top_k: top-k sampling filter.
        top_p: nucleus sampling threshold.

    Returns:
        List of dicts with keys 'prompt', 'generated', 'num_tokens'.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)

            generated_ids = generate(
                model,
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
            )

            generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
            results.append({
                "prompt": prompt,
                "generated": generated_text,
                "num_tokens": generated_ids.shape[1],
            })

    return results
