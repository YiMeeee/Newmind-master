import math
import os
import torch


class TrainerEngine:
    """Unified training engine encapsulating the core training loop primitives."""

    def __init__(self, model, optimizer, scheduler=None, scaler=None, args=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.args = args

    def train_step(self, batch, loss_fn, ctx=None):
        """Execute a single forward + backward + optimizer step.

        Args:
            batch: tuple of tensors passed to model forward.
            loss_fn: callable(outputs, batch) → scalar loss.
            ctx: optional autocast context (e.g. torch.cuda.amp.autocast).

        Returns:
            loss value (float).
        """
        ctx = ctx or torch.inference_mode.__class__  # fallback no-op
        self.optimizer.zero_grad()

        with (ctx if ctx is not None else _nullcontext()):
            outputs = self.model(*batch) if isinstance(batch, (list, tuple)) else self.model(**batch)
            loss = loss_fn(outputs, batch)

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    def save_checkpoint(self, path: str):
        """Save model + optimizer state to path."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        torch.save(state, path)

    @staticmethod
    def get_lr(current_step: int, total_steps: int, lr: float) -> float:
        """Cosine decay with 10% warm floor."""
        return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


class _nullcontext:
    """Minimal no-op context manager for Python < 3.7 compatibility."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
