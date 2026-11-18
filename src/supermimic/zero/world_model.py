"""Return reduction shared by planning and imagined policy rollouts."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ModelRollout:
  """Per-member rewards, termination probabilities, and final critic values."""

  rewards: Tensor  # [E, B, H], including reward on the terminal step
  termination_probability: Tensor  # [E, B, H]
  terminal_value: Tensor  # [E, B]

  def pessimistic_return(self, *, gamma: float, beta: float) -> Tensor:
    """Reduce a rollout to one risk-adjusted return per candidate."""
    if self.rewards.ndim != 3 or any(size == 0 for size in self.rewards.shape):
      raise ValueError("rewards must have shape [E, B, H] with nonempty axes")
    if self.termination_probability.shape != self.rewards.shape:
      raise ValueError("termination_probability must match rewards")
    if self.terminal_value.shape != self.rewards.shape[:2]:
      raise ValueError("terminal_value must have shape [E, B]")
    if not 0 < gamma <= 1 or not 0 <= beta < float("inf"):
      raise ValueError(
        "gamma must be in (0, 1] and beta must be finite and nonnegative"
      )

    continuation = 1 - self.termination_probability.clamp(0, 1)
    survival_after = continuation.cumprod(dim=-1)
    survival_before = torch.cat(
      (torch.ones_like(survival_after[..., :1]), survival_after[..., :-1]), dim=-1
    )
    horizon = self.rewards.shape[-1]
    steps = torch.arange(horizon, device=self.rewards.device, dtype=self.rewards.dtype)
    reward_return = (self.rewards * survival_before * gamma**steps).sum(dim=-1)
    tail_return = gamma**horizon * survival_after[..., -1] * self.terminal_value
    member_return = reward_return + tail_return
    return member_return.mean(dim=0) - beta * member_return.std(dim=0, correction=0)
