"""ZeroMimic's task-conditioned actor, value network, and learning targets."""

from __future__ import annotations

from math import isfinite

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _features(obs_dim: int, hidden_dims: tuple[int, ...]) -> tuple[nn.Sequential, int]:
  if obs_dim <= 0 or any(width <= 0 for width in hidden_dims):
    raise ValueError("Observation and hidden dimensions must be positive.")
  layers: list[nn.Module] = []
  width = obs_dim
  for next_width in hidden_dims:
    layers.extend((nn.Linear(width, next_width), nn.ELU()))
    width = next_width
  return nn.Sequential(*layers), width


def _actions_for_mode(actions: Tensor, mode: Tensor) -> Tensor:
  index = mode.unsqueeze(-1).unsqueeze(-1).expand(*mode.shape, 1, actions.shape[-1])
  return actions.gather(-2, index).squeeze(-2)


class ModeActor(nn.Module):
  """Receives task geometry and observation history as a flat feature vector."""

  def __init__(
    self,
    obs_dim: int,
    action_dim: int,
    num_modes: int = 4,
    hidden_dims: tuple[int, ...] = (256, 128),
  ) -> None:
    super().__init__()
    if action_dim <= 0 or num_modes <= 0:
      raise ValueError("The action dimension and number of modes must be positive.")
    self.action_dim = action_dim
    self.num_modes = num_modes
    self.trunk, width = _features(obs_dim, hidden_dims)
    self.mode_selector = nn.Linear(width, num_modes)
    self.action_heads = nn.Linear(width, num_modes * action_dim)

  def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
    features = self.trunk(observations)
    logits = self.mode_selector(features)
    actions = self.action_heads(features).unflatten(
      -1, (self.num_modes, self.action_dim)
    )
    return logits, actions.tanh()

  def act(
    self,
    observations: Tensor,
    mode: Tensor | None = None,
    *,
    deterministic: bool = True,
    generator: torch.Generator | None = None,
  ) -> tuple[Tensor, Tensor]:
    """Select a mode at a segment boundary, or reuse the caller's active mode."""
    logits, actions = self(observations)
    if mode is None:
      if deterministic:
        mode = logits.argmax(dim=-1)
      else:
        probabilities = logits.detach().softmax(dim=-1).reshape(-1, self.num_modes)
        sampled = torch.multinomial(probabilities, 1, generator=generator)
        mode = sampled.reshape(logits.shape[:-1])
    elif mode.shape != logits.shape[:-1] or mode.dtype != torch.long:
      raise ValueError(
        "The active mode must be a long tensor matching the batch dimensions."
      )
    return _actions_for_mode(actions, mode), mode


class ValueNetwork(nn.Module):
  """Predicts the value of the search-free actor from critic observations."""

  def __init__(self, obs_dim: int, hidden_dims: tuple[int, ...] = (256, 128)) -> None:
    super().__init__()
    self.trunk, width = _features(obs_dim, hidden_dims)
    self.value = nn.Linear(width, 1)

  def forward(self, observations: Tensor) -> Tensor:
    return self.value(self.trunk(observations)).squeeze(-1)


@torch.no_grad()
def td_lambda_targets(
  rewards: Tensor,
  next_values: Tensor,
  terminated: Tensor,
  truncated: Tensor,
  *,
  gamma: float = 0.995,
  lambda_: float = 0.95,
) -> Tensor:
  """Compute time-major targets from the critic's pre-reset successor values.

  A task terminal suppresses bootstrapping. A collection truncation, including
  the end of this rollout, keeps its successor value and stops the recursion.
  """
  if rewards.ndim == 0 or any(
    tensor.shape != rewards.shape for tensor in (next_values, terminated, truncated)
  ):
    raise ValueError("All inputs must have the same [time, ...] shape.")
  if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
    raise ValueError("Termination and truncation flags must be boolean.")
  if not 0.0 <= gamma <= 1.0 or not 0.0 <= lambda_ <= 1.0:
    raise ValueError("gamma and lambda_ must be in [0, 1].")
  if not rewards.is_floating_point() or not next_values.is_floating_point():
    raise ValueError("Rewards and successor values must be floating-point tensors.")

  if gamma == 0.0:
    return rewards.clone()

  next_values = next_values.to(dtype=rewards.dtype)
  targets = torch.empty_like(rewards)
  for step in range(rewards.shape[0] - 1, -1, -1):
    bootstrap = next_values[step]
    if lambda_ > 0.0 and step + 1 < rewards.shape[0]:
      future = torch.lerp(bootstrap, targets[step + 1], lambda_)
      bootstrap = torch.where(truncated[step], bootstrap, future)
    bootstrap = torch.where(terminated[step], 0.0, bootstrap)
    targets[step] = rewards[step] + gamma * bootstrap
  return targets


def distillation_loss(
  actor: ModeActor,
  observations: Tensor,
  teacher_actions: Tensor,
  modes: Tensor,
  advantages: Tensor,
  valid: Tensor,
  *,
  temperature: float = 1.0,
  max_log_weight: float = 5.0,
  mode_weight: float = 1.0,
) -> Tensor:
  """Fit one batch of verified prefixes from the same task.

  Each prefix starts at a mode boundary. Valid steps share one prefix weight,
  and cross-entropy supervises the mode selector at the first step.
  """
  if observations.ndim != 3 or observations.shape[1] == 0:
    raise ValueError(
      "Observations must have shape [batch, time, features] with time > 0."
    )
  batch, time, _ = observations.shape
  if (
    teacher_actions.shape != (batch, time, actor.action_dim)
    or modes.shape != (batch,)
    or advantages.shape != (batch,)
    or valid.shape != (batch, time)
  ):
    raise ValueError(
      "Teacher tensors must match the observation batch and time dimensions."
    )
  if modes.dtype != torch.long or valid.dtype != torch.bool:
    raise ValueError(
      "Modes must be long indices and the validity mask must be boolean."
    )
  if (
    not all(isfinite(value) for value in (temperature, max_log_weight, mode_weight))
    or temperature <= 0.0
    or max_log_weight < 0.0
    or mode_weight < 0.0
  ):
    raise ValueError(
      "The temperature must be finite and positive and the weights finite and non-negative."
    )

  advantages = advantages.detach()
  active = valid[:, 0] & torch.isfinite(advantages) & (advantages > 0.0)
  valid = valid & active.unsqueeze(-1)
  modes = torch.where(active, modes, 0)
  masked_obs = torch.where(valid.unsqueeze(-1), observations, 0.0)
  masked_actions = torch.where(valid.unsqueeze(-1), teacher_actions.detach(), 0.0)
  logits, actions = actor(masked_obs)
  prediction = _actions_for_mode(actions, modes[:, None].expand(batch, time))
  action_loss = F.huber_loss(prediction, masked_actions, reduction="none").mean(dim=-1)
  action_loss = (action_loss * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
  mode_loss = F.cross_entropy(logits[:, 0], modes, reduction="none")
  scaled = (advantages / temperature).clamp(0.0, max_log_weight)
  log_weights = scaled.masked_fill(~active, -torch.inf)
  normalizer = log_weights.logsumexp(dim=0)
  weights = (log_weights - torch.where(active.any(), normalizer, 0.0)).exp()
  return ((action_loss + mode_weight * mode_loss) * weights).sum()
