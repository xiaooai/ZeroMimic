"""Mode-conditioned chunk CEM and projected action refinement."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

Score = Callable[[torch.Tensor], torch.Tensor]
RefineMask = torch.Tensor | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class SearchConfig:
  population: int = 256
  iterations: int = 3
  refine_elites: int = 8
  refine_steps: int = 5
  verify_count: int = 4
  knot_interval: int = 4
  elite_fraction: float = 0.125
  initial_std: float = 0.35
  min_std: float = 0.05
  refine_lr: float = 0.05
  refine_radius: float = 0.2
  refine_grad_clip: float = 1.0

  def __post_init__(self) -> None:
    for name in ("population", "iterations", "verify_count", "knot_interval"):
      if type(value := getattr(self, name)) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    for name in ("refine_elites", "refine_steps"):
      if type(value := getattr(self, name)) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    for name in (
      "initial_std",
      "min_std",
      "refine_lr",
      "refine_radius",
      "refine_grad_clip",
    ):
      if not math.isfinite(value := getattr(self, name)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if not 0 < self.elite_fraction <= 1 or self.min_std > self.initial_std:
      raise ValueError("Require 0 < elite_fraction <= 1 and min_std <= initial_std")


@dataclass(frozen=True)
class SearchResult:
  """Predicted candidates to verify, with mode indices referring to the seed rows."""

  actions: torch.Tensor  # [K, H, A]
  scores: torch.Tensor  # [K]
  mode_ids: torch.Tensor  # [K], long
  refined: torch.Tensor  # [K], bool


class SearchPlanner:
  def __init__(self, cfg: SearchConfig | None = None):
    self.cfg = cfg if cfg is not None else SearchConfig()

  def plan(
    self,
    scorer: Score,
    seeds: torch.Tensor,
    previous_action: torch.Tensor,
    *,
    action_limit: float | torch.Tensor = 1.0,
    slew_limit: float | torch.Tensor,
    refine_mask: RefineMask | None = None,
    generator: torch.Generator | None = None,
  ) -> SearchResult:
    """Plan from each seed in `[M, H, A]` using larger-is-better scorer outputs.

    The scorer takes `[N, H, A]` and returns `[N]`. Keep its state fixed for
    the duration of the call. A refinement mask may be shared `[H]`, per-mode
    `[M, H]`, or a function of elite sequences and their mode indices returning
    `[N, H]`. False entries retain the coarse actions. A result may contain
    fewer candidates than requested when scores are non-finite.
    """
    if seeds.ndim != 3 or min(seeds.shape) < 1 or not seeds.is_floating_point():
      raise ValueError("seeds must be a nonempty floating tensor of shape [M, H, A]")
    if not torch.isfinite(seeds).all():
      raise ValueError("seeds must be finite")
    num_modes, horizon, action_dim = seeds.shape
    if num_modes > self.cfg.population:
      raise ValueError("population must allocate at least one candidate per mode")
    if previous_action.shape != (action_dim,):
      raise ValueError("previous_action must have shape [A]")
    previous = previous_action.detach().to(seeds)
    amplitude = _limit(action_limit, seeds, "action_limit")
    slew = _limit(slew_limit, seeds, "slew_limit")
    if not torch.isfinite(previous).all() or (previous.abs() > amplitude).any():
      raise ValueError("previous_action must be finite and within action_limit")
    if isinstance(refine_mask, torch.Tensor):
      _check_mask(refine_mask, (horizon,), (num_modes, horizon))
    elif refine_mask is not None and not callable(refine_mask):
      raise TypeError("refine_mask must be a boolean tensor or a callable")

    with torch.no_grad():
      coarse = self._cem(scorer, seeds.detach(), previous, amplitude, slew, generator)
      elite = _mode_first(coarse.scores, coarse.mode_ids)
    refined = None
    if self.cfg.refine_steps and self.cfg.refine_elites and elite.numel():
      picked = elite[: self.cfg.refine_elites]
      ref_actions, ref_modes = coarse.actions[picked], coarse.mode_ids[picked]
      mask = _refine_mask(refine_mask, ref_actions, ref_modes)
      refined = self._refine(
        scorer,
        ref_actions,
        coarse.scores[picked],
        ref_modes,
        previous,
        amplitude,
        slew,
        mask,
      )
    return self._shortlist(coarse, refined)

  def _cem(self, scorer, seeds, previous, amplitude, slew, generator) -> SearchResult:
    cfg = self.cfg
    num_modes, horizon, _ = seeds.shape
    steps = torch.arange(horizon, device=seeds.device)
    knots = steps[:: cfg.knot_interval]
    if (horizon - 1) % cfg.knot_interval:
      knots = torch.cat((knots, steps[-1:]))
    left = (steps // cfg.knot_interval).clamp(max=knots.numel() - 1)
    right = (left + 1).clamp(max=knots.numel() - 1)
    fraction = ((steps - knots[left]) / (knots[right] - knots[left]).clamp_min(1)).to(
      seeds
    )
    fraction = fraction.view(1, horizon, 1)

    means = list(seeds[:, knots].unbind())
    deviations = [torch.full_like(mean, cfg.initial_std) for mean in means]
    size, remainder = divmod(cfg.population, num_modes)
    quotas = [size + (mode < remainder) for mode in range(num_modes)]
    modes = torch.repeat_interleave(
      torch.arange(num_modes, device=seeds.device),
      torch.tensor(quotas, device=seeds.device),
      output_size=cfg.population,
    )
    archive = []
    for iteration in range(cfg.iterations):
      rows = []
      for mean, deviation, quota in zip(means, deviations, quotas, strict=True):
        noise = torch.randn(
          (quota, *mean.shape),
          device=seeds.device,
          dtype=seeds.dtype,
          generator=generator,
        )
        if iteration == 0:
          noise[0].zero_()
        rows.append((mean + noise * deviation).clamp(-amplitude, amplitude))
      proposals = torch.cat(rows)
      actions = _project(
        torch.lerp(proposals[:, left], proposals[:, right], fraction),
        previous,
        amplitude,
        slew,
      )
      scores = _score(scorer, actions)
      archive.append((actions, scores))
      for mode, (controls, values) in enumerate(
        zip(rows, scores.split(quotas), strict=True)
      ):
        elite_count = max(1, math.ceil(cfg.elite_fraction * quotas[mode]))
        indices = values.topk(elite_count).indices
        elites = controls[indices[torch.isfinite(values[indices])]]
        if elites.numel():
          means[mode] = elites.mean(dim=0)
          deviations[mode] = elites.std(dim=0, correction=0).clamp_min(cfg.min_std)
    actions, scores = (torch.cat(parts) for parts in zip(*archive, strict=True))
    mode_ids = modes.repeat(cfg.iterations)
    return SearchResult(
      actions, scores, mode_ids, torch.zeros_like(mode_ids, dtype=torch.bool)
    )

  def _refine(
    self, scorer, coarse, coarse_scores, mode_ids, previous, amplitude, slew, mask
  ) -> SearchResult:
    cfg = self.cfg
    bounds = _refinement_bounds(coarse, amplitude, slew, mask, cfg.refine_radius)
    adam_dtype = torch.float64 if coarse.dtype == torch.float64 else torch.float32
    best, best_scores = coarse, coarse_scores
    with torch.enable_grad():
      residual = torch.zeros_like(coarse, dtype=adam_dtype, requires_grad=True)
      optimizer = torch.optim.Adam([residual], lr=cfg.refine_lr)
      for _ in range(cfg.refine_steps):
        proposed = coarse + residual.to(coarse.dtype) * mask[..., None]
        scores = _score(scorer, proposed)
        best, best_scores = _best(best, best_scores, proposed, scores)
        if not scores.requires_grad or not torch.isfinite(scores).any():
          break
        gradient = torch.autograd.grad(
          -scores[torch.isfinite(scores)].sum(), residual, allow_unused=True
        )[0]
        if gradient is None:
          break
        residual.grad = gradient.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp_(
          -cfg.refine_grad_clip, cfg.refine_grad_clip
        )
        optimizer.step()
        with torch.no_grad():
          projected = _project(
            coarse + residual.to(coarse.dtype), previous, amplitude, slew, bounds
          )
          residual.copy_(projected - coarse)
    with torch.no_grad():
      proposed = coarse + residual.to(coarse.dtype)
      best, best_scores = _best(best, best_scores, proposed, _score(scorer, proposed))
      improved = best_scores > coarse_scores
      return SearchResult(
        best[improved], best_scores[improved], mode_ids[improved], improved[improved]
      )

  def _shortlist(
    self, coarse: SearchResult, refined: SearchResult | None
  ) -> SearchResult:
    count = self.cfg.verify_count
    if refined is not None and refined.scores.numel() and count == 1:
      both = _stack(coarse, refined)
      return _take(both, _mode_first(both.scores, both.mode_ids)[:1])
    if refined is None or not refined.scores.numel():
      return _take(coarse, _mode_first(coarse.scores, coarse.mode_ids)[:count])
    refinement_count = min(count // 2, refined.scores.numel())
    refinements = _take(
      refined, _mode_first(refined.scores, refined.mode_ids)[:refinement_count]
    )
    original_order = _mode_first(
      coarse.scores, coarse.mode_ids, refinements.mode_ids.tolist()
    )
    originals = _take(coarse, original_order[: count - refinement_count])
    shortlist = _stack(originals, refinements)
    return _take(shortlist, shortlist.scores.argsort(descending=True, stable=True))


def _limit(value, seeds: torch.Tensor, name: str) -> torch.Tensor:
  bound = torch.as_tensor(value, device=seeds.device, dtype=seeds.dtype).detach()
  if bound.shape not in ((), seeds.shape[-1:]):
    raise ValueError(f"{name} must be scalar or have shape [A]")
  if not torch.isfinite(bound).all() or (bound < 0).any():
    raise ValueError(f"{name} must be finite and nonnegative")
  return bound


def _score(scorer: Score, actions: torch.Tensor) -> torch.Tensor:
  scores = scorer(actions)
  if (
    not isinstance(scores, torch.Tensor)
    or scores.shape != actions.shape[:1]
    or scores.device != actions.device
    or not scores.is_floating_point()
  ):
    raise ValueError("scorer must return a floating [N] tensor on the action device")
  return torch.where(torch.isfinite(scores), scores, -torch.inf)


def _project(proposals, previous, amplitude, slew, bounds=None) -> torch.Tensor:
  """Apply action bounds and slew limits from the actual previous action."""
  rows = []
  for step, proposed in enumerate(proposals.unbind(dim=1)):
    lo, hi = (
      (-amplitude, amplitude)
      if bounds is None
      else (bounds[0][:, step], bounds[1][:, step])
    )
    previous = proposed.clamp(
      torch.maximum(lo, previous - slew), torch.minimum(hi, previous + slew)
    )
    rows.append(previous)
  return torch.stack(rows, dim=1)


def _check_mask(mask, *shapes) -> None:
  if (
    not isinstance(mask, torch.Tensor)
    or mask.dtype != torch.bool
    or mask.shape not in shapes
  ):
    raise ValueError(f"refine_mask must return a boolean tensor with shape in {shapes}")


def _refine_mask(spec, coarse, mode_ids) -> torch.Tensor:
  n, horizon = coarse.shape[:2]
  if spec is None:
    return torch.ones((n, horizon), device=coarse.device, dtype=torch.bool)
  if callable(spec):
    with torch.no_grad():
      mask = spec(coarse, mode_ids)
    _check_mask(mask, (n, horizon))
    return mask.to(coarse.device)
  mask = spec.to(coarse.device)
  return mask.expand(n, -1) if mask.ndim == 1 else mask[mode_ids]


def _refinement_bounds(coarse, amplitude, slew, mask, radius):
  """Tighten limits backward so fixed future controls remain reachable."""
  mask = mask[..., None]
  lower = torch.where(mask, torch.maximum(coarse - radius, -amplitude), coarse)
  upper = torch.where(mask, torch.minimum(coarse + radius, amplitude), coarse)
  for step in range(coarse.shape[1] - 2, -1, -1):
    lower[:, step] = torch.maximum(lower[:, step], lower[:, step + 1] - slew)
    upper[:, step] = torch.minimum(upper[:, step], upper[:, step + 1] + slew)
  return lower, upper


def _best(best, best_scores, proposed, scores):
  scores = scores.detach()
  updated = scores > best_scores
  return torch.where(updated[:, None, None], proposed.detach(), best), torch.maximum(
    best_scores, scores
  )


def _mode_first(scores, modes, represented=()) -> torch.Tensor:
  """Rank the best candidate of each missing mode before filling remaining slots."""
  order = scores.argsort(descending=True, stable=True)
  order = order[torch.isfinite(scores[order])]
  seen = set(represented)
  first, later = [], []
  for index, mode in zip(order.tolist(), modes[order].tolist(), strict=True):
    (later if mode in seen else first).append(index)
    seen.add(mode)
  return torch.tensor(first + later, dtype=torch.long, device=scores.device)


def _take(result: SearchResult, order: torch.Tensor) -> SearchResult:
  return SearchResult(
    result.actions[order],
    result.scores[order],
    result.mode_ids[order],
    result.refined[order],
  )


def _stack(first: SearchResult, second: SearchResult) -> SearchResult:
  return SearchResult(
    torch.cat((first.actions, second.actions)),
    torch.cat((first.scores, second.scores)),
    torch.cat((first.mode_ids, second.mode_ids)),
    torch.cat((first.refined, second.refined)),
  )
