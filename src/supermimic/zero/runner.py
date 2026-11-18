"""Coordinate one ZeroMimic training round through a simulator backend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from supermimic.zero.replay import PhysicalReplay, TeacherPrefix
from supermimic.zero.search import SearchPlanner
from supermimic.zero.tasks import TASK_SPECS, TaskId

if TYPE_CHECKING:
  from torch import Tensor

  from supermimic.zero.search import SearchResult
  from supermimic.zero.sim import PhysicalTransition, SimulationSnapshot
  from supermimic.zero.world_model import ModelRollout


@dataclass(frozen=True)
class Collection:
  trajectories: tuple[tuple[PhysicalTransition, ...], ...]
  on_policy: tuple[tuple[PhysicalTransition, ...], ...]
  anchors: tuple[SimulationSnapshot, ...]


@dataclass(frozen=True)
class VerifiedBranch:
  """Transitions after the branch point, through candidate and policy tail."""

  candidate_index: int
  transitions: tuple[PhysicalTransition, ...]
  # False for task-constraint violations. The transitions still train dynamics.
  feasible: bool


class TrainingBackend(Protocol):
  """Adapter for physical collection, simulator verification and learners.

  Collect and verify with the same actor throughout each round. Each verified
  branch starts at the supplied snapshot, uses a distinct episode ID, and
  excludes reconstruction steps from its returned transitions.
  """

  @property
  def step_dt(self) -> float: ...

  def collect(self, tasks: tuple[TaskId, ...]) -> Collection: ...

  def fit_world_model(self, replay: PhysicalReplay) -> bool:
    """Fit using real replay and report when the model is ready to score."""
    ...

  def fit_critic(
    self, on_policy: tuple[tuple[PhysicalTransition, ...], ...]
  ) -> None: ...

  def seed_actions(self, snapshot: SimulationSnapshot, horizon: int) -> Tensor:
    """Produce a proposed control sequence per mode, shaped [M, H, A]."""
    ...

  def action_limits(self, snapshot: SimulationSnapshot) -> tuple[Tensor, Tensor]:
    """Return the amplitude and per-control-step slew limits in policy units."""
    ...

  def rollout_model(
    self, snapshot: SimulationSnapshot, actions: Tensor
  ) -> ModelRollout:
    """Roll out frozen dynamics and critic with gradients to the actions."""
    ...

  def refinement_mask(
    self, snapshot: SimulationSnapshot, actions: Tensor, mode_ids: Tensor
  ) -> Tensor:
    """Choose the box candidates' contact windows, as a boolean [B, H] mask."""
    ...

  def verify(
    self, snapshot: SimulationSnapshot, candidates: SearchResult
  ) -> tuple[VerifiedBranch, ...]: ...

  def value(self, snapshot: SimulationSnapshot) -> float: ...

  def update_actor(
    self, replay: PhysicalReplay, teachers: tuple[TeacherPrefix, ...]
  ) -> None:
    """Apply imagined BPTT and any available simulator-verified labels."""
    ...


@dataclass(frozen=True)
class RoundStats:
  collected_steps: int
  verified_steps: int
  anchored_searches: int
  teacher_prefixes: int


class ZeroMimicRunner:
  def __init__(
    self,
    backend: TrainingBackend,
    replay: PhysicalReplay,
    planner: SearchPlanner,
    *,
    tasks: tuple[TaskId, ...] = tuple(TaskId),
    gamma: float = 0.995,
    beta: float = 0.5,
    mode_steps: int = 8,
    generator: torch.Generator | None = None,
  ) -> None:
    if not tasks or len(set(tasks)) != len(tasks):
      raise ValueError("tasks must contain one or more distinct task IDs")
    if not 0 < gamma <= 1 or not math.isfinite(beta) or beta < 0:
      raise ValueError(
        "gamma must be in (0, 1] and beta must be finite and nonnegative"
      )
    if mode_steps <= 0:
      raise ValueError("mode_steps must be positive")
    self.backend = backend
    self.replay = replay
    self.planner = planner
    self.tasks = tasks
    self.gamma = gamma
    self.beta = beta
    self.mode_steps = mode_steps
    self.generator = generator

  def run_round(self) -> RoundStats:
    batch = self.backend.collect(self.tasks)
    for episode in batch.trajectories:
      self.replay.append(episode)
    collected_steps = sum(map(len, batch.trajectories))
    if not self.backend.fit_world_model(self.replay):
      return RoundStats(collected_steps, 0, 0, 0)
    if batch.on_policy:
      self.backend.fit_critic(batch.on_policy)

    teachers: list[TeacherPrefix] = []
    verified_steps = 0
    for anchor in batch.anchors:
      candidates = self._search(anchor)
      if not candidates.actions.shape[0]:
        continue
      branches = self.backend.verify(anchor, candidates)
      if len({branch.candidate_index for branch in branches}) != len(branches):
        raise ValueError("a search candidate was verified more than once")
      for branch in branches:
        self._validate_branch(anchor, candidates, branch)
        self.replay.append(branch.transitions)
        verified_steps += len(branch.transitions)
      teacher = self._best_teacher(anchor, candidates, branches)
      if teacher is not None:
        teachers.append(teacher)

    self.backend.update_actor(self.replay, tuple(teachers))
    return RoundStats(
      collected_steps, verified_steps, len(batch.anchors), len(teachers)
    )

  def _search(self, anchor: SimulationSnapshot) -> SearchResult:
    if anchor.task_id not in self.tasks:
      raise ValueError("search anchor belongs to a disabled task")
    if anchor.step_index % self.mode_steps or anchor.mode_steps_remaining:
      raise ValueError("search anchors must be located at a policy mode boundary")
    task = TASK_SPECS[anchor.task_id]
    remaining = task.deadline_steps(self.backend.step_dt) - anchor.step_index
    if anchor.task_state.terminated or anchor.task_state.truncated or remaining <= 0:
      raise ValueError("search anchors must be live")
    horizon = min(remaining, task.planning_steps(self.backend.step_dt, self.mode_steps))
    with torch.no_grad():
      seeds = self.backend.seed_actions(anchor, horizon)
      amplitude, slew = self.backend.action_limits(anchor)

    def score(actions: Tensor) -> Tensor:
      rollout = self.backend.rollout_model(anchor, actions)
      return rollout.pessimistic_return(gamma=self.gamma, beta=self.beta)

    def refinement_mask(actions: Tensor, mode_ids: Tensor) -> Tensor:
      return self.backend.refinement_mask(anchor, actions, mode_ids)

    return self.planner.plan(
      score,
      seeds,
      anchor.prev_action,
      action_limit=amplitude,
      slew_limit=slew,
      refine_mask=refinement_mask if anchor.task_id == TaskId.BOX else None,
      generator=self.generator,
    )

  def _validate_branch(
    self, anchor: SimulationSnapshot, candidates: SearchResult, branch: VerifiedBranch
  ) -> None:
    if not 0 <= branch.candidate_index < len(candidates.actions):
      raise ValueError("verified branch does not identify a search candidate")
    if not branch.transitions:
      raise ValueError("verified branches must contain a physical transition")
    first = branch.transitions[0]
    if first.episode_id == anchor.episode_id:
      raise ValueError("a verified branch must have its own episode ID")
    if (
      first.scene_id != anchor.scene_id
      or first.task_id != anchor.task_id
      or first.step_index != anchor.step_index
    ):
      raise ValueError("verified branch does not start at the search anchor")

  def _best_teacher(
    self,
    anchor: SimulationSnapshot,
    candidates: SearchResult,
    branches: tuple[VerifiedBranch, ...],
  ) -> TeacherPrefix | None:
    winner = None
    best_return = float("-inf")
    for branch in branches:
      if not branch.feasible or not branch.transitions[-1].terminated:
        continue
      actual_return = sum(
        self.gamma**step * transition.reward
        for step, transition in enumerate(branch.transitions)
      )
      if actual_return > best_return:
        winner, best_return = branch, actual_return
    if winner is None:
      return None

    with torch.no_grad():
      baseline = self.backend.value(anchor)
    advantage = best_return - baseline
    if not math.isfinite(advantage) or advantage <= 0:
      return None

    prefix = winner.transitions[: min(self.mode_steps, candidates.actions.shape[1])]
    observations = [item.actor_obs for item in prefix if item.actor_obs is not None]
    if len(observations) != len(prefix):
      raise ValueError("eligible verified prefixes require pre-step actor observations")
    return TeacherPrefix(
      snapshot_id=anchor.snapshot_id,
      task_id=anchor.task_id,
      mode=int(candidates.mode_ids[winner.candidate_index].item()),
      advantage=advantage,
      observations=torch.stack(observations),
      actions=torch.stack([item.action for item in prefix]),
    )
