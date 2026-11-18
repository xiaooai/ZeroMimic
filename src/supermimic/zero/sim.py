"""Simulation contracts for a future mjlab/MuJoCo Warp adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from supermimic.zero.tasks import PhysicsFacts, SceneSpec, TaskId, TaskState

if TYPE_CHECKING:
  from torch import Tensor


@dataclass(frozen=True, slots=True)
class SimulationSnapshot:
  """A branch point captured before the next action.

  The action prefix contains the commands actually applied from the reset.
  Tensors belong to this record and must not alias mutable simulator buffers.
  """

  snapshot_id: str
  episode_id: str
  scene: SceneSpec
  task_state: TaskState
  state: Tensor
  prev_action: Tensor
  actor_history: Tensor
  applied_action_prefix: tuple[Tensor, ...]
  mode_id: int | None = None
  mode_steps_remaining: int = 0

  @property
  def scene_id(self) -> str:
    return self.scene.scene_id

  @property
  def task_id(self) -> TaskId:
    return self.scene.task_id

  @property
  def step_index(self) -> int:
    return self.task_state.step_index


@dataclass(frozen=True, slots=True)
class PhysicalTransition:
  """A solver transition with the successor captured before any reset.

  `step_index` identifies the starting state within the episode. `action`
  is the projected command in policy coordinates, before scale and offset.
  """

  task_id: TaskId
  episode_id: str
  scene_id: str
  step_index: int
  state: Tensor
  next_state: Tensor
  action: Tensor
  prev_action: Tensor
  reward: float
  terminated: bool
  truncated: bool
  success: bool
  facts: PhysicsFacts | None = None
  actor_obs: Tensor | None = None
  next_actor_obs: Tensor | None = None


class Simulator(Protocol):
  """An adapter for one logical episode, which may run in a batched pool."""

  @property
  def step_dt(self) -> float:
    """Seconds per control step, including all physics substeps."""
    ...

  def reset(self, scene: SceneSpec) -> SimulationSnapshot:
    """Reset the solver, episode counters, and actor history."""
    ...

  def step(self, action: Tensor) -> PhysicalTransition:
    """Advance the solver and return the actual pre-reset outcome."""
    ...

  def snapshot(self) -> SimulationSnapshot:
    """Record the current nonterminal control boundary."""
    ...

  def replicate(self, snapshot: SimulationSnapshot) -> SimulationSnapshot:
    """Restore a branch point and verify parity before returning.

    An adapter may reset and replay the recorded action prefix on the same
    backend. It should reject a replay if state, action, task counters, or
    observation history differ beyond the configured tolerances.
    """
    ...
