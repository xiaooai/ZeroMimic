"""Episode-aware physical replay and verified teacher records."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from random import Random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from torch import Tensor

  from .sim import PhysicalTransition
  from .tasks import TaskId


@dataclass(frozen=True, slots=True)
class TeacherPrefix:
  """The executed prefix of one verified winner, with an advantage at its start."""

  snapshot_id: str
  task_id: TaskId
  mode: int
  advantage: float
  observations: Tensor
  actions: Tensor

  def __post_init__(self) -> None:
    if self.mode < 0 or not isfinite(self.advantage) or self.advantage <= 0.0:
      raise ValueError(
        "Teacher prefixes require a mode index and a finite positive advantage."
      )
    if (
      self.observations.ndim != 2
      or self.actions.ndim != 2
      or self.actions.shape[0] != self.observations.shape[0]
      or self.actions.shape[0] == 0
    ):
      raise ValueError(
        "Teacher observations and actions must share a non-empty time dimension."
      )


class PhysicalReplay:
  """A bounded store of contiguous physical fragments.

  Capacity is measured in fragments. Each fragment can start mid-episode.
  A world-model window may end at a terminal or truncation but stays within
  one fragment, so its final transition retains the pre-reset successor.
  """

  def __init__(self, capacity: int) -> None:
    if capacity <= 0:
      raise ValueError("Replay capacity must be positive.")
    self._fragments: deque[tuple[PhysicalTransition, ...]] = deque(maxlen=capacity)

  def __len__(self) -> int:
    return len(self._fragments)

  def append(self, transitions: Iterable[PhysicalTransition]) -> None:
    fragment = tuple(transitions)
    if not fragment:
      raise ValueError("A replay fragment must contain at least one transition.")
    first = fragment[0]
    for offset, step in enumerate(fragment):
      if (
        step.episode_id != first.episode_id
        or step.scene_id != first.scene_id
        or step.task_id != first.task_id
        or step.step_index != first.step_index + offset
      ):
        raise ValueError(
          "Replay fragments must contain contiguous transitions from one episode."
        )
      if offset + 1 < len(fragment) and (step.terminated or step.truncated):
        raise ValueError(
          "A transition ending an episode or segment must also end the fragment."
        )
    self._fragments.append(fragment)

  def sample_windows(
    self,
    length: int,
    batch_size: int,
    *,
    rng: Random,
    task_id: TaskId | None = None,
  ) -> list[tuple[PhysicalTransition, ...]]:
    """Sample uniformly over available windows, with replacement."""
    if length <= 0 or batch_size < 0:
      raise ValueError("Window length must be positive and batch size non-negative.")
    if batch_size == 0:
      return []
    fragments = [
      fragment
      for fragment in self._fragments
      if len(fragment) >= length and (task_id is None or fragment[0].task_id == task_id)
    ]
    if not fragments:
      return []

    counts = [len(fragment) - length + 1 for fragment in fragments]
    samples = []
    for fragment in rng.choices(fragments, weights=counts, k=batch_size):
      start = rng.randrange(len(fragment) - length + 1)
      samples.append(fragment[start : start + length])
    return samples
