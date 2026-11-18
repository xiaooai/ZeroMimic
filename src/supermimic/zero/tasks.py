"""Task configuration and hard rules evaluated from solver measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class TaskId(StrEnum):
  GET_UP = "get-up"
  TUNNEL = "tunnel"
  BOX = "box"


def _positive(name: str, value: float) -> None:
  if not math.isfinite(value) or value <= 0.0:
    raise ValueError(f"{name} must be finite and positive")


def _ceil_steps(seconds: float, step_dt: float) -> int:
  _positive("step_dt", step_dt)
  steps = seconds / step_dt
  nearest = round(steps)
  if nearest > 0 and math.isclose(steps, nearest, rel_tol=0.0, abs_tol=1e-12):
    return nearest
  return math.ceil(steps)


@dataclass(frozen=True, slots=True)
class GetUpGeometry:
  """A side wall at the given distance from the scene origin."""

  wall_side: str
  wall_distance_m: float

  def __post_init__(self) -> None:
    if self.wall_side not in {"left", "right"}:
      raise ValueError("wall_side must be 'left' or 'right'")
    _positive("wall_distance_m", self.wall_distance_m)


@dataclass(frozen=True, slots=True)
class TunnelSegment:
  length_m: float
  clearance_m: float

  def __post_init__(self) -> None:
    _positive("length_m", self.length_m)
    _positive("clearance_m", self.clearance_m)


@dataclass(frozen=True, slots=True)
class TunnelGeometry:
  """Segments ordered from the entrance towards the exit."""

  width_m: float
  segments: tuple[TunnelSegment, ...]

  def __post_init__(self) -> None:
    _positive("width_m", self.width_m)
    if not self.segments:
      raise ValueError("a tunnel needs at least one ceiling segment")


@dataclass(frozen=True, slots=True)
class BoxGeometry:
  """Solid box in a laterally bounded route."""

  height_m: float
  depth_m: float
  width_m: float
  route_width_m: float

  def __post_init__(self) -> None:
    _positive("height_m", self.height_m)
    _positive("depth_m", self.depth_m)
    _positive("width_m", self.width_m)
    _positive("route_width_m", self.route_width_m)
    if self.route_width_m < self.width_m:
      raise ValueError("route_width_m must be at least width_m")


Geometry = GetUpGeometry | TunnelGeometry | BoxGeometry
Vec3 = tuple[float, float, float]

_GEOMETRY_TYPES = {
  TaskId.GET_UP: GetUpGeometry,
  TaskId.TUNNEL: TunnelGeometry,
  TaskId.BOX: BoxGeometry,
}


@dataclass(frozen=True, slots=True)
class RobotInitialState:
  """Robot reset state in scene coordinates and simulator joint order."""

  root_pos_m: Vec3
  root_quat_wxyz: tuple[float, float, float, float]
  joint_pos_rad: tuple[float, ...]
  root_linear_velocity_m_s: Vec3
  root_angular_velocity_rad_s: Vec3
  joint_velocity_rad_s: tuple[float, ...]

  def __post_init__(self) -> None:
    num_joints = len(self.joint_pos_rad)
    if not num_joints or num_joints != len(self.joint_velocity_rad_s):
      raise ValueError(
        "joint positions and velocities must have the same positive length"
      )


@dataclass(frozen=True, slots=True)
class SceneSpec:
  """Immutable parameters required to rebuild a simulator episode."""

  task_id: TaskId
  scene_id: str
  geometry: Geometry
  reset_seed: int
  initial_state: RobotInitialState
  scene_origin_m: Vec3 = (0.0, 0.0, 0.0)
  scene_yaw_rad: float = 0.0
  physics_parameters: tuple[tuple[str, float], ...] = ()

  def __post_init__(self) -> None:
    geometry_type = _GEOMETRY_TYPES[self.task_id]
    if not isinstance(self.geometry, geometry_type):
      raise ValueError(f"{self.task_id} requires {geometry_type.__name__}")
    if not self.scene_id:
      raise ValueError("scene_id must be nonempty")


@dataclass(frozen=True, slots=True)
class PhysicsFacts:
  """Measurements and thresholded contact flags from one physical control step.

  The adapter derives passage flags from collision bodies and the scene route.
  Box passage evidence comes from moving over the box footprint towards the
  far edge. All speeds are magnitudes. Contact flags summarize all substeps.
  """

  pelvis_height_ratio: float
  torso_tilt_deg: float
  root_linear_speed_m_s: float
  root_angular_speed_rad_s: float
  foot_ground_support: bool = False
  hand_torso_or_wall_support: bool = False
  entered_through_tunnel_aperture: bool = False
  fully_past_tunnel_exit: bool = False
  forbidden_ceiling_contact: bool = False
  crossed_over_box_footprint: bool = False
  fully_past_box: bool = False
  in_box_landing_zone: bool = False
  irrecoverable_violation: bool = False


@dataclass(frozen=True, slots=True)
class TaskState:
  step_index: int = 0
  entered_tunnel: bool = False
  forbidden_tunnel_contact: bool = False
  stable_steps: int = 0
  crossed_box_route: bool = False
  terminated: bool = False
  truncated: bool = False
  success: bool = False


@dataclass(frozen=True, slots=True)
class TaskEvaluation:
  state: TaskState
  reason: str | None = None

  @property
  def success(self) -> bool:
    return self.state.success

  @property
  def terminated(self) -> bool:
    return self.state.terminated

  @property
  def truncated(self) -> bool:
    return self.state.truncated


@dataclass(frozen=True, slots=True)
class PhysicsTaskSpec:
  task_id: TaskId
  deadline_s: float
  planning_seconds: float
  hold_s: float = 0.5
  min_pelvis_height_ratio: float = 0.9
  max_torso_tilt_deg: float = 20.0
  max_root_linear_speed_m_s: float = 0.25
  max_root_angular_speed_rad_s: float = 0.7

  def __post_init__(self) -> None:
    _positive("deadline_s", self.deadline_s)
    _positive("planning_seconds", self.planning_seconds)
    _positive("hold_s", self.hold_s)
    _positive("min_pelvis_height_ratio", self.min_pelvis_height_ratio)
    _positive("max_torso_tilt_deg", self.max_torso_tilt_deg)
    _positive("max_root_linear_speed_m_s", self.max_root_linear_speed_m_s)
    _positive("max_root_angular_speed_rad_s", self.max_root_angular_speed_rad_s)

  def deadline_steps(self, step_dt: float) -> int:
    return _ceil_steps(self.deadline_s, step_dt)

  def hold_steps(self, step_dt: float) -> int:
    return _ceil_steps(self.hold_s, step_dt)

  def planning_steps(self, step_dt: float, mode_steps: int = 8) -> int:
    """Round to the nearest positive multiple of a policy mode segment."""
    _positive("step_dt", step_dt)
    if mode_steps <= 0:
      raise ValueError("mode_steps must be positive")
    segments = self.planning_seconds / (step_dt * mode_steps)
    return max(1, math.floor(segments + 0.5)) * mode_steps

  def evaluate(
    self,
    state: TaskState,
    facts: PhysicsFacts,
    step_dt: float,
    *,
    external_cutoff: bool = False,
  ) -> TaskEvaluation:
    """Advance the hard task rules using measurements from the next state."""
    if state.terminated or state.truncated:
      raise ValueError("task state must be reset after the episode ends")

    deadline_steps = self.deadline_steps(step_dt)
    entered = state.entered_tunnel
    crossed_box_route = state.crossed_box_route
    forbidden_contact = state.forbidden_tunnel_contact
    stable_steps = 0
    if self.task_id == TaskId.TUNNEL:
      entered = entered or facts.entered_through_tunnel_aperture
      forbidden_contact = forbidden_contact or facts.forbidden_ceiling_contact
      success = entered and facts.fully_past_tunnel_exit and not forbidden_contact
    else:
      supporting = facts.foot_ground_support and not facts.hand_torso_or_wall_support
      still = (
        facts.root_linear_speed_m_s < self.max_root_linear_speed_m_s
        and facts.root_angular_speed_rad_s < self.max_root_angular_speed_rad_s
      )
      upright = facts.torso_tilt_deg <= self.max_torso_tilt_deg
      if self.task_id == TaskId.GET_UP:
        at_goal = facts.pelvis_height_ratio >= self.min_pelvis_height_ratio
      else:
        crossed_box_route = crossed_box_route or facts.crossed_over_box_footprint
        at_goal = (
          crossed_box_route and facts.fully_past_box and facts.in_box_landing_zone
        )
      if supporting and still and upright and at_goal:
        stable_steps = state.stable_steps + 1
      success = stable_steps >= self.hold_steps(step_dt)

    step_index = state.step_index + 1
    reason = None
    if facts.irrecoverable_violation:
      reason = "irrecoverable-violation"
    elif forbidden_contact:
      reason = "forbidden-ceiling-contact"
    elif success:
      reason = "success"
    elif step_index >= deadline_steps:
      reason = "deadline"

    terminated = reason is not None
    truncated = external_cutoff and not terminated
    next_state = TaskState(
      step_index=step_index,
      entered_tunnel=entered,
      forbidden_tunnel_contact=forbidden_contact,
      stable_steps=stable_steps,
      crossed_box_route=crossed_box_route,
      terminated=terminated,
      truncated=truncated,
      success=reason == "success",
    )
    return TaskEvaluation(next_state, "external-cutoff" if truncated else reason)


TASK_SPECS = {
  TaskId.GET_UP: PhysicsTaskSpec(TaskId.GET_UP, deadline_s=5.0, planning_seconds=1.3),
  TaskId.TUNNEL: PhysicsTaskSpec(TaskId.TUNNEL, deadline_s=12.0, planning_seconds=1.3),
  TaskId.BOX: PhysicsTaskSpec(TaskId.BOX, deadline_s=12.0, planning_seconds=2.6),
}
