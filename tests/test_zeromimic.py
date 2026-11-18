"""CPU contracts for ZeroMimic, independent of the G1 simulator."""

import unittest

import torch

from supermimic.zero.policy import td_lambda_targets
from supermimic.zero.replay import PhysicalReplay
from supermimic.zero.runner import Collection, VerifiedBranch, ZeroMimicRunner
from supermimic.zero.search import SearchConfig, SearchPlanner
from supermimic.zero.sim import PhysicalTransition, SimulationSnapshot
from supermimic.zero.tasks import (
  TASK_SPECS,
  BoxGeometry,
  GetUpGeometry,
  PhysicsFacts,
  RobotInitialState,
  SceneSpec,
  TaskId,
  TaskState,
  TunnelGeometry,
  TunnelSegment,
)
from supermimic.zero.world_model import ModelRollout


class ContractTests(unittest.TestCase):
  def test_box_side_bypass_cannot_succeed(self):
    spec = TASK_SPECS[TaskId.BOX]
    facts = PhysicsFacts(
      1.0,
      0.0,
      0.0,
      0.0,
      foot_ground_support=True,
      fully_past_box=True,
      in_box_landing_zone=True,
    )
    state = TaskState()
    for _ in range(spec.hold_steps(0.02)):
      state = spec.evaluate(state, facts, 0.02).state
    self.assertFalse(state.success)
    self.assertEqual(state.stable_steps, 0)

  def test_last_reward_kept_and_tail_masked_on_termination(self):
    rewards = torch.tensor([[[1.0, 2.0, 50.0]]])
    rollout = ModelRollout(
      rewards, torch.tensor([[[0.0, 1.0, 0.0]]]), torch.tensor([[100.0]])
    )
    torch.testing.assert_close(
      rollout.pessimistic_return(gamma=0.5, beta=1.0), torch.tensor([2.0])
    )

  def test_critic_bootstraps_truncation_without_crossing_boundary(self):
    rewards = torch.tensor([[1.0], [10.0]])
    next_values = torch.tensor([[4.0], [99.0]])
    targets = td_lambda_targets(
      rewards,
      next_values,
      torch.tensor([[False], [True]]),
      torch.tensor([[True], [False]]),
      gamma=0.5,
      lambda_=1.0,
    )
    torch.testing.assert_close(targets, torch.tensor([[3.0], [10.0]]))


class RoundTests(unittest.TestCase):
  def test_each_task_routes_verified_actions_into_teachers(self):
    backend = ToyBackend()
    replay = PhysicalReplay(30)
    planner = SearchPlanner(
      SearchConfig(
        population=8, iterations=2, refine_elites=2, refine_steps=2, verify_count=2
      )
    )
    runner = ZeroMimicRunner(
      backend, replay, planner, generator=torch.Generator().manual_seed(0)
    )
    stats = runner.run_round()
    self.assertEqual(stats.anchored_searches, len(TaskId))
    self.assertEqual(stats.teacher_prefixes, len(TaskId))
    self.assertEqual({teacher.task_id for teacher in backend.teachers}, set(TaskId))
    self.assertGreater(stats.verified_steps, 0)
    self.assertEqual(len(replay), stats.verified_steps)


class ToyBackend:
  step_dt = 0.2

  def collect(self, tasks):
    geometries = {
      TaskId.GET_UP: GetUpGeometry("left", 0.5),
      TaskId.TUNNEL: TunnelGeometry(0.9, (TunnelSegment(1.0, 0.6),)),
      TaskId.BOX: BoxGeometry(0.6, 0.5, 0.9, 0.9),
    }
    initial = RobotInitialState(
      (0.0, 0.0, 0.8),
      (1.0, 0.0, 0.0, 0.0),
      (0.0,),
      (0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0),
      (0.0,),
    )
    anchors = tuple(
      SimulationSnapshot(
        snapshot_id=str(task),
        episode_id=f"episode-{task}",
        scene=SceneSpec(task, str(task), geometries[task], 0, initial),
        task_state=TaskState(),
        state=torch.zeros(1),
        prev_action=torch.zeros(1),
        actor_history=torch.zeros(4, 1),
        applied_action_prefix=(),
      )
      for task in tasks
    )
    return Collection((), (), anchors)

  def fit_world_model(self, replay):
    return True

  def seed_actions(self, snapshot, horizon):
    return torch.zeros(2, horizon, 1)

  def action_limits(self, snapshot):
    return torch.ones(1), torch.full((1,), 0.3)

  def rollout_model(self, snapshot, actions):
    reward = -(actions[..., 0] - 0.4).square().unsqueeze(0)
    return ModelRollout(
      reward, torch.zeros_like(reward), reward.new_zeros(reward.shape[:2])
    )

  def refinement_mask(self, snapshot, actions, mode_ids):
    return torch.ones(actions.shape[:2], dtype=torch.bool)

  def verify(self, snapshot, candidates):
    return tuple(
      VerifiedBranch(
        candidate_index=index,
        transitions=(
          PhysicalTransition(
            task_id=snapshot.task_id,
            episode_id=f"{snapshot.episode_id}/{index}",
            scene_id=snapshot.scene_id,
            step_index=0,
            state=snapshot.state,
            next_state=snapshot.state,
            action=sequence[0],
            prev_action=snapshot.prev_action,
            reward=1.0,
            terminated=True,
            truncated=False,
            success=True,
            actor_obs=torch.zeros(1),
          ),
        ),
        feasible=True,
      )
      for index, sequence in enumerate(candidates.actions)
    )

  def value(self, snapshot):
    return 0.0

  def update_actor(self, replay, teachers):
    self.teachers = teachers


if __name__ == "__main__":
  unittest.main()
