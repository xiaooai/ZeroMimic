# ZeroMimic

ZeroMimic is a research project for humanoid control without motion-capture
demonstrations. It builds on [SuperMimic](https://github.com/HybridRobotics/supermimic)
and its PyTorch, mjlab, and MuJoCo Warp stack.

The design covers constrained get-up, a variable-clearance tunnel, and
adjustable box traversal with a shared policy. See the
[technical design](docs/zeromimic-design.md) for the task rules, training loop,
and simulator integration.

## Current scaffold

- Task scene definitions and hard success and episode rules.
- Physical replay and simulator contracts for pre-reset states and verified branches.
- Mode-conditioned chunk CEM, projected gradient refinement, and ensemble return scoring.
- A multi-mode policy, real-transition TD(λ) targets, verified distillation, and round orchestration.

The scaffold runs with a supplied training backend. The concrete G1 scene
builder, Warp adapter, learned dynamics training and checkpoint playback are
planned integration work.

Inspect the task configuration with Python 3.11 or newer:

```bash
PYTHONPATH=src python -m supermimic.scripts.train_zero --describe
```

Run the scaffold's CPU checks in a Python environment with PyTorch:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_zeromimic.py
```

## Inherited SuperMimic tracking

The original tracking tools use Python 3.13 and the sibling mjlab checkout.
The upstream [project page](https://supermimic.github.io) and
[paper](https://arxiv.org/abs/2610.28480) describe the tracking system.

### Upstream motion-tracking setup

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone both repos as siblings:

```bash
mkdir supermimic_ws && cd supermimic_ws
git clone https://github.com/xiaooai/ZeroMimic.git
git clone git@github.com:HybridRobotics/mjlab_research.git
cd ZeroMimic
```

### Upstream motion-tracking usage

Available task IDs:

- `SuperMimic-GenCoord-G1` — generalized coordinates representation
- `SuperMimic-MaxCoord-G1` — maximal coordinates representation

Train a world model and policy:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run src/supermimic/scripts/train.py \
  <TASK_ID> --runner.run_name <description>
```

Play back a trained model:

```bash
uv run src/supermimic/scripts/play.py \
  <TASK_ID> --wandb-path <RUN_PATH>
```

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
