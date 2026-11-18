"""Inspect ZeroMimic's task configuration without loading the physics backend."""

import argparse
import json

from supermimic.zero.tasks import TASK_SPECS, TaskId


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--describe", action="store_true", help="print the scaffold configuration"
  )
  parser.add_argument(
    "--tasks",
    nargs="+",
    type=TaskId,
    default=list(TaskId),
    choices=list(TaskId),
    help="tasks in the training configuration",
  )
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", default=0, type=int)
  args = parser.parse_args()
  if not args.describe:
    parser.error(
      "G1 training requires a mjlab backend for ZeroMimicRunner. "
      "Use --describe to inspect the scaffold configuration."
    )
  config = {
    "device": args.device,
    "seed": args.seed,
    "tasks": {
      task.value: {
        "deadline_s": TASK_SPECS[task].deadline_s,
        "planning_seconds": TASK_SPECS[task].planning_seconds,
      }
      for task in dict.fromkeys(args.tasks)
    },
  }
  print(json.dumps(config, indent=2))


if __name__ == "__main__":
  main()
