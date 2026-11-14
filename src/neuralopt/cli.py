"""Command line entry point."""

from __future__ import annotations

import argparse

from .tasks import task_split
from .training import save_checkpoint, train


def main() -> None:
    parser = argparse.ArgumentParser(prog="neuralopt")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train-demo")
    p.add_argument("--output", default="neuralopt-demo.npz")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "train-demo":
        records = task_split(args.seed, 20, 4)
        params, losses = train(
            [r.problem for r in records if r.split == "train"], args.seed, args.steps
        )
        from .model import ModelConfig

        save_checkpoint(
            args.output,
            params,
            ModelConfig(),
            {"seed": args.seed, "steps": args.steps, "final_loss": losses[-1]},
        )
        print(f"saved {args.output}; loss {losses[0]:.6g} -> {losses[-1]:.6g}")
