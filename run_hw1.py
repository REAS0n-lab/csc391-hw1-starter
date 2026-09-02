#!/usr/bin/env python3
"""Run one HW1 configuration and append its history to a JSON Lines file.

    python3 run_hw1.py --method sgd --scale small --repeat 3
    python3 run_hw1.py --method rcd --scale medium --repeat 3 --gap-tol 1e-8

Each repetition uses a different seed for the method while holding the problem
fixed, so the spread across repetitions measures method and machine
variability rather than problem variability.
"""

import argparse
import sys

from src import problems
from src.harness import run, write_jsonl
from src.rcd import RCD
from src.sgd import SGD


def build(method_name, problem, seed, args):
    if method_name == "sgd":
        return SGD(problem, step0=args.step0, schedule=args.schedule,
                   batch_size=args.batch_size,
                   with_replacement=args.with_replacement, seed=seed)
    if method_name == "rcd":
        return RCD(problem, sampling=args.sampling,
                   step_scale=args.step_scale, seed=seed)
    raise SystemExit(f"unknown method {method_name}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=["sgd", "rcd"])
    p.add_argument("--scale", default="small", choices=sorted(problems.SCALES))
    p.add_argument("--problem-seed", type=int, default=0)
    p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument("--repeat", type=int, default=3,
                   help="independent runs, at least 3 is required")
    p.add_argument("--max-epochs", type=float, default=30)
    p.add_argument("--gap-tol", type=float, default=None)
    p.add_argument("--grad-tol", type=float, default=None)
    p.add_argument("--time-budget", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--schedule", default="inverse",
                   choices=["constant", "inverse", "sqrt"])
    p.add_argument("--step0", type=float, default=None)
    p.add_argument("--with-replacement", action="store_true")
    p.add_argument("--sampling", default="uniform",
                   choices=["uniform", "lipschitz"])
    p.add_argument("--step-scale", type=float, default=1.0)
    p.add_argument("--out", default="results/hw1.jsonl")
    args = p.parse_args(argv)

    problem = problems.by_scale(args.scale, seed=args.problem_seed, lam=args.lam)
    print(problem.describe())

    results = []
    for rep in range(args.repeat):
        method = build(args.method, problem, seed=1000 + rep, args=args)
        r = run(method, problem, max_epochs=args.max_epochs,
                gap_tol=args.gap_tol, grad_tol=args.grad_tol,
                time_budget=args.time_budget, seed=1000 + rep)
        r.config["repeat"] = rep
        r.config["scale"] = args.scale
        last = r.checkpoints[-1]
        print(f"  rep {rep}  epochs {last.epoch:6.2f}  gap {last.gap:.3e}  "
              f"grad {last.grad_norm:.3e}  {last.seconds:7.3f} s  "
              f"stopped by {r.stopped_by}")
        results.append(r)

    path = write_jsonl(results, args.out)
    print(f"wrote {sum(len(r.checkpoints) for r in results)} checkpoints to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
