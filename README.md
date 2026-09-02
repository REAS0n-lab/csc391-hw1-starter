# CSC 391/691 HW1 starter

Working implementations of mini-batch stochastic gradient descent and
randomized coordinate descent for ridge regression, a harness that records
convergence against four different measures of progress, and the plotting and
reporting templates the assignment expects.

Both methods work. The assignment is not to make them run. It is to decide
what can be claimed once they have.

**Opens Friday 9/11. Due Wednesday 9/23 at 2:00 p.m.**

## Layout

```
src/problems.py       synthetic and LIBSVM problem instances
src/objective.py      objective, gradients, Lipschitz constants, exact solve
src/sgd.py            mini-batch SGD with three step-size schedules
src/rcd.py            randomized coordinate descent, residual kept incrementally
src/costs.py          work accounting, two functions are yours to write
src/harness.py        run loop, checkpointing, stopping rules
run_hw1.py            command-line driver, writes JSON Lines
scripts/plot_hw1.py   four-panel convergence figure with a band across runs
jobs/hw1_sweep.slurm  array submission, both methods at two scales
results/SCHEMA.md     what every field in the output file means
report/               the report template
tests/                correctness tests and the cost-model specification
```

## Quick start

```bash
python3 -m pytest tests/test_correctness.py -q
python3 run_hw1.py --method rcd --scale small --repeat 3 --max-epochs 30
python3 run_hw1.py --method sgd --scale small --repeat 3 --max-epochs 30
python3 scripts/plot_hw1.py results/hw1.jsonl -o figures/hw1.png
```

On DEAC, run the sweep as a job rather than on a login node.

```bash
./submit.sh cpu jobs/hw1_sweep.slurm
```

## What you write

**`src/costs.py`.** Two functions, `sgd_update_cost` and `rcd_update_cost`,
each returning the flops and bytes moved by a single update. Read the
implementations in `src/sgd.py` and `src/rcd.py` and count what they execute.
`tests/test_costs.py` checks how your answers scale with n, d, and batch size.
It does not check the constant, because two defensible derivations can differ
there.

Until you write them, the harness records zero flops and the third panel of
the figure is empty. That is the intended failure mode, and it is visible.

**The report.** `report/report-template.md` has the five sections.

## Two implementation choices worth knowing about

**Every accumulation is float64.** A sum of n residuals in float32 loses
accuracy exactly where the objective gap gets small, which is the region every
convergence plot here cares about. Storage precision and accumulation
precision are separate choices and `src/objective.py` makes the second one
explicitly.

**Features are centered and scaled by default.** Column scaling changes the
conditioning of the problem and therefore changes how both methods behave. A
comparison run on unscaled features measures the scaling as much as it
measures the algorithms. Pass `scale_features=False` if that is the effect you
want to study, and say so in the report.

**The residual in `rcd.py` is maintained incrementally.** Updating one
coordinate changes the residual by a scaled column, which costs O(n) rather
than the O(n d) a recomputation would cost. That difference is the reason the
method is competitive at all, and it is the first thing your cost model has to
capture.

## Problem scales

| Name | n | d | condition number |
|---|---:|---:|---:|
| `small` | 2000 | 100 | 50 |
| `medium` | 8000 | 400 | 100 |
| `large` | 32000 | 800 | 200 |

CSC 391 uses two of the three. CSC 691 uses all three and reports whether the
conclusion survives the third.

The synthetic family has a known closed-form minimizer, so every plot shows a
real objective gap f(w) minus f(w*) rather than a gap against the best value
some run happened to reach.

## Stopping rules

`run_hw1.py` accepts `--max-epochs`, `--gap-tol`, `--grad-tol`, and
`--time-budget`, and the result records which one fired. These are four
different rules and they can order the two methods differently. Applying one
common rule and arguing that it is fair to both methods is part of the
assignment.

## Threads

The sweep job sets `OMP_NUM_THREADS=1`. A multithreaded BLAS turns a
wall-clock comparison between a matrix-vector method and a coordinate method
into a comparison of how well each one saturates a thread pool, which is a
different question than the one HW1 asks. Change it deliberately if you want
that question, and record the change.
