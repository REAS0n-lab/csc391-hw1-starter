# Result schema

`run_hw1.py` appends one JSON object per checkpoint to a JSON Lines file. One
line is one measurement of one run at one point in its history.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `method` | string | `sgd` or `rcd` |
| `problem` | string | problem instance name, encodes n, d, and condition number |
| `stopped_by` | string | `max_epochs`, `gap_tol`, `grad_tol`, or `time_budget` |
| `iteration` | int | updates applied so far |
| `epoch` | float | iterations divided by updates per epoch |
| `objective` | float | f(w) at this checkpoint |
| `gap` | float | f(w) minus f(w*), using the closed-form minimizer |
| `grad_norm` | float | two-norm of the full gradient |
| `seconds` | float | cumulative time inside the method, excluding checkpoints |
| `flops` | float | cumulative estimated arithmetic, zero until the cost model is written |
| `effective_passes` | float | flops converted to equivalent passes over X |
| `config.*` | mixed | every parameter that defines the run |
| `env.*` | mixed | hostname, Python and NumPy versions, Slurm ids, thread count |

## Rules this schema enforces

**One line per checkpoint, never one line per run.** A summary row hides the
shape of the curve, and the shape is what distinguishes a method that
converged from one that stalled at a level you happened to find acceptable.

**Four progress axes on every line.** `iteration`, `effective_passes`,
`flops`, and `seconds` are recorded together so that no plot has to assume the
four are interchangeable.

**`config.repeat` distinguishes repetitions.** Three runs of the same
configuration differ only in that field. Any aggregation that ignores it
reports a spread of zero, which is not the same as having measured no spread.

**Timing excludes checkpoints.** Evaluating the objective costs a
matrix-vector product. Charging that to the method would make a frequently
checkpointed run look slower than it is.

## Reading the file

```python
import json
rows = [json.loads(line) for line in open("results/hw1.jsonl") if line.strip()]
```

Or use `scripts/plot_hw1.py`, which groups by method and scale and draws a
band across repetitions.
