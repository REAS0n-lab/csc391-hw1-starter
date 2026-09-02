# HW1 report

**Name.**   **Course number.** CSC 391 or CSC 691.   **Date.**

Three to four pages. The five headings below are the same five that every
graded artifact in this course uses.

---

## 1. Question and claim

State the comparison you are making and the claim your evidence could support.
Name both methods, the problem family, the hardware, the problem scales, and
the metric.

A claim at the right scope reads like this. "On synthetic ridge regression
with n = 8000, d = 400, and condition number 100, running single threaded on
one DEAC CPU core, randomized coordinate descent reached an objective gap of
1e-6 in less wall-clock time than mini-batch SGD with batch size 32 at every
one of three repetitions." A claim at the wrong scope drops one of those
clauses.

## 2. Cost model

Derive the arithmetic and memory-access cost of one SGD update and one
coordinate descent update. Put both in a common work unit and say what that
unit is.

State what the model omits. Every model in this course omits something, and
naming the omission is worth more than pretending there is none.

| Quantity | SGD update | RCD update |
|---|---|---|
| Flops | | |
| Bytes read | | |
| Bytes written | | |
| Updates per effective pass | | |

## 3. Baseline and measurement

Name the baseline configuration before any tuning. Report at least three
independent runs at each of at least two problem scales. CSC 691 adds a third
scale and reports whether the conclusion survives it.

State the stopping rule and whether it treats the two methods fairly. A gap
tolerance, a gradient tolerance, a fixed epoch budget, and a fixed time budget
are four different rules, and they can order the two methods differently.

Include the four-panel figure. Say what the band around each curve represents.

## 4. Explanation and disagreement

Where the panels disagree, explain the mechanism. "Coordinate descent needed
more iterations but each one was cheaper" is a start. What made it cheaper, in
terms of the cost model in section 2, is the answer.

Report anything that came out the wrong way round. A negative or null result
earns full credit when the investigation is sound.

## 5. Limits and next measurement

Name one plausible comparison your evidence does not support.

Name one additional measurement that would change your conclusion, and say
which way it would have to come out.

---

## Reproducibility

| Item | Value |
|---|---|
| Repository commit | |
| Slurm job ids | |
| Node type | |
| Threads per run | |
| NumPy version and BLAS backend | |
| Command lines used | |

Attach `results/hw1.jsonl`, the job scripts, and the figures.

## AI-use note

What you prompted, what you accepted, what you rejected and why. Not graded,
and not optional. If you did not use AI, write that.
