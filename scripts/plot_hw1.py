#!/usr/bin/env python3
"""Plot the objective gap against four measures of progress.

    python3 scripts/plot_hw1.py results/hw1.jsonl -o figures/hw1.png

Four panels, one per x axis. Iterations, effective data passes, estimated
arithmetic work, and wall-clock seconds. A method can lead on one panel and
trail on another, and the point of the figure is that the four panels are not
redundant.

The band around each curve is the range across repetitions at each checkpoint.
A curve drawn from a single run has no band and should not be presented as
though the spread were known.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AXES = [
    ("iteration", "iterations"),
    ("effective_passes", "effective data passes"),
    ("flops", "estimated flops"),
    ("seconds", "wall-clock seconds"),
]


def load(paths):
    rows = []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def group(rows, by=("method", "config.scale")):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r.get(k) for k in by)].append(r)
    return out


def series(rows, xkey):
    """Return x, median gap, low, high, aggregated across repetitions."""
    per_iter = defaultdict(list)
    for r in rows:
        per_iter[r["iteration"]].append((r.get(xkey, 0.0) or 0.0, r["gap"]))
    xs, med, lo, hi = [], [], [], []
    for it in sorted(per_iter):
        pairs = per_iter[it]
        gaps = np.array([g for _, g in pairs])
        xs.append(float(np.median([x for x, _ in pairs])))
        med.append(float(np.median(gaps)))
        lo.append(float(gaps.min()))
        hi.append(float(gaps.max()))
    return np.array(xs), np.array(med), np.array(lo), np.array(hi)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+")
    p.add_argument("-o", "--out", default="figures/hw1.png")
    p.add_argument("--title", default="objective gap against four progress axes")
    args = p.parse_args(argv)

    rows = load(args.paths)
    if not rows:
        print("no records", file=sys.stderr)
        return 1
    groups = group(rows)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (xkey, xlabel) in zip(axes.ravel(), AXES):
        drew = False
        for key, grows in sorted(groups.items()):
            x, med, lo, hi = series(grows, xkey)
            if not len(x) or np.allclose(x, 0):
                continue
            label = " ".join(str(k) for k in key if k is not None)
            line, = ax.plot(x, med, marker="", label=label)
            ax.fill_between(x, lo, hi, alpha=0.2, color=line.get_color())
            drew = True
        ax.set_xlabel(xlabel)
        ax.set_ylabel("f(w) - f(w*)")
        ax.set_yscale("log")
        if xkey in ("iteration", "flops"):
            ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        if not drew:
            ax.text(0.5, 0.5, f"no data for {xkey}\n(is the cost model written?)",
                    ha="center", va="center", transform=ax.transAxes)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(args.title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
