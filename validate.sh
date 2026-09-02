#!/bin/bash
# Environment validation for the CSC 391/691 HW1 starter.
#
#   ./validate.sh            offline checks, safe on a login node
#   ./validate.sh --submit   also submit the sweep array
#
# The offline checks run a small problem end to end in a few seconds. They do
# not touch the scheduler.

set -uo pipefail
cd "$(dirname "$0")"
PASS=0; FAIL=0
ok()   { echo "[ok  ] $*"; PASS=$((PASS+1)); }
bad()  { echo "[FAIL] $*"; FAIL=$((FAIL+1)); }
info() { echo "[info] $*"; }

echo "=== CSC 391/691 HW1 starter validation ==="
echo "host $(hostname)   date $(date -Iseconds)"
echo

if source deac/site.sh 2>/dev/null; then ok "deac/site.sh sources"; else bad "deac/site.sh sources"; fi
deac_load_cpu 2>/dev/null || info "deac_load_cpu reported an error, continuing"

for mod in numpy matplotlib; do
  if python3 -c "import $mod" 2>/dev/null; then ok "$mod importable"; else bad "$mod importable"; fi
done

if python3 - <<'PY'
import sys
sys.path.insert(0, ".")
import numpy as np
from src import objective as obj, problems
from src.harness import run
from src.rcd import RCD
from src.sgd import SGD

p = problems.synthetic(n=300, d=30, condition=10.0, seed=0, lam=1e-2)
w_star = obj.solve_exact(p)
assert np.linalg.norm(obj.gradient(p, w_star)) < 1e-9, "exact solve is wrong"
r = run(RCD(p, seed=0), p, max_epochs=200, checkpoints_per_epoch=1)
assert r.checkpoints[-1].gap < 1e-10, f"rcd did not converge, gap {r.checkpoints[-1].gap}"
s = run(SGD(p, seed=0, batch_size=8), p, max_epochs=30, checkpoints_per_epoch=1)
assert s.checkpoints[-1].gap < s.checkpoints[0].gap / 5, "sgd did not make progress"
print(f"       rcd gap {r.checkpoints[-1].gap:.2e}   sgd gap {s.checkpoints[-1].gap:.2e}")
PY
then ok "both methods converge on a small problem"; else bad "both methods converge on a small problem"; fi

if timeout 900 python3 run_hw1.py --method rcd --scale small --repeat 2 \
     --max-epochs 5 --out results/validate.jsonl >/dev/null 2>&1; then
  ok "run_hw1.py end to end"
else
  bad "run_hw1.py end to end"
fi

if python3 scripts/plot_hw1.py results/validate.jsonl -o figures/validate.png >/dev/null 2>&1; then
  ok "plotting script produced figures/validate.png"
else
  bad "plotting script produced figures/validate.png"
fi
rm -f results/validate.jsonl figures/validate.png

info "the two cost functions in src/costs.py raise NotImplementedError by design"
info "students write them, and the flops panel of the figure stays empty until they do"

for tool in sbatch sacct sinfo; do
  if command -v $tool >/dev/null; then ok "$tool on PATH"; else bad "$tool on PATH"; fi
done

if [ "${1:-}" = "--submit" ]; then
  echo
  if JOB=$(./submit.sh cpu jobs/hw1_sweep.slurm 2>&1); then
    ok "sweep submitted, id $JOB"
  else
    bad "sweep submitted ($JOB)"
  fi
fi

echo
echo "passed $PASS   failed $FAIL"
[ "$FAIL" -eq 0 ]
