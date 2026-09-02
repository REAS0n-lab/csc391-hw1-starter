#!/bin/bash
# HW1 sweep. One array submission covers both methods at two problem scales.
#
#   ./submit.sh cpu jobs/hw1_sweep.slurm
#
# CSC 691 adds the third scale by changing --array=0-3 to --array=0-5 and
# extending the METHODS and SCALES arrays below.

#SBATCH --job-name=hw1-sweep
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/%x-%A_%a.out
#SBATCH --error=results/%x-%A_%a.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; mkdir -p results
source deac/site.sh; deac_load_cpu

# One thread. A multithreaded BLAS turns a wall-clock comparison between a
# matrix-vector method and a coordinate method into a comparison of how well
# each one saturates a thread pool, which is a different question than the one
# HW1 asks. If you want that question, change this deliberately and say so.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

METHODS=(sgd sgd rcd rcd)
SCALES=(small medium small medium)
M=${METHODS[$SLURM_ARRAY_TASK_ID]}
SC=${SCALES[$SLURM_ARRAY_TASK_ID]}

echo "task $SLURM_ARRAY_TASK_ID  method $M  scale $SC  host $(hostname)"
python3 run_hw1.py --method "$M" --scale "$SC" --repeat 3 --max-epochs 50 \
        --out "results/hw1.jsonl"
