#!/bin/bash
#SBATCH --job-name=stability_2k
#SBATCH --time=24:00:00
#SBATCH --partition=shared
#SBATCH --account=blangme2
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail
# sbatch copies this file into its spool: use an explicit project path.
STABILITY_BIN="${STABILITY_BIN:-/scratch4/blangme2/sjhwang/ok_k_mer/bin/stability}"
STABILITY_PYTHON="${STABILITY_PYTHON:-${HOME}/miniconda3/envs/python310/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [[ "${1:-}" == "--manifest" ]]; then
    exec "$STABILITY_PYTHON" "$STABILITY_BIN/stability.py" run-task --manifest "$2"
fi
exec "$STABILITY_PYTHON" "$STABILITY_BIN/grid_search_2k.py" "$@"
