#!/bin/bash
#SBATCH --job-name=stability_3k
#SBATCH --time=36:00:00
#SBATCH --partition=shared
#SBATCH --account=blangme2
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail
STABILITY_BIN="${STABILITY_BIN:-/scratch4/blangme2/sjhwang/ok_k_mer/bin/stability}"
STABILITY_PYTHON="${STABILITY_PYTHON:-${HOME}/miniconda3/envs/python310/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [[ "${1:-}" == "--manifest" ]]; then
    exec "$STABILITY_PYTHON" "$STABILITY_BIN/stability.py" run-task --manifest "$2"
fi
exec "$STABILITY_PYTHON" "$STABILITY_BIN/grid_search_3k.py" "$@"
