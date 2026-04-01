#!/bin/bash
#SBATCH --job-name=run_grid_three_search
#SBATCH --time=04:00:00
#SBATCH --partition=shared
#SBATCH --account=blangme2
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shwang45@jhu.edu

set -euxo pipefail

# 3-k grid search via grid_search.py (forward any extra flags to Python).
#
# Usage (from anywhere):
#   sbatch /path/to/grid_search_slurm.sh TREE_PATH DIR_PATH K_MAIN [EXTRA grid_search.py ARGS...]
#
# Examples:
#   sbatch bin/grid_search_slurm.sh \
#     data/neisseriaceae/f__Neisseriaceae.nwk \
#     data/neisseriaceae/dashingfull_genomes_ncbi \
#     20 --threads 16 --json-out ./neis_summary.json --save-npy ./neis_tensor.npy
#
#   sbatch bin/grid_search_slurm.sh tree.nwk /path/to/dashing_dir 24 \
#     --s 3 --k-low 6 --k-high 40 --w-step 0.1 --threads "${SLURM_CPUS_PER_TASK:-16}"
#
# Optional: export PYTHON=/path/to/python (default: ~/miniconda3/envs/python310/bin/python)

JOBID="${SLURM_JOB_ID:-${SLURM_JOBID:-local}}"
echo "Starting grid search job ${JOBID}"
echo "Executing on: $(hostname)"

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 TREE_PATH DIR_PATH K_MAIN [--threads N] [--json-out PATH] [--save-npy PATH] [other grid_search.py options]" >&2
  exit 1
fi

TREE_PATH="$1"
DIR_PATH="$2"
K_MAIN="$3"
shift 3

GRID_SEARCH_PY="./grid_search.py"
PYTHON="${HOME}/miniconda3/envs/python310/bin/python"

if [ ! -f "${GRID_SEARCH_PY}" ]; then
  echo "grid_search.py not found at ${GRID_SEARCH_PY}" >&2
  exit 1
fi

"${PYTHON}" "${GRID_SEARCH_PY}" \
  --tree-path "${TREE_PATH}" \
  --dir-path "${DIR_PATH}" \
  --k-main "${K_MAIN}" \
  "$@"

echo "Finished job ${JOBID}"
