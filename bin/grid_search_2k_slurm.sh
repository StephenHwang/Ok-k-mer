#!/bin/bash
#SBATCH --job-name=run_grid_2k_search
#SBATCH --time=12:00:00
#SBATCH --partition=shared
#SBATCH --account=blangme2
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shwang45@jhu.edu
# Note: cyano took ~16hours, otherwise <12

set -euxo pipefail

# 3-k grid search via grid_search_2k.py. k_main is chosen inside Python (initial single-k Spearman search).
# Forward any extra flags to grid_search_2k.py.
#
# Usage:
#   sbatch /path/to/grid_search_slurm.sh TREE_PATH DIR_PATH [EXTRA grid_search_2k.py ARGS...]
#
# Examples:
#   sbatch bin/grid_search_slurm.sh \
#     data/neisseriaceae/f__Neisseriaceae.nwk \
#     data/neisseriaceae/dashingfull_genomes_ncbi \
#     --threads 16 --json-out ./neis_summary.json --save-npy ./neis_tensor.npy
#
# Optional: export PYTHON=/path/to/python (default: ~/miniconda3/envs/python310/bin/python)

JOBID="${SLURM_JOB_ID:-${SLURM_JOBID:-local}}"
echo "Starting grid search job ${JOBID}"
echo "Executing on: $(hostname)"

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 TREE_PATH DIR_PATH [--k-low K] [--k-high K] [--json-out PATH] [--save-npy PATH] ..." >&2
  exit 1
fi

TREE_PATH="$1"
DIR_PATH="$2"
shift 2

# BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# GRID_SEARCH_PY="${BIN_DIR}/grid_search_2k.py"
# PYTHON="${PYTHON:-${HOME}/miniconda3/envs/python310/bin/python}"
GRID_SEARCH_PY="./grid_search_2k.py"
PYTHON="${HOME}/miniconda3/envs/python310/bin/python"

if [ ! -f "${GRID_SEARCH_PY}" ]; then
  echo "grid_search_2k.py not found at ${GRID_SEARCH_PY}" >&2
  exit 1
fi

"${PYTHON}" "${GRID_SEARCH_PY}" \
  --tree-path "${TREE_PATH}" \
  --dir-path "${DIR_PATH}" \
  "$@"

echo "Finished job ${JOBID}"
