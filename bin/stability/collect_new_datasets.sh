#!/bin/bash
#SBATCH --job-name=new_dataset_figures
#SBATCH --partition=shared
#SBATCH --account=blangme2
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
set -euo pipefail
export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1
cd /scratch4/blangme2/sjhwang/ok_k_mer
exec /home/shwang45/miniconda3/envs/python310/bin/python bin/stability/add_datasets.py collect
