#!/bin/bash
#SBATCH --job-name=run_gtdb_k_range
#SBATCH --time=1:00:00
#SBATCH --partition=shared
#SBATCH --account=blangme2
#SBATCH --mem=12G
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=fail
#SBATCH --mail-user=shwang45@jhu.edu

# Run array job: sbatch gtdb_k_range_slurm.sh k_min k_max input_paths output_path output_prefix [threads]
# Example usage with k_min=5 and k_max=64:
# > sbatch gtdb_k_range_slurm.sh 5 64 input_paths.txt /path/to/output test_prefix 8

echo "Starting moni job $SLURM_JOBID, task ID: $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

# Inputs:
#   k_min, k_max           : integer k range (inclusive)
#   input_paths            : input file(s) or list file for dashing (-F)
#   output_path            : directory to write outputs into
#   output_prefix          : prefix for output file names
#   threads (optional)     : number of threads (default: 8)

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 k_min k_max input_paths output_path output_prefix [threads]" >&2
  exit 1
fi

k_min="$1"
k_max="$2"
input_paths="$3"
output_path="$4"
output_prefix="$5"
threads="${6:-8}"

mkdir -p "$output_path"

# Run
for ((k = k_min; k <= k_max; k++)); do
  dashing dist --use-full-khash-sets --use-nthash -k "$k" -p "$threads" --full-tsv \
    -F "$input_paths" \
    -O "${output_path}/${output_prefix}_k${k}.txt"
done

echo "Finished with job $SLURM_JOBID"
