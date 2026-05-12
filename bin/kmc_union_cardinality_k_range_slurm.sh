#!/bin/bash
#SBATCH --job-name=run_kmc3_union_cardinality
#SBATCH --time=24:00:00
#SBATCH --partition=shared
#SBATCH --account=blangme2
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shwang45@jhu.edu

# set -euxo pipefail
set -euo pipefail

get_total_kmers() {
  local db_prefix="$1"
  kmc_tools info "$db_prefix" | awk -F':' '/^total k-mers/ {gsub(/[[:space:]]/, "", $2); print $2 + 0; found=1} END {if (!found) exit 1}'
}

# Run:
#   sbatch kmc_union_cardinality_k_range_slurm.sh \
#     k_min k_max input_paths.txt output_path output_prefix [threads]
#
# input_paths.txt must contain one sequence file path per line.
# For each k in [k_min, k_max], this script:
#   1) counts k-mers for every input with KMC3
#   2) computes all pairwise unions
#   3) writes one square matrix (N x N) of union cardinalities

echo "Starting moni job $SLURM_JOBID"
echo "Executing on the machine:" $(hostname)

# Inputs:
#   k_min, k_max           : integer k range (inclusive)
#   input_paths            : text file with one input sequence file path per line
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

if ! [[ "$k_min" =~ ^[0-9]+$ && "$k_max" =~ ^[0-9]+$ ]]; then
  echo "k_min and k_max must be integers" >&2
  exit 1
fi

if (( k_min > k_max )); then
  echo "k_min must be <= k_max" >&2
  exit 1
fi

if [ ! -s "$input_paths" ]; then
  echo "input_paths file does not exist or is empty: $input_paths" >&2
  exit 1
fi

mkdir -p "$output_path"
tmp_root="${output_path}/kmc_tmp_${SLURM_JOBID}"
mkdir -p "$tmp_root"

mapfile -t samples < "$input_paths"
if [ "${#samples[@]}" -eq 0 ]; then
  echo "No input files listed in $input_paths" >&2
  exit 1
fi

for ((k = k_min; k <= k_max; k++)); do
  k_dir="${output_path}/${output_prefix}_k${k}"
  db_dir="${k_dir}/db"
  op_dir="${k_dir}/ops"
  k_tmp="${tmp_root}/k${k}"
  mkdir -p "$db_dir" "$op_dir" "$k_tmp"

  db_list="${k_dir}/db_list.txt"
  : > "$db_list"
  labels_file="${k_dir}/labels.txt"
  : > "$labels_file"

  for fasta in "${samples[@]}"; do
    if [ -z "$fasta" ]; then
      continue
    fi
    if [ ! -f "$fasta" ]; then
      echo "Input file not found: $fasta" >&2
      exit 1
    fi

    base="$(basename "$fasta")"
    safe_base="${base//[^[:alnum:]_.-]/_}"
    db_prefix="${db_dir}/${safe_base}"
    echo "$db_prefix" >> "$db_list"
    echo "$safe_base" >> "$labels_file"

    # -fm uses multi-FASTA/FASTQ parser; use canonical k-mers (-b) default settings otherwise.
    kmc -k"${k}" -ci1 -fm -t"${threads}" "$fasta" "$db_prefix" "$k_tmp"
  done

  mapfile -t dbs < "$db_list"
  mapfile -t labels < "$labels_file"
  n="${#dbs[@]}"
  if [ "$n" -eq 0 ]; then
    echo "No valid input files listed in $input_paths" >&2
    exit 1
  fi

  matrix_file="${output_path}/${output_prefix}_union_matrix_k${k}.tsv"
  declare -A pair_card=()

  for ((i = 0; i < n; i++)); do
    for ((j = i; j < n; j++)); do
      if (( i == j )); then
        card="$(get_total_kmers "${dbs[$i]}")"
      else
        pair_union="${op_dir}/union_${i}_${j}"
        kmc_tools simple "${dbs[$i]}" "${dbs[$j]}" union "$pair_union" > /dev/null
        card="$(get_total_kmers "$pair_union")"
        rm -f "${pair_union}.kmc_pre" "${pair_union}.kmc_suf"
      fi
      pair_card["${i},${j}"]="$card"
      if (( i != j )); then
        pair_card["${j},${i}"]="$card"
      fi
    done
  done

  {
    printf "sample"
    for label in "${labels[@]}"; do
      printf "\t%s" "$label"
    done
    printf "\n"
    for ((i = 0; i < n; i++)); do
      printf "%s" "${labels[$i]}"
      for ((j = 0; j < n; j++)); do
        printf "\t%s" "${pair_card["${i},${j}"]}"
      done
      printf "\n"
    done
  } > "$matrix_file"

done

rm -rf "$tmp_root"

echo "Finished with job $SLURM_JOBID"
