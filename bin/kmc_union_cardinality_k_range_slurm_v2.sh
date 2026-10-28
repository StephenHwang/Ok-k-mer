#!/bin/bash
#SBATCH --job-name=run_kmc3_union_cardinality_v2
#SBATCH --time=24:00:00
#SBATCH --partition=shared
#SBATCH --account=blangme2
#SBATCH --mem=24G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shwang45@jhu.edu

# v2 changes vs. kmc_union_cardinality_k_range_slurm.sh:
#   1) Pair-union loop runs in parallel (job-slot semaphore via `wait -n`)
#   2) Cores bumped 8 -> 24; memory dropped 64G -> 24G. With default
#      threads_per_op=2 this gives pair_jobs = build_jobs = 12. KMC builds
#      are capped at -m2 per process (KMC's documented minimum), so the
#      build-phase peak is ~12 * 2 = 24 GB worst case. kmc_tools simple
#      uses only small sort buffers (~100-300 MB each, ~3-4 GB peak across
#      12 concurrent unions). If you raise threads_per_op (and so lower
#      build_jobs), the build-phase peak drops accordingly. Tunable via
#      threads_per_op / pair_jobs / build_jobs.
#   3) ALL intermediate KMC DBs live on local node storage (${TMPDIR:-/tmp});
#      only the final TSV is copied to $output_path
#   4) (No SLURM array here — see _array.sh variant if/when added)
#   5) Per-genome KMC build step also runs in parallel

set -euo pipefail

get_total_kmers() {
  local db_prefix="$1"
  kmc_tools info "$db_prefix" \
    | awk -F':' '/^total k-mers/ {gsub(/[[:space:]]/, "", $2); print $2 + 0; found=1} END {if (!found) exit 1}'
}

# Usage:
#   sbatch kmc_union_cardinality_k_range_slurm_v2.sh \
#     k_min k_max input_paths.txt output_path output_prefix \
#     [threads_per_op] [pair_jobs] [build_jobs]
#
# threads_per_op : -t passed to each kmc / kmc_tools call    (default 2)
# pair_jobs      : number of concurrent pair-union ops       (default = $SLURM_CPUS_PER_TASK / threads_per_op)
# build_jobs     : number of concurrent kmc build ops        (default = pair_jobs)

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 k_min k_max input_paths output_path output_prefix [threads_per_op] [pair_jobs] [build_jobs]" >&2
  exit 1
fi

k_min="$1"
k_max="$2"
input_paths="$3"
output_path="$4"
output_prefix="$5"
threads_per_op="${6:-2}"
ncpus="${SLURM_CPUS_PER_TASK:-$(nproc)}"
pair_jobs="${7:-$(( ncpus / threads_per_op ))}"
build_jobs="${8:-$pair_jobs}"
(( pair_jobs  < 1 )) && pair_jobs=1
(( build_jobs < 1 )) && build_jobs=1

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

echo "Starting v2 job ${SLURM_JOBID:-local} on $(hostname)"
echo "  cpus=$ncpus  threads_per_op=$threads_per_op  pair_jobs=$pair_jobs  build_jobs=$build_jobs"

mkdir -p "$output_path"

# All intermediate KMC DBs and pair outputs go to local node storage.
# Final per-k matrix TSVs are the only thing copied back to $output_path.
work_root="${TMPDIR:-/tmp}/kmc_${SLURM_JOBID:-$$}"
mkdir -p "$work_root"
trap 'rm -rf "$work_root"' EXIT

mapfile -t samples < "$input_paths"
if [ "${#samples[@]}" -eq 0 ]; then
  echo "No input files listed in $input_paths" >&2
  exit 1
fi

# Helpers used inside background subshells. Keep them dependency-free so they
# work regardless of `export -f` semantics across bash versions.
build_one() {
  local fasta="$1" db_prefix="$2" k="$3" tmp="$4" t="$5"
  # -m2 caps each kmc invocation at 2 GB RAM (KMC's documented minimum).
  # Cyano genomes (~7 MB) need only a few hundred MB in practice, so this
  # is mostly defensive against the tighter --mem allocation. With
  # build_jobs concurrent builds the build-phase peak is ~build_jobs * 2 GB.
  # kmc flags:
  #   -k"${k}"   : Sets the k-mer size to $k
  #   -ci1       : Minimum k-mer occurrence count = 1 (every k-mer counted if seen at least once)
  #   -cs1       : Maximum k-mer occurrence count = 1 (all counts above 1 stored as 1; makes DB presence/absence)
  #                 unset this flag
  #   -m2        : Maximum RAM in GB for this process is 2GB
  #   -fm        : Input is in FASTA/FASTQ multi-line format (for genome/metagenome input)
  #   -t"${t}"   : Number of threads = $t
  #   "$fasta"   : Input file path
  #   "$db_prefix": Output prefix for the KMC DB
  #   "$tmp"     : Temporary directory for KMC internals
  # Suppresses output by redirecting stdout to /dev/null.
  kmc -k"${k}" -ci1 -m2 -fm -t"${t}" "$fasta" "$db_prefix" "$tmp" > /dev/null
}

# Computes |A ∪ B| for one pair, appends "i\tj\tcard" to $idx_file, then
# deletes the temporary union DB so /tmp doesn't fill up.
union_one() {
  local a="$1" b="$2" out="$3" idx_file="$4" i="$5" j="$6" t="$7"
  kmc_tools -t"${t}" simple "$a" "$b" union "$out" > /dev/null
  local c
  c="$(kmc_tools info "$out" \
        | awk -F':' '/^total k-mers/ {gsub(/[[:space:]]/,"",$2); print $2+0; exit}')"
  rm -f "${out}.kmc_pre" "${out}.kmc_suf"
  printf "%s\t%s\t%s\n" "$i" "$j" "$c" >> "$idx_file"
}

for ((k = k_min; k <= k_max; k++)); do
  k_dir="${work_root}/k${k}"
  db_dir="${k_dir}/db"
  op_dir="${k_dir}/ops"
  k_tmp="${k_dir}/tmp"
  mkdir -p "$db_dir" "$op_dir" "$k_tmp"

  # ---- 1) Plan per-genome builds and label/db lists ----
  build_list="${k_dir}/build_list.txt" ; : > "$build_list"
  labels_file="${k_dir}/labels.txt"    ; : > "$labels_file"
  db_list="${k_dir}/db_list.txt"       ; : > "$db_list"

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
    printf "%s\t%s\n" "$fasta" "$db_prefix" >> "$build_list"
    echo "$db_prefix" >> "$db_list"
    echo "$safe_base" >> "$labels_file"
  done

  n_build=$(wc -l < "$build_list")
  echo "[k=$k] Building $n_build per-genome KMC DBs ($build_jobs concurrent, -t$threads_per_op)"

  job_count=0
  while IFS=$'\t' read -r fasta db_prefix; do
    build_one "$fasta" "$db_prefix" "$k" "$k_tmp" "$threads_per_op" &
    if (( ++job_count >= build_jobs )); then
      wait -n
      (( --job_count ))
    fi
  done < "$build_list"
  wait

  mapfile -t dbs    < "$db_list"
  mapfile -t labels < "$labels_file"
  n="${#dbs[@]}"
  if [ "$n" -eq 0 ]; then
    echo "No valid input files listed in $input_paths" >&2
    exit 1
  fi

  # ---- 2) Diagonal entries: |A| via kmc_tools info, in parallel ----
  diag_file="${k_dir}/diag.tsv" ; : > "$diag_file"
  echo "[k=$k] Diagonal cardinalities for $n DBs"
  job_count=0
  for ((i = 0; i < n; i++)); do
    {
      c="$(get_total_kmers "${dbs[$i]}")"
      printf "%s\t%s\n" "$i" "$c" >> "$diag_file"
    } &
    if (( ++job_count >= pair_jobs )); then
      wait -n
      (( --job_count ))
    fi
  done
  wait

  # ---- 3) Off-diagonal pair unions in parallel ----
  pairs_file="${k_dir}/pairs.tsv" ; : > "$pairs_file"
  total_pairs=$(( n * (n - 1) / 2 ))
  echo "[k=$k] Computing $total_pairs pair unions ($pair_jobs concurrent, -t$threads_per_op)"

  pair_idx=0
  job_count=0
  for ((i = 0; i < n; i++)); do
    for ((j = i + 1; j < n; j++)); do
      out="${op_dir}/u_${i}_${j}"
      union_one "${dbs[$i]}" "${dbs[$j]}" "$out" "$pairs_file" "$i" "$j" "$threads_per_op" &
      if (( ++job_count >= pair_jobs )); then
        wait -n
        (( --job_count ))
      fi
      if (( ++pair_idx % 5000 == 0 )); then
        echo "  [k=$k] $pair_idx / $total_pairs pair unions issued"
      fi
    done
  done
  wait

  # ---- 4) Assemble symmetric matrix and write to $output_path ----
  matrix_file="${output_path}/${output_prefix}_union_matrix_k${k}.tsv"
  awk -v n="$n" -v diag="$diag_file" -v pairs="$pairs_file" -v labels_f="$labels_file" '
    BEGIN {
      i = 0
      while ((getline ln < labels_f) > 0) { lab[i++] = ln }
      close(labels_f)

      while ((getline ln < diag) > 0) {
        split(ln, a, "\t"); m[a[1] "," a[1]] = a[2]
      }
      close(diag)

      while ((getline ln < pairs) > 0) {
        split(ln, a, "\t")
        m[a[1] "," a[2]] = a[3]
        m[a[2] "," a[1]] = a[3]
      }
      close(pairs)

      printf "sample"
      for (i = 0; i < n; i++) printf "\t%s", lab[i]
      printf "\n"

      for (i = 0; i < n; i++) {
        printf "%s", lab[i]
        for (j = 0; j < n; j++) printf "\t%s", m[i "," j]
        printf "\n"
      }
    }' > "$matrix_file"

  echo "[k=$k] wrote $matrix_file"

  # Free local /tmp before moving on to the next k.
  rm -rf "$k_dir"
done

echo "Finished v2 job ${SLURM_JOBID:-local}"
