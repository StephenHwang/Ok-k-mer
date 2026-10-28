#!/bin/bash
#SBATCH --job-name=s3_download
#SBATCH --time=12:00:00
#SBATCH --partition=parallel
#SBATCH --account=blangme2
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-99%25
#SBATCH --output=log/s3_download_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shwang45@jhu.edu

set -euo pipefail

# Download a large list of S3 objects in parallel using a Slurm array job.
#
# Run:
#   sbatch s3_download_array_slurm.sh input_paths.txt output_dir [parallel_per_task]
#
# input_paths.txt : one object per line. Each line is either
#                     s3://bucket/key/to/file        (full URI), or
#                     bucket/key/to/file             (no scheme), or
#                     key/to/file                    (requires S3_BUCKET env var)
# output_dir      : local directory to download into. The object key path is
#                   recreated underneath it (e.g. s3://b/a/b/c.fa -> output_dir/a/b/c.fa).
# parallel_per_task (optional): concurrent `aws s3 cp` calls per array task.
#                               Default: number of allocated CPUs.
#
# Work is split across array tasks by a simple stride: task t handles every
# line where (line_index % num_array_tasks) == t. This needs no precomputed
# counts and stays balanced regardless of how many files there are.
#
# The job is idempotent: files already present and non-empty are skipped, so a
# failed/partial run can be re-submitted (e.g. `sbatch --array=<failed ids>`)
# safely.

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 input_paths.txt output_dir [parallel_per_task]" >&2
  exit 1
fi

input_paths="$1"
output_dir="$2"
parallel_per_task="${3:-${SLURM_CPUS_PER_TASK:-8}}"

if [ ! -s "$input_paths" ]; then
  echo "input_paths file does not exist or is empty: $input_paths" >&2
  exit 1
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "This script must be run as a Slurm array job (SLURM_ARRAY_TASK_ID unset)." >&2
  echo "Submit with: sbatch $0 $input_paths $output_dir" >&2
  exit 1
fi

task_id="$SLURM_ARRAY_TASK_ID"
# SLURM_ARRAY_TASK_COUNT is the total number of tasks in the array.
num_tasks="${SLURM_ARRAY_TASK_COUNT:-$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))}"

echo "Starting S3 download job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOBID} task ${task_id}/${num_tasks}"
echo "Executing on the machine: $(hostname)"
echo "Parallel downloads per task: ${parallel_per_task}"

mkdir -p "$output_dir"

# Build this task's slice of the manifest (strip blanks/comments, then stride).
task_manifest="$(mktemp)"
trap 'rm -f "$task_manifest"' EXIT

awk -v t="$task_id" -v n="$num_tasks" '
  { line=$0; sub(/^[ \t]+/, "", line); sub(/[ \t]+$/, "", line) }
  line == "" || line ~ /^#/ { next }
  { if (idx % n == t) print line; idx++ }
' "$input_paths" > "$task_manifest"

num_files="$(wc -l < "$task_manifest")"
echo "This task will download ${num_files} file(s)."
if [ "$num_files" -eq 0 ]; then
  echo "Nothing to do for task ${task_id}."
  exit 0
fi

# Per-file download worker. Exported so xargs subshells can call it.
download_one() {
  local raw="$1"
  local out_root="$2"

  # Normalize to s3://bucket/key and extract the key for the local path.
  local uri key
  if [[ "$raw" == s3://* ]]; then
    uri="$raw"
    key="${raw#s3://}"     # bucket/key...
    key="${key#*/}"        # drop bucket -> key...
  elif [ -n "${S3_BUCKET:-}" ]; then
    uri="s3://${S3_BUCKET}/${raw#/}"
    key="${raw#/}"
  else
    # Treat the first path component as the bucket name.
    uri="s3://${raw#/}"
    key="${raw#/}"
    key="${key#*/}"
  fi

  local dest="${out_root}/${key}"

  if [ -s "$dest" ]; then
    echo "SKIP  (exists) $dest"
    return 0
  fi

  mkdir -p "$(dirname "$dest")"

  local attempt
  for attempt in 1 2 3; do
    if aws s3 cp "$uri" "$dest" --only-show-errors; then
      echo "OK    $uri -> $dest"
      return 0
    fi
    echo "RETRY ($attempt) $uri" >&2
    sleep $(( attempt * 5 ))
  done

  echo "FAIL  $uri" >&2
  return 1
}
export -f download_one
export output_dir

# Run downloads concurrently. -P sets parallelism; failures are collected so a
# single bad object does not abort the whole task, but the task exits non-zero
# if any file failed (so Slurm marks it FAILED and you can re-run that task id).
fail_log="$(mktemp)"
trap 'rm -f "$task_manifest" "$fail_log"' EXIT

export FAIL_LOG="$fail_log"
if ! tr '\n' '\0' < "$task_manifest" | \
     xargs -0 -P "$parallel_per_task" -I {} \
       bash -c 'download_one "$1" "$2" || echo "$1" >> "$FAIL_LOG"' _ {} "$output_dir"; then
  echo "xargs reported an error" >&2
fi

if [ -s "$fail_log" ]; then
  echo "Task ${task_id} finished with $(wc -l < "$fail_log") failed download(s):" >&2
  cat "$fail_log" >&2
  exit 1
fi

echo "Task ${task_id} finished successfully: ${num_files} file(s)."
