# Genome-masking stability simulations

These new entry points call the functions in the original `bin/grid_search_2k.py`
and `bin/grid_search_3k.py`. The originals and `src/explore_card.ipynb` are unchanged.
Run the commands below from `/scratch4/blangme2/sjhwang/ok_k_mer`.

## Test: one 50% mask per dataset

The prepared test manifest contains six tasks, one per dataset at 50% masking.
It uses the full three-k search settings (k=6–40, weight step 0.05, s=3), so each
task is representative of the full experiment and may take substantial time.
Seeds match replicate 0 at 50% masking in the main experiment.

```bash
/home/shwang45/miniconda3/envs/python310/bin/python bin/stability/stability.py submit \
  --manifest /vast/blangme2/sjhwang/ok_k_mer_data/stability_3k/test_run/manifest.json \
  --max-concurrent 6
```

Results, logs and submission records go under the separate `test_run` directory.
This submits only six jobs, each requesting 8 CPUs, 32 GB and up to 36 hours.
Collect the test results after completion:

```bash
/home/shwang45/miniconda3/envs/python310/bin/python bin/stability/stability.py collect \
  --manifest /vast/blangme2/sjhwang/ok_k_mer_data/stability_3k/test_run/manifest.json
```

There is no unmasked baseline in this test, so baseline-relative metrics remain
empty. Recommendations, correlations, seeds, runtime and Slurm IDs are recorded.
To prepare another such test in a new or empty directory:

```bash
/home/shwang45/miniconda3/envs/python310/bin/python bin/stability/stability.py prepare \
  --run-dir /vast/blangme2/sjhwang/ok_k_mer_data/stability_3k/test_run_new \
  --max-k 3 --masking-fractions 0.5 --repeats 1 --master-seed 0
```

## Submit the six-dataset experiment

The prepared `/vast/blangme2/sjhwang/ok_k_mer_data/stability_3k/manifest.json` contains **636 tasks**: six
datasets × (one unmasked baseline + seven masking fractions × 15 replicates).
Each array task runs exactly one dataset/fraction/replicate and has its own result
and log files. Replicates are numbered 0–14; the baseline is replicate 0.

All run artifacts (results, optional tensors, logs, submission records, collected
CSVs and plots) are stored under `/vast/blangme2/sjhwang/ok_k_mer_data/stability_3k`.
The old `results/stability_3k` path is a symlink to that directory, so earlier
commands also use `/vast` storage.

  ```bash
/home/shwang45/miniconda3/envs/python310/bin/python bin/stability/stability.py submit \
  --manifest /vast/blangme2/sjhwang/ok_k_mer_data/stability_3k/manifest.json --max-concurrent 24
```

This command actually calls `sbatch --parsable`. It saves the command, raw Slurm
response, array job ID, optional cluster name and submission time in
`/vast/blangme2/sjhwang/ok_k_mer_data/stability_3k/submissions/<timestamp>.json`. The matching CSV has one row
per task, including `slurm_task_id` (`ARRAY_JOB_ID_TASK_ID`), dataset, fraction,
percentage, simulation, actual masking seed, master seed, search settings and
output path. The manifest and `tasks.csv` exist before submission.

Default three-k resources are 8 CPUs and 32 GB per task, with a 36-hour limit,
on `shared` under `blangme2`, matching the partition's
[maximum runtime](https://docs.arch.jhu.edu/en/latest/1_Clusters/Rockfish/3_Slurm/Partitions.html#shared).
The default concurrency is 24 tasks, requesting 192 CPUs and 768 GB in total.
The live `shared` partition QoS checked on 2026-09-22 limits each account to
320 CPUs across its shared jobs (40 of these 8-CPU tasks if otherwise idle).
Other jobs under `blangme2` share that cap; resource availability and priority
can reduce actual concurrency. Resource needs and runtime are dataset dependent; the original
three-k exhaustive loops can be expensive. Override resources at submission:

```bash
/home/shwang45/miniconda3/envs/python310/bin/python bin/stability/stability.py submit \
  --manifest /vast/blangme2/sjhwang/ok_k_mer_data/stability_3k/manifest.json --max-concurrent 3 \
  --cpus-per-task 8 --mem 32G --time 36:00:00
```

Use either submission command, once. After cancelling an earlier array and
verifying it has stopped, resubmit unfinished tasks at the higher concurrency:

```bash
/home/shwang45/miniconda3/envs/python310/bin/python bin/stability/stability.py submit \
  --manifest /vast/blangme2/sjhwang/ok_k_mer_data/stability_3k/manifest.json \
  --resume-missing --max-concurrent 24
```

This creates a new array and submission receipt, skips existing result JSONs,
and preserves task IDs and seeds. `--resume-missing` checks output files, not
live scheduler state: use it only after earlier jobs have stopped.

To explicitly submit a subset or retry
failed tasks, use `--task-ids 4,5,6`; this creates another submission receipt.
Check that those tasks are no longer running before retrying. Existing completed
results are protected against overwriting. Move any incomplete `.npy` or
`.json.partial` artifacts aside before retrying a task that produced them.

The Slurm wrappers default to `$HOME/miniconda3/envs/python310/bin/python`.
Override with `export STABILITY_PYTHON=/path/to/python` before submitting.
`STABILITY_BIN` can override the scripts' absolute directory if the project moves;
the submission helper sets it automatically. The scripts do not depend on Slurm's
spooled script location or the submission working directory.

## Collect results and plots

After jobs complete, or while they are running:

```bash
/home/shwang45/miniconda3/envs/python310/bin/python bin/stability/stability.py collect \
  --manifest /vast/blangme2/sjhwang/ok_k_mer_data/stability_3k/manifest.json --plots
```

Outputs in the run directory:

- `stability_simulations.csv`: one row per completed task, all three k/weight
  columns, correlation, effective k, seed and actual Slurm IDs, counts, timing,
  and deviations/retention relative to that dataset's unmasked baseline.
- `stability_summary.csv`: counts, means, standard deviations, effective-k match
  rate (within ±1 by default; change with `--tol`), and correlation retention.
- `effective_k_vs_masking.{pdf,png}` and `spearman_vs_masking.{pdf,png}`:
  per-dataset means with ±1 sample standard deviation. The single baseline is
  shown without spread.
- `status.csv`: every planned task and whether a completed result file exists,
  plus its most recently recorded submission ID. “Missing” may mean pending,
  running or failed; use Slurm for scheduler state.

For example, substitute the recorded array ID:

```bash
squeue -r -j ARRAY_JOB_ID
sacct -j ARRAY_JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Collection accepts partial completion, prints the completed/expected count, and
leaves baseline-relative fields empty until the baseline is available. It can
be rerun to refresh CSVs and plots. Per-task JSONs also contain the exact masked
genome IDs, input/code SHA-256 hashes, package versions and raw optimizer output.
The manuscript directory is not modified.

## Prepare a different experiment

```bash
/home/shwang45/miniconda3/envs/python310/bin/python bin/stability/stability.py prepare \
  --run-dir /vast/blangme2/sjhwang/ok_k_mer_data/stability_3k_new --max-k 3 --master-seed 0

/home/shwang45/miniconda3/envs/python310/bin/python bin/stability/stability.py prepare \
  --run-dir results/stability_2k --max-k 2 --master-seed 0
```

`prepare` only creates files; `submit` schedules the work. Run directories must
be new or empty. Defaults are k=6–40, weight step 0.05, `s=3`, and 15 nonzero replicates.
Set `--data-root` for a different data location. The manifest uses the six cached
`grid_search/*_merged_df.csv` files used in the notebook. Use `--save-tensors` at
preparation to additionally save the full `.npy` tensor for every task; these
are omitted by default to limit storage.

Seeds are deterministic SHA-256-derived integers from master seed, dataset,
masking percentage and simulation number. Two-k and three-k experiments with
the same master seed therefore use the same masks. These are **new replicates**,
not the notebook's sequential seed stream. Each mask follows the notebook:
`floor(fraction * number_of_tree_leaves)` leaves sampled without replacement by
NumPy `default_rng`, in the original tree leaf order, and every pair involving
a masked leaf removed before ranking.

The original search minimizes Spearman rho between weighted k-mer ranks and
tree distances. Three-k search recomputes the best single k after masking and
restricts k1 to ±`s` around it; k2 and k3 span the full range. Its zero-weight
boundaries include the one/two-k candidates. For an unrestricted first-k range,
set `--s 34` when preparing a k=6–40 experiment, at a substantial runtime cost.
The three-k weight step here is 0.05 to match the notebook, while the original
three-k CLI defaults to 0.1. Change with `--w-step 0.1` if desired.

Recommendations combine repeated k values, drop zero-weight terms, and sort by
k; `n_ks` counts the remaining terms. `raw_ks` and `raw_weights` preserve the
original search's output. Ties retain the original grid's traversal behavior.

## Submit one simulation directly

Both the Python scripts and Slurm wrappers accept the masking fraction and seed:

```bash
sbatch --output=/vast/blangme2/sjhwang/ok_k_mer_data/stability_3k/logs/single_%j.out \
  --error=/vast/blangme2/sjhwang/ok_k_mer_data/stability_3k/logs/single_%j.err \
  bin/stability/grid_search_3k_slurm.sh \
  --tree-path data/caldisericia/c__Caldisericia.nwk \
  --merged-csv data/caldisericia/grid_search/caldisericia_merged_df.csv \
  --dataset Caldisericia --masking-fraction 0.3 --simulation 0 --seed 123 \
  --json-out /vast/blangme2/sjhwang/ok_k_mer_data/stability_3k/results/single_caldisericia_3k.json
```

Use `grid_search_2k_slurm.sh` for two-k. For original raw input, replace
`--merged-csv ...` with `--dir-path data/caldisericia/dashingfull_genomes_ncbi`.
Individual jobs record their Slurm job ID in the output JSON; the array helper
also captures submission-time IDs before computation finishes.

Regression checks (small synthetic grids, mocked submission; no jobs scheduled):

```bash
/home/shwang45/miniconda3/envs/python310/bin/python -m unittest discover \
  -s bin/stability -p 'test_stability.py' -v
```

## Anaerolineales and Dethiobacteria (October 2026)

`add_datasets.py prepare` validates all 35 square Jaccard distance matrices
against the tree accessions and writes inputs under
`results/new_datasets_20261028`. It negates distances so the existing
similarity-oriented minimization and sign-flipped reporting remain equivalent
to maximizing distance concordance, without subtractive rounding.
The manifest copies the original experiment's fractions and search settings.
The `2k` manifest has only the two unmasked baselines; `3k` has all replicates.
Submit each with `stability.py submit --manifest PATH --max-concurrent 24`.

After both arrays finish, `add_datasets.py collect` produces Table 2/A1 data in
`table2_and_A1.csv`, first-three-trial rows for all fractions in
`table3_all_fractions.csv`, and Fig 4/A1 plots under `3k/` (Spearman and effective
weighted k, respectively). Raw correlations in simulation CSVs retain the
original negative convention; comparison-table rhos and plots are positive.
The manuscript is never modified. `collect_new_datasets.sh` can be submitted
with an `afterok` dependency on both arrays for automatic export.

The confirmed Anaerolineales mappings are recorded alongside its merged CSV.
Dethiobacteria explicitly excludes `GCF_000174415.1` and `GCF_022558445.1`
from both comparisons and the masking population (102 genomes). Its original
104-leaf tree is read unchanged; retained distances are calculated on that tree.
Unexpected missing genomes still fail validation. The collector also combines
original and new simulations into eight-dataset Fig 4/A1 PDFs and PNGs (0–60%,
matching the manuscript), plus first-three-trial CSVs for all fractions and 30%.
The full simulation run retains the original 0–70% experimental range.
