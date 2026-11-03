# Manuscript experiments

The reusable CLI is `okkmer`; these workflows and configurations describe the
paper-specific datasets and analyses. Existing scripts, notebooks, source data,
and results are preserved in place rather than moved during package extraction.
The manuscript directory is never modified by these commands.

## Existing experiment inventory

| Experiment / manuscript artifact | Preserved implementation or outputs |
|---|---|
| One/two/three-length recommendations, Table 2 | `bin/grid_search_2k.py`, `bin/grid_search_3k.py`, `data/*/grid_search/` |
| Genome masking, Fig 4 / Table 3 / Fig A1 | `bin/stability/`, `data/stability_3k/` |
| Two added datasets and 21/31-mer comparisons | `bin/stability/add_datasets.py`, `results/new_datasets_20261028/` |
| Annotated subtrees | `src/ok_k_mer_tree_mask.py`, `src/res/` |
| Distance preprocessing and runtime measurements | `data/*/genomes_timing_test/scripts/` and its recorded provenance |
| Near-optimal distinct three-length ensembles | `results/{cyano,chlam}_cluster_rho/` |
| Exploratory analyses and original figures | `src/explore_card.ipynb`, `src/ok_k_mer*.py` |

`data/` is a symlink to external storage; raw data and existing results are not
packaged with the CLI. Large artifacts remain where their recorded paths point.

## Run the new implementation on a manuscript dataset

`configs/datasets.json` records exact input semantics, tree paths, exclusions,
and approved assembly aliases for all eight datasets. Paths are relative to the
repository root. The wrapper invokes the CLI as a subprocess and saves the
config and commands alongside results. It has no Slurm dependency and does not
submit jobs. Run it inside an allocated job for expensive searches:

```bash
python experiments/manuscript/workflows/run_dataset.py \
    --dataset Chlamydiales --output results/cli_chlamydiales --threads 8
```

Defaults: global search, inclusive k=6–40, max 3 lengths, 0.05 weights. Use
`--local 3` for the published progressive local search. Add `--dry-run` to
prepare the index and validate/count the search without running optimization.
Output paths must be new or empty. New outputs are kept separate from legacy
results. These new runs are methodological validations, not automatic
replacements for manuscript values. See `docs/compatibility.md`.

## Validate legacy reference scores

With experiment dependencies including pandas available:

```bash
PYTHONPATH=src python experiments/manuscript/workflows/validate_legacy.py \
    --output results/cli_validation.json
```

This checks the best single length and scores the previously reported two- and
three-length recommendations for Chlamydiales and Cyanobacteriales using the
new tie-correct objective. It records differences; it does not repeat exhaustive
searches or claim that historical restricted searches were global.

The tests in `tests/` independently compare small exhaustive searches against
SciPy. The original regression checks remain under `bin/stability/`.
