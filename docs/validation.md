# Initial package validation

Validated in the repository's Python 3.10 environment on 2026-11-03.

- 12 package tests passed: exact weight grids, average-rank ties, independent
  exhaustive SciPy reference comparisons, progressive local candidate ranges,
  early stopping, index semantics/coverage/checksums, explicit exclusions,
  raw weight export, subprocess preprocessing, and plot generation.
- All four preserved `bin/stability/` regression tests passed. Their Slurm
  submissions are mocked; those tests do not schedule cluster jobs.
- A wheel built and installed into `/tmp/okkmer-install-check` without network
  access. An editable installation was also created in the checkout's `.venv`
  using the available Python 3.10 environment's scientific dependencies.
- The real pinned Dashing2 binary computed two square matrices (k=6–7) on
  three synthetic genomes; the CLI validated and imported them successfully.
  This used the benchmark revision and `module load GCC/11.3.0`.
- The Chlamydiales, Anaerolineales, and Dethiobacteria manuscript wrappers
  prepared complete k=6–40 indices and passed `search --dry-run` validation.
  Anaerolineales retained all 525 genomes using the confirmed explicit aliases;
  Dethiobacteria retained 102 after the two explicit tree-only exclusions.
- The installed CLI searched Chlamydiales k=9–18 with two lengths. It selected
  (9,18), weights (0.45,0.55), rho approximately 0.5132143559.

## Reference recommendation scores

The validation workflow scored previously reported recommendations against the
cached reference distances. It did not rerun the full manuscript experiment.
Differences below are new minus legacy rho:

| Dataset | Ensemble | Difference |
|---|---|---:|
| Chlamydiales | single | 5.55e-17 |
| Chlamydiales | two | -3.64e-8 |
| Chlamydiales | three | -4.02e-8 |
| Cyanobacteriales | single | -2.22e-16 |
| Cyanobacteriales | two | -3.52e-11 |
| Cyanobacteriales | three | -5.72e-12 |

The report is `results/cli_validation.json` in this checkout. Reproduce it with
`experiments/manuscript/workflows/validate_legacy.py` and a new output path.
These small differences reflect the explicitly changed tie/weight arithmetic
and source preprocessing; source trees can also differ slightly in floating
point summation from cached tree-distance columns. See `compatibility.md`.

Full eight-dataset global searches, masking simulations, and large-tree plot
styling have not been rerun as part of package extraction. Existing manuscript
results remain the source of record for the historical experiments.
