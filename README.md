# Ok-k-mer

Ok-k-mer selects weighted ensembles of k-mer lengths whose pairwise genomic
distance ranks best agree with distances in a reference phylogeny.
The executable and Python package are named `okkmer`.

## Install

Python 3.10 or newer:

```bash
python -m pip install '.[plot]'
okkmer help
```

For development use `python -m pip install -e '.[plot]'`.
Dashing2 **v2.1.21** (or the manuscript benchmark revision
`72d16c72fdb43f37cff06f8a13c94b7fdc9d29af`) is a separate executable dependency only when computing
an index from genomes. Importing existing distances does not require Dashing2.
The executable must be runnable in your environment; the benchmark build on
Rockfish requires `module load GCC/11.3.0`.
The plotting extra uses Matplotlib and does not require a graphical desktop.

## Prepare an index

A genome manifest is a TSV with `genome_id` and `path` columns. IDs must match
Newick leaf labels exactly. Relative paths resolve against the manifest's folder.

```text
genome_id	path
genome_A	genomes/A.fna
genome_B	genomes/B.fna
genome_C	genomes/C.fna
```

```bash
okkmer index --genomes genomes.tsv --k-low 6 --k-high 40 \
    --threads 8 --output my_index
```

This invokes Dashing2 with `dist --set --long-kmers --square` for each k,
normalizes similarity to Jaccard distance, and records versions, commands,
input hashes, and genome IDs. Successful invocations remove their own private
intermediate sets. Raw similarity matrices and stderr logs are retained.

Alternatively, import a TSV/CSV containing `genome_1`, `genome_2`, and `k6`,
`k7`, ... columns (the legacy names `k6_distance`, etc. are also accepted):

```bash
okkmer index --distances distances.tsv --input-kind distance \
    --k-low 6 --k-high 40 --output my_index
```

Every unordered genome pair must appear exactly once. Values must be finite
and in [0,1]. Use `--input-kind similarity` for Jaccard similarities, including
legacy experiment CSVs whose `*_distance` columns actually contain similarities.
Other metadata columns are ignored. Distances from metrics other than Jaccard
are not supported by this first index schema.

For Dashing2 square TSVs, pass their directory to `--distances`. Distance
files must be named `jaccard_k6.tsv`, etc.; similarity files must be named
`jaccard_similarity_k6.tsv`, etc. The `#Sources` header and row labels must agree.
An optional `--genome-map mapping.tsv` maps exact matrix labels to tree IDs
with `source_id` and `genome_id` columns. No accession prefix/version changes
are inferred by the tool.

## Search

```bash
okkmer search --tree reference.nwk --index my_index --global \
    --max-kmers 3 --k-low 6 --k-high 40 --w-step 0.05 \
    --threads 8 --output search_results
```

Global search is the default. K bounds are inclusive. Every requested k must
exist in the index. Add `--dry-run` to validate inputs and report candidate
counts before computation; no output directory is created.

Local search varies **each** previously selected length within ±DELTA_K,
while the additional length spans the complete requested range:

```bash
okkmer search --tree reference.nwk --index my_index --local 3 \
    -n 3 --output local_results
```

`--local` without a number uses 3. `--global` and `--local` are mutually exclusive.
`-n` is the maximum number of distinct lengths, not a guarantee of that many.
Search stops when another length fails to improve rho by more than
`--improvement-tol` (default 1e-12).

Exactly m-length candidates have m distinct k values, all with positive weights
on the specified grid summing to one. Zero-weight boundaries are represented
by earlier search stages. The weight step must divide 1 exactly; m cannot
exceed the number of requested lengths or 1/w-step.

The objective is Spearman correlation between reference branch distances and
the weighted sum of per-k Jaccard-distance ranks. Both ranking steps use average
ranks for ties. Integer coefficients preserve exact weighted-score ties.
Correlations are signed; larger is better. Ties choose lexicographically smaller
k combinations and then weight coefficients. Constant genomic score vectors
are skipped; constant reference distances are rejected.

Genomes must match the tree exactly. For deliberate omissions provide
`--exclude-genomes excluded.txt`, with one exact ID per line. Exclusions affect
comparisons and subtree membership; source trees are never edited.

Outputs:

- `run.json`: recommendation, settings, search scope, versions, provenance,
  timing, code hashes, counts, and stopping reason.
- `recommendations.tsv`: best ensemble at each evaluated size; `selected=True`
  identifies the final recommendation. The last stage may not improve it.
- `single_k.tsv`: single-length scores.
- `combinations_2.tsv`, `combinations_3.tsv`, ...: one row per evaluated distinct
  combination, including its best weights, rho, and gap from that stage's best.
- `reference.nwk`: a bundled copy for plotting.

The `ks` and `weights` table cells contain JSON arrays. Undefined combinations
are omitted. Local outputs cover only visited combinations. Search stops early,
so stages after the first non-improving stage are not evaluated. Full
weight-specific scores can be saved with `--save-grid` (potentially large).
Outputs must be new or empty directories; existing results are protected.

## Plot

```bash
okkmer plot heatmap --results search_results --ensemble-size 1 --output single.pdf
okkmer plot heatmap --results search_results --ensemble-size 2 --output pairs.pdf
okkmer plot heatmap --results search_results --ensemble-size 3 --fixed-k 20 --output triples.pdf
```

Heatmaps show the best rho over weights. Three-length plots require a fixed-k
slice. Gray entries are unsearched, undefined, or repeated lengths.

To annotate internal tree nodes, first compute their recommendations:

```bash
okkmer search --tree reference.nwk --index my_index -n 3 \
    --annotate-nodes --output annotated_results
okkmer plot tree --results annotated_results --output annotated_tree.pdf
```

This repeats optimization per internal node and may take much longer. Nodes
with fewer than three genomes or constant distances are marked unscorable.
`nodes.json`, `nodes.tsv`, and per-node combination tables are saved. Search
mode `--local` refers to k neighborhoods, independently of subtree annotation.

## Manuscript experiments and development

Reusable code lives in `src/okkmer/`, with package tests under `tests/`.
Manuscript workflows, legacy analysis code, notebooks, dataset configurations,
and results live in the separate **`ok_k_mer_experiments`** repository, located
alongside this checkout at `../ok_k_mer_experiments`. Its README documents the
experiment entry points and data locations. See
[docs/compatibility.md](docs/compatibility.md) before comparing old and new scores.

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
