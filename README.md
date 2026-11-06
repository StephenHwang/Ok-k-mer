# Ok-*k*-mer: Selecting *k*-mer length ensembles for calculating genomic distance

Ok-*k*-mer selects weighted ensembles of *k*-mer lengths for improving pairwise genomic
distance comparisons across a phylogenetic tree.
The executable and Python package are named `okkmer`.

## Installation

Python 3.10 or newer:

```bash
python -m pip install .
okkmer help
```

[Dashing2 **v2.1.21**](https://github.com/dnbaker/dashing2/releases/tag/v2.1.21) is a separate executable dependency for computing the *k*-mer
Jaccards distanced required by Ok-*k*-mer. 
Importing existing distances does not require Dashing2.

For development use `python -m pip install -e .`.

## Building the index
Ok-*k*-mer performs it's *k*-mer ensemble search over pairwise genomic distances 
from genomes of a phylogenetic tree.
To specify the input genome, a genome manifest is needed.
A genome manifest is a TSV with `genome_id` and `path` columns. 
IDs must match the input reference tree genomes.
Relative paths resolve against the manifest's folder.

```text
genome_id	path
genome_A	genomes/A.fna
genome_B	genomes/B.fna
genome_C	genomes/C.fna
```
By default, Ok-*k*-mer uses Dashing2 to compute the starting pairwise genomic distances.
```bash
okkmer index --genomes genomes.tsv --k-low 6 --k-high 40 \
    --threads 8 --output my_index
```

This runs Dashing2 with `dist --set --long-kmers --square` for each *k*
and records versions, commands, input hashes, and genome IDs. 
Raw similarity matrices and stderr logs are retained.

Alternatively, Ok-*k*-mer can use a TSV/CSV containing `genome_1`, `genome_2`, and `k6`,
`k7`, ... columns:

```bash
okkmer index --distances distances.tsv --input-kind distance \
    --k-low 6 --k-high 40 --output my_index
```

Every unordered genome pair must appear exactly once. Values must be finite and in [0,1]. 
Use `--input-kind similarity` for Jaccard similarities, including
legacy CSVs whose `*_distance` columns contain similarities. 
Jaccard similarity is complementary to Jaccard distance: `distance = 1 - similarity`.

For Dashing2 square TSVs, pass their directory to `--distances`. 
Distance files must be named `jaccard_k*.tsv`.
Similarity files must be named `jaccard_similarity_k*.tsv`. 

The `#Sources` header and row labels must agree.
An optional `--genome-map mapping.tsv` maps exact matrix labels to tree IDs
with `source_id` and `genome_id` columns. No accession prefix/version changes
are inferred by the tool.

## Ok-*k*-mer search
Ok-*k*-mer searches for ensembles of *k*-mers and weights that maximize agreeement of 
pariwise distance relationships to a phylogenetic tree.
Ok-*k*-mer search has two modes, `--global` and `--local`. These are mutually exclusive.
Global search is the default.

*k* bounds are inclusive. Every requested *k* must
exist in the index. Add `--dry-run` to validate inputs and report candidate
counts before computation; no output directory is created.

```bash
okkmer search --tree reference.nwk --index my_index --global \
    --max-kmers 3 --k-low 6 --k-high 40 --w-step 0.05 \
    --threads 8 --output search_results
```

Local search varies **each** previously selected length within ±DELTA_K,
while the additional length spans the complete requested range:

```bash
okkmer search --tree reference.nwk --index my_index --local 3 \
    -n 3 --output local_results
```

Without an input integer, `--local` defaults to use +-3 *k*-mer. 
`--global` and `--local` are mutually exclusive.
`-n` is the maximum number of *k*-mers checked in the *k*-mer ensembles.

Ok-*k*-mer's search stops when another length fails to improve rho by more than 
some tolerance, `--improvement-tol` (default 1e-12).
Use `--no-early-stop` to evaluate every ensemble size through `--max-kmers` even 
when an intermediate stage fails to improve rho. 
The final recommendation still retains the best ensemble found.

For an ensemble of *m* *k*-mers, each *k*-mer will have a distinct length *k* and a positive weight. 
The weights sum to one.
Zero-weight boundaries are represented by earlier search stages. 

The objective is Spearman correlation between reference branch distances and
the weighted sum of per-*k* Jaccard-distance ranks. 
Both ranking steps use average ranks for ties. 
Integer coefficients preserve exact weighted-score ties.
Correlations are signed; a larger correlation is better. 
Ties choose lexicographically smaller *k* combinations and then weight coefficients.

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
are omitted. Local outputs cover only visited combinations. 
Searches may stop early, so stages after the first non-improving stage are not evaluated. 
Full weight-specific scores can be saved with `--save-grid`.
Outputs must be new or empty directories; existing results are protected.

## Ok-*k*-mer plotting
Ok-*k*-mer provides two plotting functions: `heatmap` and `tree`.

Heatmaps show the best rho over weights. Three-length plots require a fixed-*k*
slice. Gray entries are unsearched, undefined, or repeated lengths.
```bash
okkmer plot heatmap --results search_results --ensemble-size 1 --output single.pdf
okkmer plot heatmap --results search_results --ensemble-size 2 --output pairs.pdf
okkmer plot heatmap --results search_results --ensemble-size 3 --fixed-k 20 --output triples.pdf
```

To annotate internal tree nodes, first compute their recommendations:
```bash
okkmer search --tree reference.nwk --index my_index -n 3 \
    --annotate-nodes --output annotated_results
okkmer plot tree --results annotated_results --output annotated_tree.pdf
```

This repeats optimization per internal node. Nodes with fewer than three genomes 
or constant distances are marked unscorable.
`nodes.json`, `nodes.tsv`, and per-node combination tables are saved. Search
mode `--local` refers to *k* neighborhoods, independently of subtree annotation.

## Citing Ok-*k*-mer and reproducing results
Preprint: **in-progress**

Scripts, experiements, notebooks to reproduce the results are found in a separate repo:
[`ok_k_mer_experiments`](https://github.com/StephenHwang/Ok-k-mer_Experiments).


## Getting help
If you run into any issues or have any questions, reach out either through:

1. GitHub Issues
2. shwang45 [at] jhu.edu


## License
Ok-*k*-mer is licensed under the GNU General Public License version 3.  
See [LICENSE](LICENSE) for the full terms.
