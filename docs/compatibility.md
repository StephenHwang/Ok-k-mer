# Scientific compatibility and migration

The new package implements the method described in manuscript Section 4.5.
Legacy experiment scripts remain in their original locations to preserve paths,
input hashes, and provenance. They do not import the new package implicitly.

## Objective and ties

Legacy merged CSVs often have `k*_distance` columns holding similarities. Their
optimizers minimize rho; manuscript tables reverse its sign. Import those CSVs
with `--input-kind similarity`. The new package stores actual distances and
maximizes positive signed rho. It does not maximize absolute correlation.

Legacy three-k scripts use floating-point weighted sums. Some accelerated
subtree routines assign distinct ranks to ties. The new implementation uses
average ranks at both steps, with integer coefficients applied to doubled
average ranks, making exact composite ties independent of arithmetic rounding.
Small score/weight differences are possible at tied candidates and must be
reported, rather than hidden by adjusting signs or rounding results.

## Global versus local

`bin/grid_search_3k.py` limits its first k to a neighborhood of the best single
length; the other two span the full range. This is a historical restricted
search, not the new progressive local search. Its `--s 34` over k=6–40 covers
the complete range, but still evaluates repeated lengths, permutations, and
zero-weight boundaries. The new global mode enumerates unordered distinct
lengths with positive weights, retaining lower-order results separately.

The manuscript local mode perturbs every length of the preceding winning
ensemble independently before adding another length from the full range.

## Early stopping

The public CLI stops when a new stage fails to improve the preceding winner.
A near-optimal-ensemble experiment requiring a particular ensemble size even
when an earlier stage fails should use its preserved dedicated workflow. A
missing combinations file means the stage was not reached, not zero possible
combinations. The tool records the stopping reason and search mode.

## Data contracts

Tree leaves and index genome IDs match exactly unless an explicit exclusion
file is provided. Prefix/version aliases belong in experiment configurations,
not in the package's biological identity rules. The Dethiobacteria experiment
excludes two unavailable genomes; the Anaerolineales experiment records its
user-confirmed assembly aliases.

Index schema 1 stores Jaccard distances for complete unordered pair coverage.
It is independent of any tree. The source files and installed package are not
modified by searches. Plotting reads saved results and performs no optimization.
