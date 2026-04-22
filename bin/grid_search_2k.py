#!/usr/bin/env python3
"""
Grid search over 2 k-mer scales vs tree distance (Spearman on weighted ranks).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from ete3 import Tree
from scipy.stats import spearmanr


def dist_tree_calc(filename1):
    """Calculate tree distance."""
    tree1 = Tree(filename1, format=1)
    dist_tree = {}
    samples = [i.name for i in tree1.get_leaves()]
    num_sample = len(samples)
    for i in range(num_sample):
        tax_i = samples[i]
        for j in range(i + 1, num_sample):
            tax_j = samples[j]
            dist_tree[(min(tax_i, tax_j) + ".fa", max(tax_i, tax_j) + ".fa")] = tree1.get_distance(
                tax_i, tax_j, topology_only=False
            )
    return dist_tree


def make_tree_dict(tree_path):
    """
    Given a path to a Newick tree file, return a symmetric dictionary of distances between all pairs of taxa names
    (with or without '.fa' extension).

    Parameters:
        tree_path (str): Path to the Newick tree file.

    Returns:
        dict: A dictionary of dictionaries: { taxon_name: { other_taxon: distance, ... }, ... }
    """
    dist_tree = dist_tree_calc(tree_path)

    names = set()
    processed_pairs = []
    for (g1, g2), dist in dist_tree.items():
        g1_clean = g1.rstrip(".fa")
        g2_clean = g2.rstrip(".fa")
        names.add(g1_clean)
        names.add(g2_clean)
        processed_pairs.append((g1_clean, g2_clean, float(dist)))

    tree_dict = {name: {other: 0.0 for other in names} for name in names}
    for g1, g2, d in processed_pairs:
        tree_dict[g1][g2] = d
        tree_dict[g2][g1] = d  # Ensure symmetry

    return tree_dict


def parse_dashing_dfs(dir_path):
    """
    Parse all distance_* files in the given directory into a dictionary of pandas dataframes.
    Each dataframe skips the first line, assumes tab-separation and adjusts indices and columns as expected.

    Args:
        dir_path (str): Path to the directory containing distance_* files.

    Returns:
        dict: Dictionary mapping filename to corresponding dataframe.
    """
    files = [
        f
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f)) and f.startswith("distance_")
    ]

    dashing_dfs = {}

    for file in files:
        file_path = os.path.join(dir_path, file)
        try:
            df = pd.read_csv(file_path, sep="\t", header=None, skiprows=1)
            df.index = df.iloc[:, 0].str.replace("genomes_ncbi/", "")
            df = df.iloc[:, 1:]
            df.columns = df.index
            dashing_dfs[file] = df
        except Exception as e:
            print(f"error reading {file}: {str(e)}", file=sys.stderr)

    print(f"\nloaded {len(dashing_dfs)} files")
    for file, df in dashing_dfs.items():
        print(f"{file}: {df.shape}")
    return dashing_dfs


def build_merged_df(tree_dict, dashing_dfs):
    """
    Build merged dataframe for all genome pairs among `genomes`,
    including tree distances and distances for each k from `dashing_dfs`.
    """
    genomes = sorted(tree_dict.keys())

    pairs = []
    for i in range(len(genomes)):
        for j in range(i + 1, len(genomes)):
            pair = "_".join(sorted([genomes[i], genomes[j]]))
            pairs.append(pair)

    merged_df = pd.DataFrame(index=pairs)

    genome_pairs = []
    genomes_1s = []
    genomes_2s = []
    tree_distances = []
    for index in merged_df.index:
        genome1 = "_".join(index.split("_", 2)[:2])
        genome2 = "_".join(index.split("_", 2)[2:])

        genome_pairs.append(index)
        genomes_1s.append(genome1)
        genomes_2s.append(genome2)
        tree_distances.append(tree_dict[genome1][genome2])

    merged_df["genome_pair"] = genome_pairs
    merged_df["genome_1"] = genomes_1s
    merged_df["genome_2"] = genomes_2s
    merged_df["tree_distance"] = tree_distances

    print(f"Final dataframe shape: {merged_df.shape}")

    for fname, df in dashing_dfs.items():
        k_val = int(str(fname).replace("distance_k", "").replace(".txt", ""))
        pair_to_dist = {}
        for i in range(len(df)):
            g1 = df.iloc[i].name
            for g2, d in dict(df.iloc[i]).items():
                if g1 == g2:
                    continue
                pair = "_".join(sorted([g1, g2]))
                pair_to_dist[pair] = d

        colname = f"k{k_val}_distance"
        merged_df[colname] = merged_df.index.map(pair_to_dist.get)
        print(f"Added column '{colname}' to merged_df.")

    return merged_df


def process_data(tree_path, dashing_dir_path):
    tree_dict = make_tree_dict(tree_path)
    dashing_dfs = parse_dashing_dfs(dashing_dir_path)
    merged_df = build_merged_df(tree_dict, dashing_dfs)
    return tree_dict, dashing_dfs, merged_df


################################################################################
# Full Grid search: unfixed k-mers; with and without weights
def run_full_grid_corr(merged_df, k_low=5, k_high=32, w_low=0, w_high=1, w_step=0.05):
    """2-k full grid search over (k1, k2, w) for weighted-rank Spearman."""
    ws = np.round(np.arange(w_low, w_high + w_step, w_step), 3)
    num_ws = len(ws)
    ks = list(range(k_low, k_high + 1))
    num_ks = len(ks)

    results_matrix = np.zeros((num_ks, num_ks, num_ws))
    for k1_idx, k1 in enumerate(ks):
        k1_rank = merged_df[f'k{k1}_distance'].rank()
        for k2_idx, k2 in enumerate(ks):
            k2_rank = merged_df[f'k{k2}_distance'].rank()
            for w_idx, w in enumerate(ws):
                avg_k1_k2_w_rank = w * k1_rank + (1 - w) * k2_rank   # weight the ranks by w and 1-w
                corr, _pval = spearmanr(avg_k1_k2_w_rank, merged_df['tree_distance'])
                results_matrix[k1_idx, k2_idx, w_idx] = corr

    # find global min
    results_matrix = np.nan_to_num(results_matrix, nan=np.inf)
    min_val = np.min(results_matrix)
    min_idx = np.unravel_index(np.argmin(results_matrix), results_matrix.shape)
    k1_min, k2_min, w_min = ks[min_idx[0]], ks[min_idx[1]], ws[min_idx[2]]
    print(f"Grid Ks: {k1_min, k2_min} with weight {w_min}; corr: {min_val}")
    print(min_idx)
    return results_matrix, min_val, k1_min, k2_min, w_min, min_val




################################################################################
def main():
    p = argparse.ArgumentParser(
        description="2-k weighted rank grid search vs tree distance (Spearman)."
    )
    p.add_argument(
        "--dir-path",
        required=True,
        help="Directory containing distance_k*.txt (dashing) files",
    )
    p.add_argument("--tree-path", required=True, help="Path to Newick tree (.nwk)")
    p.add_argument("--k-low", type=int, default=6, help="Minimum k for grid search (default 6)")
    p.add_argument("--k-high", type=int, default=40, help="Maximum k for grid search (default 40)")
    p.add_argument("--w-step", type=float, default=0.05, help="Weight grid step (default 0.1)")
    p.add_argument(
        "--save-npy",
        metavar="PATH",
        default=None,
        help="Optional path to save the full results tensor as .npy",
    )
    p.add_argument(
        "--json-out",
        metavar="PATH",
        default=None,
        help="Optional path to write summary JSON (best k, weights, min Spearman)",
    )
    args = p.parse_args()

    if not os.path.isdir(args.dir_path):
        print(f"Not a directory: {args.dir_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.tree_path):
        print(f"Tree file not found: {args.tree_path}", file=sys.stderr)
        sys.exit(1)

    _tree_dict, _dashing_dfs, merged_df = process_data(args.tree_path, args.dir_path)

    results_matrix, min_val, k1_min, k2_min, w_min, _min_dup = run_full_grid_corr(
        merged_df,
        k_low=args.k_low,
        k_high=args.k_high,
        w_low=0.0,
        w_high=1.0,
        w_step=args.w_step,
    )

    summary = {
        "k1": int(k1_min),
        "k2": int(k2_min),
        "w_k1": float(w_min),
        "w_k2": float(round(1.0 - w_min, 3)),
        "min_spearman_corr": float(min_val),
        "dir_path": os.path.abspath(args.dir_path),
        "tree_path": os.path.abspath(args.tree_path),
        "k_low": args.k_low,
        "k_high": args.k_high,
        "w_step": args.w_step,
    }

    print(json.dumps(summary, indent=2))

    if args.save_npy:
        np.save(args.save_npy, results_matrix)
        print(f"Saved results tensor to {args.save_npy}", file=sys.stderr)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote summary JSON to {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
