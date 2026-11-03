"""Mask whole genomes, then call the unmodified original grid-search functions."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy
from ete3 import Tree

ORIGINAL_BIN = Path(__file__).resolve().parent.parent


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mask_pairs(tree, df, fraction, seed, excluded_genomes=()):
    """Same leaf order, floor rounding, RNG and pair removal as mask_genomes."""
    leaves = tree.get_leaf_names()
    excluded = set(excluded_genomes)
    if not excluded <= set(leaves):
        raise ValueError("Excluded genomes are not in the tree")
    leaves = [name for name in leaves if name not in excluded]
    if len(set(leaves)) != len(leaves):
        raise ValueError("Tree contains duplicate leaf names")
    names = set(df.genome_1) | set(df.genome_2)
    if names != set(leaves):
        raise ValueError("Tree leaves and input genome IDs do not match exactly")
    masked = set(np.random.default_rng(seed).choice(
        np.asarray(leaves), size=math.floor(fraction * len(leaves)),
        replace=False).tolist())
    keep = ~(df.genome_1.isin(masked) | df.genome_2.isin(masked))
    sub = df.loc[keep].copy()
    if len(leaves) - len(masked) < 3 or len(sub) < 3:
        raise ValueError("Mask leaves fewer than three genomes/pairs")
    if sub.tree_distance.nunique() < 2:
        raise ValueError("Retained tree distances are constant")
    return sub, sorted(masked)


def canonical_recommendation(ks, weights):
    """Combine repeated k values and omit zero weights for up-to-N summaries."""
    combined = {}
    for k, weight in zip(ks, weights):
        if weight > 0:
            combined[int(k)] = combined.get(int(k), 0.0) + float(weight)
    pairs = sorted(combined.items())
    result = {"n_ks": len(pairs), "k_eff": sum(k * w for k, w in pairs)}
    for i in range(3):
        result[f"k{i + 1}"] = pairs[i][0] if i < len(pairs) else None
        result[f"w{i + 1}"] = pairs[i][1] if i < len(pairs) else 0.0
    return result


def main(max_k):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tree-path", required=True)
    p.add_argument("--exclude-genomes", nargs="*", default=[])
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--dir-path", help="Original distance_k*.txt input")
    source.add_argument("--merged-csv", help="Notebook's cached merged dataframe")
    p.add_argument("--masking-fraction", type=float, required=True)
    p.add_argument("--seed", type=int, required=True, help="Actual NumPy masking seed")
    p.add_argument("--simulation", type=int, default=0, help="Zero-based replicate")
    p.add_argument("--dataset", required=True)
    p.add_argument("--k-low", type=int, default=6)
    p.add_argument("--k-high", type=int, default=40)
    p.add_argument("--w-step", type=float, default=0.05)
    p.add_argument("--s", type=int, default=3, help="Original three-k k1 window radius")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p.add_argument("--json-out", type=Path, required=True)
    p.add_argument("--save-npy", type=Path, help="Optional full grid tensor")
    args = p.parse_args()
    if not 0 <= args.masking_fraction < 1:
        p.error("masking-fraction must be in [0, 1)")
    if args.seed < 0 or args.simulation < 0 or args.s < 0 or args.threads < 1:
        p.error("seed, simulation and s must be nonnegative; threads must be positive")
    if args.k_low < 1 or args.k_high < args.k_low:
        p.error("invalid k range")
    if not (0 < args.w_step <= 1) or not math.isclose(1 / args.w_step, round(1 / args.w_step)):
        p.error("w-step must divide 1 exactly, e.g. 0.05 or 0.1")
    if not math.isclose(args.w_step, round(args.w_step, 3)):
        p.error("original search supports weight steps with at most three decimal places")
    for out in [args.json_out, args.save_npy]:
        if out is not None:
            if out.exists():
                p.error(f"Refusing to overwrite {out}")
            out.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    sys.path.insert(0, str(ORIGINAL_BIN))
    original = importlib.import_module(f"grid_search_{max_k}k")
    tree = Tree(args.tree_path, format=1)
    if args.merged_csv:
        df = pd.read_csv(args.merged_csv)
    else:
        _, _, df = original.process_data(args.tree_path, args.dir_path)
    columns = ["tree_distance"] + [f"k{k}_distance" for k in range(args.k_low, args.k_high + 1)]
    if not np.isfinite(df[columns].to_numpy(dtype=float)).all():
        raise ValueError("Input contains missing/non-finite distances")
    sub, masked = mask_pairs(tree, df, args.masking_fraction, args.seed, args.exclude_genomes)
    n_genomes = len(tree) - len(set(args.exclude_genomes))
    print(f"{args.dataset}: masked {len(masked)}/{n_genomes} genomes; retained {len(sub)} pairs", flush=True)
    extra = {}
    if max_k == 2:
        tensor, rho, k1, k2, w1, _ = original.run_full_grid_corr(
            sub, k_low=args.k_low, k_high=args.k_high, w_step=args.w_step)
        ks, weights = [k1, k2], [float(w1), round(1 - float(w1), 3)]
    else:
        _, (k_main, single_corr, single_pvalue) = original.initial_corr(sub, args.k_low, args.k_high)
        tensor, rho, k1, k2, k3, w1, w2, w3, _ = original.run_full_3k_grid_corr_k_main_parallel(
            sub, k_main, s=args.s, k_low=args.k_low, k_high=args.k_high,
            w_step=args.w_step, num_threads=args.threads)
        ks, weights = [k1, k2, k3], [w1, w2, w3]
        extra = {"k_main": k_main, "initial_single_k_spearman": single_corr,
                 "initial_single_k_pvalue": single_pvalue}
    if not np.isfinite(rho):
        raise ValueError("Grid search found no finite correlation")
    sources = [Path(original.__file__), Path(__file__), Path(sys.argv[0]).resolve()]
    inputs = [Path(args.tree_path)]
    inputs += ([Path(args.merged_csv)] if args.merged_csv else
               [Path(args.dir_path) / f"distance_k{k}.txt" for k in range(args.k_low, args.k_high + 1)])
    summary = {
        "dataset": args.dataset, "max_k": max_k, "simulation": args.simulation,
        "masking_fraction": args.masking_fraction, "masking_percentage": args.masking_fraction * 100,
        "seed": args.seed, "masked_genomes": masked,
        "excluded_genomes": args.exclude_genomes, "n_tree_genomes": len(tree),
        "n_genomes": n_genomes, "n_masked": len(masked), "n_retained": n_genomes - len(masked),
        "actual_masking_fraction": len(masked) / n_genomes, "n_pairs": len(sub),
        "correlation": float(rho), "raw_ks": [int(k) for k in ks],
        "raw_weights": [float(w) for w in weights], **canonical_recommendation(ks, weights), **extra,
        "tree_path": str(Path(args.tree_path).resolve()),
        "merged_csv": str(Path(args.merged_csv).resolve()) if args.merged_csv else None,
        "dir_path": str(Path(args.dir_path).resolve()) if args.dir_path else None,
        "k_low": args.k_low, "k_high": args.k_high, "w_step": args.w_step,
        "s": args.s, "threads": args.threads,
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - start,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "source_sha256": {str(path): digest(path) for path in sources},
        "input_sha256": {str(path.resolve()): digest(path) for path in inputs},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
    }
    if args.save_npy:
        with args.save_npy.open("xb") as handle:
            np.save(handle, tensor)
    # A final JSON is the completion marker; interrupted writes stay .partial.
    partial = args.json_out.with_suffix(args.json_out.suffix + ".partial")
    with partial.open("x") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    partial.rename(args.json_out)
    print(f"Saved {args.json_out}", flush=True)
