#!/usr/bin/env python3
"""Prepare, submit, execute and collect independent Slurm stability simulations."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATASETS = [
    ("Aquificota", "aquificota", "p__Aquificota.nwk", "aquificota"),
    ("Bacillales", "bacillales", "o__Bacillales_A.nwk", "bacillales"),
    ("Caldisericia", "caldisericia", "c__Caldisericia.nwk", "caldisericia"),
    ("Chlamydia", "chlamydiales", "o__Chlamydiales.nwk", "chlamydia"),
    ("Cyano", "cyanobacteriales", "o__Cyanobacteriales.nwk", "cyano"),
    ("Neisseriaceae", "neisseriaceae", "f__Neisseriaceae.nwk", "neisseriaceae"),
]


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path):
    with Path(path).open() as handle:
        return json.load(handle)


def prepare(args):
    if args.repeats < 1 or args.master_seed < 0:
        raise ValueError("repeats must be positive and master-seed nonnegative")
    if args.k_low < 1 or args.k_high < args.k_low or args.s < 0:
        raise ValueError("Invalid k range or window radius")
    if not (0 < args.w_step <= 1) or not math.isclose(1 / args.w_step, round(1 / args.w_step)):
        raise ValueError("w-step must divide 1 exactly")
    if not math.isclose(args.w_step, round(args.w_step, 3)):
        raise ValueError("w-step supports at most three decimal places")
    if not args.masking_fractions or any(not 0 <= f < 1 for f in args.masking_fractions):
        raise ValueError("masking-fractions must be in [0, 1)")
    if any(not math.isclose(f * 100, round(f * 100), abs_tol=1e-9) for f in args.masking_fractions):
        raise ValueError("masking-fractions must use whole percentage points, e.g. 0.5")
    percentages = [round(f * 100) for f in args.masking_fractions]
    if len(set(percentages)) != len(percentages):
        raise ValueError("masking-fractions must not contain duplicates")
    if args.run_dir.exists() and (not args.run_dir.is_dir() or any(args.run_dir.iterdir())):
        raise ValueError("Run directory must be new or empty")
    tasks = []
    for name, folder, tree_name, csv_name in DATASETS:
        tree = (args.data_root / folder / tree_name).resolve()
        merged = (args.data_root / folder / "grid_search" / f"{csv_name}_merged_df.csv").resolve()
        for path in (tree, merged):
            if not path.is_file():
                raise FileNotFoundError(path)
        for percentage in percentages:
            for simulation in range(1 if percentage == 0 else args.repeats):
                # Stable across max_k, scheduling order, Python versions and subsets.
                key = f"{args.master_seed}:{name}:{percentage}:{simulation}"
                seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
                task_id = len(tasks)
                tasks.append({
                    "task_id": task_id, "dataset": name, "max_k": args.max_k,
                    "masking_fraction": percentage / 100, "masking_percentage": percentage,
                    "simulation": simulation, "seed": seed, "master_seed": args.master_seed,
                    "tree_path": str(tree), "merged_csv": str(merged),
                    "k_low": args.k_low, "k_high": args.k_high, "w_step": args.w_step,
                    "s": args.s, "save_tensor": args.save_tensors,
                    "json_out": str(args.run_dir.resolve() / "results" / f"{task_id:04d}_{name}_mask{percentage:02d}_sim{simulation:02d}.json"),
                })
    if len({row["seed"] for row in tasks}) != len(tasks):
        raise ValueError("Seed collision")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    for folder in ["results", "logs", "submissions"]:
        (args.run_dir / folder).mkdir()
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "max_k": args.max_k, "tasks": tasks}
    with (args.run_dir / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    write_csv(args.run_dir / "tasks.csv", tasks)
    print(f"Prepared {len(tasks)} tasks: {args.run_dir.resolve() / 'manifest.json'}")


def submit(args):
    manifest = read_manifest(args.manifest)
    run_dir = args.manifest.resolve().parent
    receipts = list((run_dir / "submissions").glob("*.json"))
    if receipts and args.task_ids is None and not args.resume_missing:
        raise ValueError("Already submitted; use --resume-missing after previous jobs stop, or --task-ids ID,ID")
    if args.resume_missing:
        ids = [i for i, task in enumerate(manifest["tasks"]) if not Path(task["json_out"]).exists()]
        if not ids:
            print("All tasks already have results; nothing to submit.")
            return
    else:
        ids = list(range(len(manifest["tasks"]))) if args.task_ids is None else [int(x) for x in args.task_ids.split(",")]
    if not ids or len(ids) != len(set(ids)) or any(i < 0 or i >= len(manifest["tasks"]) for i in ids):
        raise ValueError("Invalid or duplicate task IDs")
    if args.max_concurrent < 1:
        raise ValueError("max-concurrent must be positive")
    if any(Path(manifest["tasks"][i]["json_out"]).exists() for i in ids):
        raise ValueError("Selected tasks already have completed results; prepare a new run directory")
    spec = (f"0-{len(ids) - 1}" if ids == list(range(len(manifest["tasks"]))) else ",".join(map(str, ids)))
    cmd = ["sbatch", "--parsable", f"--array={spec}%{args.max_concurrent}",
           f"--output={run_dir}/logs/%A_%a.out", f"--error={run_dir}/logs/%A_%a.err"]
    for flag in ["account", "partition", "time", "mem", "cpus_per_task"]:
        value = getattr(args, flag)
        if value is not None:
            cmd.append(f"--{flag.replace('_', '-')}={value}")
    cmd += [str(HERE / f"grid_search_{manifest['max_k']}k_slurm.sh"), "--manifest", str(args.manifest.resolve())]
    print("Submission command:", " ".join(cmd), flush=True)
    # Persist the intent before invoking sbatch, and raw response before parsing.
    receipt_path = run_dir / "submissions" / f"{time.time_ns()}.json"
    receipt = {"submitted_utc": datetime.now(timezone.utc).isoformat(), "command": cmd,
               "task_ids": ids, "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    env = dict(os.environ, STABILITY_BIN=str(HERE))
    result = subprocess.run(cmd, text=True, capture_output=True, env=env)
    receipt.update({"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode})
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"sbatch failed: {result.stderr}; receipt: {receipt_path}")
    match = re.fullmatch(r"(\d+)(?:;([^\s]+))?", result.stdout.strip())
    if match is None:
        raise RuntimeError(f"Unrecognized sbatch response; inspect {receipt_path} before retrying")
    job_id, cluster = match.groups()
    receipt.update({"array_job_id": job_id, "cluster": cluster})
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    rows = [{**manifest["tasks"][i], "array_job_id": job_id,
             "array_task_id": i, "slurm_task_id": f"{job_id}_{i}",
             "cluster": cluster, "submitted_utc": receipt["submitted_utc"]} for i in ids]
    write_csv(receipt_path.with_suffix(".csv"), rows)
    print(f"Submitted array {job_id} ({len(ids)} independent tasks). Recorded: {receipt_path.with_suffix('.csv')}")


def run_task(args):
    manifest = read_manifest(args.manifest)
    task_id = args.task_id if args.task_id is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= task_id < len(manifest["tasks"]):
        raise ValueError("Task ID outside manifest")
    row = manifest["tasks"][task_id]
    cmd = [sys.executable, str(HERE / f"grid_search_{row['max_k']}k.py")]
    for key in ["tree_path", "merged_csv", "dataset", "masking_fraction", "simulation", "seed",
                "k_low", "k_high", "w_step", "s", "json_out"]:
        cmd += [f"--{key.replace('_', '-')}", str(row[key])]
    cmd += ["--threads", os.environ.get("SLURM_CPUS_PER_TASK", "1")]
    if row.get("excluded_genomes"):
        cmd += ["--exclude-genomes", *row["excluded_genomes"]]
    if row["save_tensor"]:
        cmd += ["--save-npy", str(Path(row["json_out"]).with_suffix(".npy"))]
    subprocess.run(cmd, check=True)


def collect(args):
    import numpy as np
    import pandas as pd

    if not math.isfinite(args.tol) or args.tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    manifest = read_manifest(args.manifest)
    run_dir = args.manifest.resolve().parent
    completed, status = [], []
    submissions = {}
    for path in sorted((run_dir / "submissions").glob("*.json")):
        receipt = json.loads(path.read_text())
        if "array_job_id" in receipt:
            for task_id in receipt["task_ids"]:
                submissions[task_id] = f"{receipt['array_job_id']}_{task_id}"
    for row in manifest["tasks"]:
        path = Path(row["json_out"])
        state = "complete" if path.exists() else "missing"
        if path.exists():
            result = json.loads(path.read_text())
            for key in ["dataset", "max_k", "masking_fraction", "simulation", "seed", "k_low", "k_high", "w_step", "s", "tree_path", "merged_csv"]:
                if result[key] != row[key]:
                    raise ValueError(f"Result/manifest mismatch ({key}): {path}")
            flat = {k: v for k, v in result.items() if not isinstance(v, (list, dict))}
            completed.append({**flat, "task_id": row["task_id"], "master_seed": row["master_seed"]})
        status.append({**row, "result_status": state,
                       "submitted_slurm_task_id": submissions.get(row["task_id"], "")})
    write_csv(run_dir / "status.csv", status)
    if not completed:
        print(f"No completed results. Wrote {run_dir / 'status.csv'}")
        return
    df = pd.DataFrame(completed)
    baseline = df.loc[df.masking_fraction == 0].set_index("dataset")
    df["k_eff_full"] = df.dataset.map(baseline.k_eff)
    df["corr_full"] = df.dataset.map(baseline.correlation)
    df["k_eff_dev"] = df.k_eff - df.k_eff_full
    df["abs_dev"] = df.k_eff_dev.abs()
    df["k_eff_match_tolerance"] = args.tol
    df["matched_k_eff"] = (df.abs_dev <= args.tol).astype("boolean").mask(df.k_eff_full.isna())
    df["corr_retention"] = df.correlation.abs() / df.corr_full.abs().replace(0, np.nan)
    for k in ["k1", "k2", "k3"]:
        df[k] = df[k].astype("Int64")
    df = df.sort_values(["dataset", "masking_fraction", "simulation"])
    df.to_csv(run_dir / "stability_simulations.csv", index=False)
    grouped = df.groupby(["dataset", "masking_percentage"])
    summary = grouped.agg(n=("simulation", "size"), k_eff_mean=("k_eff", "mean"),
                          k_eff_std=("k_eff", "std"), correlation_mean=("correlation", "mean"),
                          correlation_std=("correlation", "std"), match_rate=("matched_k_eff", "mean"),
                          mean_abs_dev=("abs_dev", "mean"), corr_retention_mean=("corr_retention", "mean"))
    summary.to_csv(run_dir / "stability_summary.csv")
    print(f"Collected {len(df)}/{len(status)} results into {run_dir / 'stability_simulations.csv'}")
    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for metric, label, stem, sign in [
            ("k_eff", "Effective k-mer length", "effective_k_vs_masking", 1),
            ("correlation", "Spearman correlation (sign flipped)", "spearman_vs_masking", -1),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for name, group in df.groupby("dataset", sort=False):
                stats = group.groupby("masking_fraction")[metric].agg(["mean", "std"])
                x = stats.index.to_numpy()
                y = sign * stats["mean"].to_numpy()
                spread = stats["std"].fillna(0).to_numpy()
                line, = ax.plot(x, y, marker="o", label=name)
                ax.fill_between(x, y - spread, y + spread, alpha=0.15, color=line.get_color())
            ax.set(xlabel="Masking fraction", ylabel=label)
            ax.legend(frameon=False)
            fig.tight_layout()
            for extension in ["pdf", "png"]:
                fig.savefig(run_dir / f"{stem}.{extension}", dpi=300)
            plt.close(fig)


def main():
    if sys.version_info < (3, 8):
        raise SystemExit("Use Python >=3.8, e.g. /home/shwang45/miniconda3/envs/python310/bin/python")
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Write a task manifest (636 tasks by default); does not submit")
    prep.add_argument("--run-dir", type=Path, required=True)
    prep.add_argument("--data-root", type=Path, default=ROOT / "data")
    prep.add_argument("--max-k", type=int, choices=[2, 3], default=3)
    prep.add_argument("--repeats", type=int, default=15)
    prep.add_argument("--masking-fractions", type=float, nargs="+",
                      default=[i / 10 for i in range(8)],
                      help="Fractions to run, in whole percentage points (default: 0.0 through 0.7)")
    prep.add_argument("--master-seed", type=int, default=0)
    prep.add_argument("--k-low", type=int, default=6)
    prep.add_argument("--k-high", type=int, default=40)
    prep.add_argument("--w-step", type=float, default=0.05)
    prep.add_argument("--s", type=int, default=3)
    prep.add_argument("--save-tensors", action="store_true")
    prep.set_defaults(func=prepare)
    submit_p = sub.add_parser("submit", help="Submit one array and record IDs plus all task parameters")
    submit_p.add_argument("--manifest", type=Path, required=True)
    submit_p.add_argument("--max-concurrent", type=int, default=24, help="Maximum simultaneous tasks (default: 24)")
    selection = submit_p.add_mutually_exclusive_group()
    selection.add_argument("--task-ids", help="Explicit comma-separated task IDs for a subset or retry")
    selection.add_argument("--resume-missing", action="store_true",
                           help="Resubmit tasks without result JSONs; use only after previous jobs have stopped")
    for flag in ["account", "partition", "time", "mem", "cpus-per-task"]:
        submit_p.add_argument(f"--{flag}")
    submit_p.set_defaults(func=submit)
    task = sub.add_parser("run-task")
    task.add_argument("--manifest", type=Path, required=True)
    task.add_argument("--task-id", type=int)
    task.set_defaults(func=run_task)
    collector = sub.add_parser("collect", help="Collect available results; missing tasks appear in status.csv")
    collector.add_argument("--manifest", type=Path, required=True)
    collector.add_argument("--tol", type=float, default=1.0)
    collector.add_argument("--plots", action="store_true")
    collector.set_defaults(func=collect)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
