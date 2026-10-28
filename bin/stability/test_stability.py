"""Small-grid regression checks; no Slurm submission or production simulation."""
import contextlib
import io
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from ete3 import Tree

import simulation
import stability

sys.path.insert(0, str(simulation.ORIGINAL_BIN))
import grid_search_2k as original2
import grid_search_3k as original3


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="kmer-stability-test-")
        self.root = Path(self.tmp.name)
        self.tree = Tree("((GCA_1:1,GCA_2:2):1,(GCA_3:3,(GCA_4:1,GCA_5:4):2):1);", format=1)
        self.tree_path = self.root / "tree.nwk"
        self.tree.write(format=1, outfile=str(self.tree_path))
        rows = []
        rng = np.random.default_rng(10)
        for a, b in itertools.combinations(self.tree.get_leaf_names(), 2):
            rows.append({"genome_1": a, "genome_2": b, "tree_distance": self.tree.get_distance(a, b),
                         **{f"k{k}_distance": float(rng.integers(0, 5)) for k in range(6, 9)}})
        self.df = pd.DataFrame(rows)
        self.csv_path = self.root / "merged.csv"
        self.df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_masking_protocol_and_floor(self):
        first, masked = simulation.mask_pairs(self.tree, self.df, 0.3, 123)
        second, repeated = simulation.mask_pairs(self.tree, self.df, 0.3, 123)
        expected = np.random.default_rng(123).choice(self.tree.get_leaf_names(), size=1, replace=False)
        self.assertEqual(masked, sorted(expected))
        self.assertEqual(masked, repeated)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 6)
        self.assertFalse(set(masked) & (set(first.genome_1) | set(first.genome_2)))
        unmasked, names = simulation.mask_pairs(self.tree, self.df, 0, 123)
        self.assertEqual(names, [])
        pd.testing.assert_frame_equal(unmasked, self.df)

    def test_original_search_equivalence_and_output_provenance(self):
        sub, masked = simulation.mask_pairs(self.tree, self.df, 0.3, 123)
        for max_k in (2, 3):
            with self.subTest(max_k=max_k), contextlib.redirect_stdout(io.StringIO()):
                if max_k == 2:
                    expected = original2.run_full_grid_corr(sub, k_low=6, k_high=8, w_step=0.5)
                else:
                    _, (k_main, _, _) = original3.initial_corr(sub, 6, 8)
                    expected = original3.run_full_3k_grid_corr_k_main_parallel(
                        sub, k_main, s=3, k_low=6, k_high=8, w_step=0.5, num_threads=1)
                output = self.root / f"{max_k}.json"
                tensor = output.with_suffix(".npy")
                cmd = [sys.executable, str(stability.HERE / f"grid_search_{max_k}k.py"),
                       "--tree-path", str(self.tree_path), "--merged-csv", str(self.csv_path),
                       "--dataset", "fixture", "--masking-fraction", "0.3", "--seed", "123",
                       "--k-low", "6", "--k-high", "8", "--w-step", "0.5", "--threads", "1",
                       "--json-out", str(output), "--save-npy", str(tensor)]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                actual = json.loads(output.read_text())
                self.assertEqual(actual["masked_genomes"], masked)
                self.assertAlmostEqual(actual["correlation"], expected[1])
                np.testing.assert_allclose(np.load(tensor), expected[0], rtol=0, atol=0)
                self.assertAlmostEqual(sum(actual[f"w{i}"] for i in (1, 2, 3)), 1)
                self.assertTrue(actual["source_sha256"])
                self.assertTrue(actual["input_sha256"])
                self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)

    def test_array_manifest_and_submission_receipts(self):
        data_root = self.root / "data"
        for _, folder, tree, csv_name in stability.DATASETS:
            target = data_root / folder
            (target / "grid_search").mkdir(parents=True)
            (target / tree).write_text(self.tree_path.read_text())
            (target / "grid_search" / f"{csv_name}_merged_df.csv").write_text(self.csv_path.read_text())
        run_dir = self.root / "run"
        cmd = [sys.executable, str(stability.HERE / "stability.py"), "prepare", "--run-dir", str(run_dir),
               "--data-root", str(data_root), "--k-low", "6", "--k-high", "8", "--w-step", "0.5"]
        subprocess.run(cmd, check=True, capture_output=True)
        path = run_dir / "manifest.json"
        tasks = json.loads(path.read_text())["tasks"]
        self.assertEqual(len(tasks), 636)
        self.assertEqual(len({row["seed"] for row in tasks}), 636)
        self.assertEqual(sum(row["masking_fraction"] == 0 for row in tasks), 6)
        self.assertTrue(all(sum(row["dataset"] == name and row["masking_fraction"] == 0.7
                                for row in tasks) == 15 for name, *_ in stability.DATASETS))
        # A pre-created empty directory is valid for the six-task 50% trial.
        trial = self.root / "test_run"
        trial.mkdir()
        trial_cmd = [str(trial) if x == str(run_dir) else x for x in cmd]
        trial_cmd += ["--masking-fractions", "0.5", "--repeats", "1"]
        subprocess.run(trial_cmd, check=True, capture_output=True)
        trial_path = trial / "manifest.json"
        trial_tasks = json.loads(trial_path.read_text())["tasks"]
        self.assertEqual(len(trial_tasks), 6)
        full_by_dataset = {row["dataset"]: row for row in tasks
                           if row["masking_fraction"] == 0.5 and row["simulation"] == 0}
        for row in trial_tasks:
            self.assertEqual(row["masking_fraction"], 0.5)
            self.assertEqual(row["simulation"], 0)
            self.assertEqual(row["seed"], full_by_dataset[row["dataset"]]["seed"])
            self.assertEqual(Path(row["json_out"]).parent, trial / "results")
        subprocess.run([sys.executable, str(stability.HERE / "stability.py"), "run-task",
                        "--manifest", str(trial_path), "--task-id", "0"], check=True, capture_output=True)
        subprocess.run([sys.executable, str(stability.HERE / "stability.py"), "collect",
                        "--manifest", str(trial_path)], check=True, capture_output=True)
        trial_results = pd.read_csv(trial / "stability_simulations.csv")
        self.assertEqual(len(trial_results), 1)
        self.assertTrue(trial_results.k_eff_full.isna().all())
        self.assertTrue(trial_results.matched_k_eff.isna().all())
        # Preparation must never overwrite an existing experiment.
        self.assertNotEqual(subprocess.run(trial_cmd, capture_output=True).returncode, 0)
        argv = ["stability.py", "submit", "--manifest", str(path), "--max-concurrent", "2"]
        fake = subprocess.CompletedProcess([], 0, "12345;testcluster\n", "")
        with patch.object(sys, "argv", argv), patch.object(stability.subprocess, "run", return_value=fake) as sbatch:
            stability.main()
        self.assertIn("--array=0-635%2", sbatch.call_args.args[0])
        records = list((run_dir / "submissions").glob("*.csv"))
        self.assertEqual(len(records), 1)
        import csv
        with records[0].open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[-1]["slurm_task_id"], "12345_635")
        self.assertEqual(rows[-1]["seed"], str(tasks[-1]["seed"]))
        with patch.object(sys, "argv", argv), self.assertRaises(ValueError):
            stability.main()
        # Exercise array-row dispatch, baseline comparison and partial collection.
        for task_id in (0, 31):
            env = dict(os.environ, SLURM_ARRAY_JOB_ID="12345", SLURM_ARRAY_TASK_ID=str(task_id),
                       SLURM_JOB_ID=str(12346 + task_id), SLURM_CPUS_PER_TASK="1")
            subprocess.run([sys.executable, str(stability.HERE / "stability.py"), "run-task",
                            "--manifest", str(path)], env=env, check=True, capture_output=True)
        subprocess.run([sys.executable, str(stability.HERE / "stability.py"), "collect",
                        "--manifest", str(path), "--plots"], check=True, capture_output=True)
        collected = pd.read_csv(run_dir / "stability_simulations.csv")
        self.assertEqual(len(collected), 2)
        self.assertEqual(collected.iloc[0].abs_dev, 0)
        self.assertAlmostEqual(collected.iloc[0].corr_retention, 1)
        self.assertFalse(collected.k_eff_full.isna().any())
        self.assertTrue((run_dir / "effective_k_vs_masking.pdf").is_file())
        self.assertEqual(len(pd.read_csv(run_dir / "status.csv")), 636)
        # Resubmission skips completed tasks, preserves their IDs, and defaults to 24.
        resume_argv = ["stability.py", "submit", "--manifest", str(path), "--resume-missing"]
        with patch.object(sys, "argv", resume_argv), patch.object(stability.subprocess, "run", return_value=fake) as sbatch:
            with contextlib.redirect_stdout(io.StringIO()):
                stability.main()
        array_flag = next(v for v in sbatch.call_args.args[0] if v.startswith("--array="))
        spec, throttle = array_flag.removeprefix("--array=").split("%")
        self.assertEqual(throttle, "24")
        self.assertEqual(set(map(int, spec.split(","))), set(range(636)) - {0, 31})


if __name__ == "__main__":
    unittest.main()
