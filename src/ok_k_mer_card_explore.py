"""
ok_k_mer_card_explore
=====================

Exploration helpers for recommending a single k-mer length from the dynamics
of -log10 k-mer cardinality across GTDB phylogenies.

The central object is a per-dataset ``stats_df`` (k-mer length as columns;
rows ``1st quantile``..``4th quantile``, ``average``, ``standard deviation``,
``variance``) as written by ``explore_card.ipynb`` to ``<Dataset>_kmer_stats.tsv``.
``C(k)`` denotes the normalized union cardinality stored in those rows
(union cardinality / number of distinct possible canonical k-mers), so
``C(k) -> 1`` for small k (saturation) and decays toward 0 as k grows.

This module is intentionally separate from ``ok_k_mer.py``; import it in the
notebook with ``import ok_k_mer_card_explore``.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # plotting is optional for headless use
    plt = None


# Asymptotic slope of -log10(C(k)) once each genome's distinct k-mer count is
# stable: C(k) ~= G / (4**k / 2), so -log10(C) ~= k*log10(4) - log10(2G).
ASYMPTOTIC_SLOPE = np.log10(4.0)  # ~= 0.60206

# Known single-best k per dataset (from the full grid search).
KNOWN_BEST_K = {
    "Aquificota": 20,
    "Bacillales": 24,
    "Caldisericia": 20,
    "Chlamydia": 20,
    "Cyano": 20,
    "Neisseriaceae": 19,
}


################################################################################
# Loading


def load_kmer_stats_tsv(path):
    """Load a ``<Dataset>_kmer_stats.tsv`` into a stats DataFrame.

    Returns a DataFrame whose columns are integer k-mer lengths and whose rows
    are the summary statistics (``average``, ``standard deviation``, ...).
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.columns = [int(c) for c in df.columns]
    df.columns.name = "k"
    return df


def load_all_kmer_stats(dir_path=".", pattern="*_kmer_stats.tsv"):
    """Load every ``*_kmer_stats.tsv`` in ``dir_path``.

    Returns ``{dataset_name: stats_df}`` keyed by the filename prefix
    (e.g. ``Aquificota_kmer_stats.tsv`` -> ``"Aquificota"``).
    """
    stats = {}
    for path in sorted(glob.glob(os.path.join(dir_path, pattern))):
        name = os.path.basename(path).split("_kmer_stats")[0]
        stats[name] = load_kmer_stats_tsv(path)
    return stats


_STAT_ALIASES = {
    "average": "average",
    "mean": "average",
    "median": "2nd quantile",
    "q2": "2nd quantile",
    "q1": "1st quantile",
    "q3": "3rd quantile",
    "max": "4th quantile",
    "q4": "4th quantile",
}


def _resolve_stat_row(stats_df, stat):
    """Map a friendly stat name to a row label present in ``stats_df``."""
    key = _STAT_ALIASES.get(stat.lower(), stat)
    if key not in stats_df.index:
        raise KeyError(
            f"stat '{stat}' (resolved to '{key}') not in stats rows {list(stats_df.index)}"
        )
    return key


################################################################################
# Derivative analysis (the requested idea)


def compute_logcard_derivatives(stats_df, stat="average", smooth=False,
                                smooth_window=5, smooth_polyorder=2):
    """Compute y = -log10(C(k)) and its 1st / 2nd derivatives vs k.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Per-dataset stats (columns = k, rows = summary stats).
    stat : str
        Which row to use as C(k). One of ``average`` (default), ``median``,
        ``q1``/``q3``/``max`` etc. (see ``_STAT_ALIASES``).
    smooth : bool
        If True, also return Savitzky-Golay-smoothed derivatives to dampen the
        discrete-step staircase. Falls back to the raw signal if SciPy is
        unavailable or the window is too large for the series.
    smooth_window, smooth_polyorder : int
        Savitzky-Golay parameters (window must be odd and > polyorder).

    Returns
    -------
    dict with keys ``k``, ``C``, ``y``, ``d1``, ``d2`` and, when ``smooth`` is
    requested, ``y_s``, ``d1_s``, ``d2_s``.
    """
    row = _resolve_stat_row(stats_df, stat)
    k = np.asarray(stats_df.columns, dtype=float)
    C = stats_df.loc[row].to_numpy(dtype=float)
    y = -np.log10(C)
    d1 = np.gradient(y, k)
    d2 = np.gradient(d1, k)

    out = {"k": k.astype(int), "C": C, "y": y, "d1": d1, "d2": d2}

    if smooth:
        y_s = _savgol(y, smooth_window, smooth_polyorder)
        d1_s = np.gradient(y_s, k)
        d2_s = np.gradient(d1_s, k)
        out.update({"y_s": y_s, "d1_s": d1_s, "d2_s": d2_s})

    return out


def _savgol(values, window, polyorder):
    """Savitzky-Golay smoothing with graceful fallback to the input."""
    try:
        from scipy.signal import savgol_filter
    except Exception:
        return np.asarray(values, dtype=float)
    n = len(values)
    win = min(window, n if n % 2 == 1 else n - 1)
    if win <= polyorder or win < 3:
        return np.asarray(values, dtype=float)
    if win % 2 == 0:
        win -= 1
    return savgol_filter(np.asarray(values, dtype=float), win, polyorder)


def plot_logcard_dynamics(name, stats_df, opt_k=None, stat="average",
                          smooth=False, ax=None):
    """3-panel plot of y=-log10(C), d1, d2 vs k, marking the known optimum.

    Returns the Matplotlib ``Figure``.
    """
    if plt is None:
        raise RuntimeError("matplotlib is not available")

    d = compute_logcard_derivatives(stats_df, stat=stat, smooth=smooth)
    k = d["k"]
    if opt_k is None:
        opt_k = KNOWN_BEST_K.get(name)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    panels = [
        ("y = -log10(C(k))", "y", "y_s"),
        ("d/dk  (1st derivative)", "d1", "d1_s"),
        ("d2/dk2  (2nd derivative)", "d2", "d2_s"),
    ]
    for ax_i, (title, raw_key, smooth_key) in zip(axes, panels):
        ax_i.plot(k, d[raw_key], marker="o", markersize=3, lw=1.5,
                  color="C0", label="raw")
        if smooth and smooth_key in d:
            ax_i.plot(k, d[smooth_key], lw=1.5, color="C1", label="smoothed")
        if raw_key == "d1":
            ax_i.axhline(ASYMPTOTIC_SLOPE, color="grey", ls="--", lw=1,
                         label=f"log10(4)={ASYMPTOTIC_SLOPE:.3f}")
        if raw_key == "d2":
            ax_i.axhline(0.0, color="grey", ls=":", lw=1)
        if opt_k is not None:
            ax_i.axvline(opt_k, color="C3", ls="-", lw=1.5,
                         label=f"known best k={opt_k}")
        ax_i.set_ylabel(title)
        ax_i.grid(True, ls=":", alpha=0.4)
        ax_i.legend(loc="best", frameon=False, fontsize=8)

    axes[-1].set_xlabel("k-mer length")
    axes[0].set_title(f"{name}: -log10 k-mer cardinality dynamics ({stat})")
    fig.tight_layout()
    return fig


################################################################################
# Candidate single-k estimators (derivative-based)


def estimate_k_elbow(deriv):
    """k at the maximum of the 2nd derivative (the transition 'elbow')."""
    k, d2 = deriv["k"], deriv["d2"]
    return int(k[np.nanargmax(d2)])


def estimate_k_slope_onset(deriv, frac=0.95):
    """Smallest k where the slope first reaches ``frac`` of the asymptote.

    This marks the onset of the constant-derivative (linear) regime.
    """
    k, d1 = deriv["k"], deriv["d1"]
    target = frac * ASYMPTOTIC_SLOPE
    hits = np.where(d1 >= target)[0]
    if len(hits) == 0:
        return int(k[-1])
    return int(k[hits[0]])


def estimate_k_transition_end(deriv, tol=0.01):
    """k where d2 first returns to ~0 after its peak (end of transition)."""
    k, d2 = deriv["k"], deriv["d2"]
    peak = int(np.nanargmax(d2))
    after = np.where(np.abs(d2[peak:]) <= tol)[0]
    if len(after) == 0:
        return int(k[-1])
    return int(k[peak + after[0]])


def estimate_k_onset_offset(deriv, offset, frac=0.95):
    """Linear-regime onset plus a (calibrated) constant offset."""
    return int(estimate_k_slope_onset(deriv, frac=frac) + round(offset))


def calibrate_onset_offset(stats_dict, opt_k_dict=None, stat="average",
                           frac=0.95):
    """Best constant offset between the slope-onset estimate and the optimum.

    Returns ``(offset, per_dataset_residuals)`` where ``offset`` is the mean of
    (true_k - onset_k) across datasets that have a known optimum.
    """
    if opt_k_dict is None:
        opt_k_dict = KNOWN_BEST_K
    diffs = {}
    for name, stats_df in stats_dict.items():
        if name not in opt_k_dict:
            continue
        deriv = compute_logcard_derivatives(stats_df, stat=stat)
        onset = estimate_k_slope_onset(deriv, frac=frac)
        diffs[name] = opt_k_dict[name] - onset
    if not diffs:
        return 0.0, {}
    offset = float(np.mean(list(diffs.values())))
    residuals = {n: v - offset for n, v in diffs.items()}
    return offset, residuals


def build_estimator_table(stats_dict, opt_k_dict=None, stat="average",
                          frac=0.95):
    """Tabulate every estimator's predicted k vs the known optimum.

    Returns a DataFrame indexed by dataset with predicted-k columns and their
    absolute errors, plus a final ``MAE`` row (mean absolute error per method).
    """
    if opt_k_dict is None:
        opt_k_dict = KNOWN_BEST_K

    offset, _ = calibrate_onset_offset(stats_dict, opt_k_dict, stat=stat,
                                       frac=frac)

    rows = {}
    for name, stats_df in stats_dict.items():
        deriv = compute_logcard_derivatives(stats_df, stat=stat)
        true_k = opt_k_dict.get(name, np.nan)
        preds = {
            "true_k": true_k,
            "elbow_d2max": estimate_k_elbow(deriv),
            "slope_onset": estimate_k_slope_onset(deriv, frac=frac),
            "transition_end": estimate_k_transition_end(deriv),
            "onset_plus_offset": estimate_k_onset_offset(deriv, offset, frac=frac),
        }
        rows[name] = preds

    table = pd.DataFrame(rows).T
    method_cols = ["elbow_d2max", "slope_onset", "transition_end",
                   "onset_plus_offset"]
    for col in method_cols:
        table[f"err_{col}"] = (table[col] - table["true_k"]).abs()

    mae = {c: np.nan for c in table.columns}
    for col in method_cols:
        mae[f"err_{col}"] = table[f"err_{col}"].mean()
    table.loc["MAE"] = mae
    table.attrs["calibrated_offset"] = offset
    return table


################################################################################
# Complementary signals to explore


def compute_spread_dynamics(stats_df):
    """Relative-spread signals vs k that may stay informative past the elbow.

    Returns a DataFrame indexed by k with columns ``cv`` (std/mean),
    ``iqr_log`` (-log10(Q1) - -log10(Q3) spread) and their 1st derivatives.
    """
    k = np.asarray(stats_df.columns, dtype=float)
    mean = stats_df.loc["average"].to_numpy(float)
    std = stats_df.loc["standard deviation"].to_numpy(float)
    q1 = stats_df.loc["1st quantile"].to_numpy(float)
    q3 = stats_df.loc["3rd quantile"].to_numpy(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mean > 0, std / mean, np.nan)
        iqr_log = -np.log10(q1) - (-np.log10(q3))

    out = pd.DataFrame(
        {
            "cv": cv,
            "cv_d1": np.gradient(cv, k),
            "iqr_log": iqr_log,
            "iqr_log_d1": np.gradient(iqr_log, k),
        },
        index=k.astype(int),
    )
    out.index.name = "k"
    return out


def fit_effective_genome_size(stats_df, stat="average", k_fit_min=16):
    """Fit effective genome size G from the linear tail of -log10(C(k)).

    In the unsaturated regime ``C(k) ~= G / (4**k / 2)`` so
    ``-log10(C) ~= k*log10(4) - log10(2G)``. Fitting the intercept over
    ``k >= k_fit_min`` recovers ``G``.

    Returns ``(G, slope, intercept)``.
    """
    deriv = compute_logcard_derivatives(stats_df, stat=stat)
    k, y = deriv["k"].astype(float), deriv["y"]
    mask = k >= k_fit_min
    if mask.sum() < 2:
        mask = np.ones_like(k, dtype=bool)
    slope, intercept = np.polyfit(k[mask], y[mask], 1)
    G = (10.0 ** (-intercept)) / 2.0
    return float(G), float(slope), float(intercept)


def theoretical_card(k, G):
    """Theoretical normalized union cardinality C(k) = min(1, G / (4**k / 2))."""
    k = np.asarray(k, dtype=float)
    denom = (4.0 ** k) / 2.0
    return np.minimum(1.0, G / denom)


################################################################################
# Regenerate a missing stats TSV (e.g. Cyano)


def compile_kmer_stats(merged_card_df, k_min=5, k_max=50):
    """Compile a stats DataFrame from a merged union-cardinality DataFrame.

    Mirrors the logic in ``explore_card.ipynb`` (cell 38) so a missing dataset
    TSV can be regenerated consistently.
    """
    cols = [f"k{k}_cardinality" for k in range(k_min, k_max + 1)
            if f"k{k}_cardinality" in merged_card_df.columns]
    sub = merged_card_df[cols].apply(pd.to_numeric, errors="coerce")
    stats_df = pd.DataFrame(
        {
            "1st quantile": sub.quantile(0.25),
            "2nd quantile": sub.quantile(0.50),
            "3rd quantile": sub.quantile(0.75),
            "4th quantile": sub.quantile(1.00),
            "average": sub.mean(),
            "standard deviation": sub.std(),
            "variance": sub.var(),
        }
    ).T
    stats_df.columns = [int(c[1:].split("_")[0]) for c in cols]
    stats_df.columns.name = "k"
    return stats_df


def generate_kmer_stats_tsv(tree_path, union_dir_path, out_path,
                            k_min=5, k_max=50):
    """Build and save a ``<Dataset>_kmer_stats.tsv`` from raw KMC union data.

    Uses ``ok_k_mer.process_kmc_union_card`` so normalization matches the other
    datasets, then writes the compiled stats to ``out_path``.
    """
    import ok_k_mer

    _, _, merged_card_df = ok_k_mer.process_kmc_union_card(tree_path,
                                                           union_dir_path)
    stats_df = compile_kmer_stats(merged_card_df, k_min=k_min, k_max=k_max)
    stats_df.to_csv(out_path, sep="\t")
    return stats_df
