"""
ok_k_mer_card_explore
=====================

Exploration helpers for recommending a single k-mer length from the dynamics
of -log10 k-mer cardinality across GTDB phylogenies.

The central object is a per-dataset ``stats_df`` (k-mer length as columns;
rows ``1st quantile``..``4th quantile``, ``average``, ``standard deviation``,
``variance``) as written by ``explore_card.ipynb`` to ``<Dataset>_kmer_stats.tsv``.
``C(k)`` denotes the raw union cardinality; the value stored in those rows
and shown in plots is the normalized ``C(k) / 4^k`` (union cardinality /
number of distinct possible canonical k-mers), so ``C(k) / 4^k -> 1`` for
small k (saturation) and decays toward 0 as k grows.

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


# Asymptotic slope of -log10(C(k) / 4^k) once each genome's distinct k-mer
# count is stable: C(k) / 4^k ~= G / (4**k / 2), so
# -log10(C(k) / 4^k) ~= k*log10(4) - log10(2G).
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


def num_distinct_canonical_kmers(k):
    """Number of distinct canonical k-mers (reverse-complement collapsed).

    Matches the normalizer used in ``ok_k_mer.build_merged_kmc_union_card``:
    for even k, ``(4**k - 4**(k/2)) / 2 + 4**(k/2)``; for odd k, ``4**k / 2``.
    Accepts a scalar or array of k and returns a float array.
    """
    k = np.asarray(k, dtype=float)
    num_pal = 4.0 ** (k / 2.0)
    even = (4.0 ** k - num_pal) / 2.0 + num_pal
    odd = 4.0 ** k / 2.0
    return np.where(k % 2 == 0, even, odd)


def denormalize_stats_df(stats_df):
    """Convert normalized stats (C(k) / 4^k) back to raw cardinality C(k).

    Multiplies each per-k column by ``num_distinct_canonical_kmers(k)``. Linear
    stats (quantiles, average, standard deviation) scale by that factor;
    ``variance`` scales by the factor squared. This is exact because every pair
    in a given k column was divided by the same constant.
    """
    k = np.asarray(stats_df.columns, dtype=float)
    factor = num_distinct_canonical_kmers(k)
    out = stats_df.copy()
    for row in out.index:
        scale = factor ** 2 if row == "variance" else factor
        out.loc[row] = out.loc[row].to_numpy(dtype=float) * scale
    return out


################################################################################
# Derivative analysis (the requested idea)


def compute_logcard_derivatives(stats_df, stat="average", smooth=False,
                                smooth_window=5, smooth_polyorder=2,
                                method="gradient", transform="neglog10",
                                normalized=True):
    """Compute y(k) and its 1st / 2nd derivatives vs k.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Per-dataset stats (columns = k, rows = summary stats).
    stat : str
        Which row to use as C(k) / 4^k. One of ``average`` (default),
        ``median``, ``q1``/``q3``/``max`` etc. (see ``_STAT_ALIASES``).
    smooth : bool
        If True, also return Savitzky-Golay-smoothed derivatives to dampen the
        discrete-step staircase. Falls back to the raw signal if SciPy is
        unavailable or the window is too large for the series.
    smooth_window, smooth_polyorder : int
        Savitzky-Golay parameters (window must be odd and > polyorder).
    method : {"gradient", "diff"}
        How to differentiate. ``"gradient"`` (default) uses ``np.gradient``
        (central differences; same-length, aligned to ``k``). ``"diff"`` uses
        ``np.diff`` (forward differences), so ``d1`` has length ``n-1`` and sits
        at the midpoints between k values, and ``d2`` has length ``n-2`` at the
        interior integer k values. Use ``k_d1`` / ``k_d2`` for plotting.
    transform : {"neglog10", "raw"}
        Whether to apply the -log10 transform. ``"neglog10"`` (default) sets
        ``y = -log10(.)``; ``"raw"`` differentiates the cardinality directly
        (no log).
    normalized : bool
        Whether to use the normalized cardinality ``C(k) / 4^k`` (default,
        ``True``) or the raw cardinality ``C(k)``. ``stats_df`` is assumed to
        store the normalized form (as the saved TSVs do); when ``normalized`` is
        ``False`` the per-k column is multiplied back by
        ``num_distinct_canonical_kmers(k)`` to recover the raw count. ``C`` in
        the returned dict reflects this choice.

    Returns
    -------
    dict with keys ``k``, ``C``, ``y``, ``d1``, ``d2`` and the matching
    x-coordinates ``k_d1`` (for ``d1``) and ``k_d2`` (for ``d2``). When
    ``smooth`` is requested, also ``y_s``, ``d1_s``, ``d2_s``.
    """
    if method not in ("gradient", "diff"):
        raise ValueError(f"method must be 'gradient' or 'diff', got {method!r}")
    if transform not in ("neglog10", "raw"):
        raise ValueError(
            f"transform must be 'neglog10' or 'raw', got {transform!r}"
        )

    row = _resolve_stat_row(stats_df, stat)
    k = np.asarray(stats_df.columns, dtype=float)
    C = stats_df.loc[row].to_numpy(dtype=float)
    if not normalized:
        C = C * num_distinct_canonical_kmers(k)
    y = -np.log10(C) if transform == "neglog10" else C.copy()

    d1, d2, k_d1, k_d2 = _differentiate(y, k, method)
    out = {
        "k": k.astype(int),
        "C": C,
        "y": y,
        "d1": d1,
        "d2": d2,
        "k_d1": k_d1,
        "k_d2": k_d2,
        "method": method,
        "transform": transform,
        "normalized": normalized,
    }

    if smooth:
        y_s = _savgol(y, smooth_window, smooth_polyorder)
        d1_s, d2_s, _, _ = _differentiate(y_s, k, method)
        out.update({"y_s": y_s, "d1_s": d1_s, "d2_s": d2_s})

    return out


def _differentiate(y, k, method):
    """Return (d1, d2, k_d1, k_d2) for ``y`` sampled at ``k`` using ``method``.

    ``gradient`` keeps everything aligned to ``k``; ``diff`` returns shorter
    arrays located at the midpoints (d1) and interior integer points (d2).
    """
    if method == "gradient":
        d1 = np.gradient(y, k)
        d2 = np.gradient(d1, k)
        return d1, d2, k, k

    # forward differences (np.diff), handling possibly non-uniform k
    d1 = np.diff(y) / np.diff(k)
    k_d1 = (k[:-1] + k[1:]) / 2.0          # midpoints between consecutive k
    if len(d1) >= 2:
        d2 = np.diff(d1) / np.diff(k_d1)
        k_d2 = k[1:-1]                     # interior integer k values
    else:
        d2 = np.array([])
        k_d2 = np.array([])
    return d1, d2, k_d1, k_d2


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
                          smooth=False, ax=None, transform="neglog10",
                          normalized=True):
    """3-panel plot of y(k), d1, d2 vs k, marking the known optimum.

    ``normalized=True`` (default) uses ``C(k) / 4^k``; ``normalized=False`` uses
    the raw cardinality ``C(k)``. ``transform="neglog10"`` (default) plots
    ``y = -log10(.)``; ``transform="raw"`` plots the cardinality directly. The
    derivatives are taken on whichever quantity is chosen.

    Returns the Matplotlib ``Figure``.
    """
    if plt is None:
        raise RuntimeError("matplotlib is not available")

    d = compute_logcard_derivatives(stats_df, stat=stat, smooth=smooth,
                                    transform=transform, normalized=normalized)
    k = d["k"]
    if opt_k is None:
        opt_k = KNOWN_BEST_K.get(name)

    base = "C(k) / 4^k" if normalized else "C(k)"
    qualifier = "normalized" if normalized else "raw"
    if transform == "neglog10":
        y_label = f"y = -log10({base})"
        metric_label = f"-log10 {qualifier} k-mer cardinality"
    else:
        y_label = f"y = {base}"
        metric_label = f"{qualifier} k-mer cardinality"

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    panels = [
        (y_label, "y", "y_s"),
        ("d/dk  (1st derivative)", "d1", "d1_s"),
        ("d2/dk2  (2nd derivative)", "d2", "d2_s"),
    ]
    for ax_i, (title, raw_key, smooth_key) in zip(axes, panels):
        ax_i.plot(k, d[raw_key], marker="o", markersize=3, lw=1.5,
                  color="C0", label="raw")
        if smooth and smooth_key in d:
            ax_i.plot(k, d[smooth_key], lw=1.5, color="C1", label="smoothed")
        # The log10(4) asymptote only applies to -log10 of the normalized C(k).
        if raw_key == "d1" and transform == "neglog10" and normalized:
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
    axes[-1].set_xticks(k)
    axes[-1].set_xticklabels(k, fontsize=7, rotation=90)
    axes[0].set_title(f"{name}: {metric_label} dynamics ({stat})")
    fig.tight_layout()
    return fig


def _aggregate_card_series(card_df, stat="average", spread=None):
    """Aggregate a per-pair card frame into a per-k normalized series.

    Picks out the ``k{k}_cardinality`` columns (already stored as the normalized
    ``C(k) / 4^k``) and aggregates across genome pairs with the mean (default)
    or median.

    Returns ``(k, central)`` sorted by ascending k. When ``spread`` is given,
    also returns lower/upper bounds describing the pairwise distribution at each
    k: ``(k, central, lo, hi)``. ``spread`` may be:

    - ``"iqr"``: 25th / 75th percentiles across pairs.
    - ``"minmax"``: minimum / maximum across pairs.
    - ``"std"``: central +/- one standard deviation.
    """
    cols = [c for c in card_df.columns if re.fullmatch(r"k\d+_cardinality", str(c))]
    k = np.array([int(str(c)[1:].split("_")[0]) for c in cols])
    order = np.argsort(k)
    cols = [cols[i] for i in order]
    sub = card_df[cols].apply(pd.to_numeric, errors="coerce")
    agg = sub.median() if stat in ("median", "q2", "2nd quantile") else sub.mean()
    central = agg.to_numpy(dtype=float)
    if spread is None:
        return k[order], central
    if spread == "iqr":
        lo = sub.quantile(0.25).to_numpy(dtype=float)
        hi = sub.quantile(0.75).to_numpy(dtype=float)
    elif spread == "minmax":
        lo = sub.min().to_numpy(dtype=float)
        hi = sub.max().to_numpy(dtype=float)
    elif spread == "std":
        s = sub.std().to_numpy(dtype=float)
        lo, hi = central - s, central + s
    else:
        raise ValueError(
            f"spread must be None, 'iqr', 'minmax' or 'std', got {spread!r}"
        )
    return k[order], central, lo, hi


def plot_union_intersection_card(name, union_card_df, intersection_card_df,
                                 opt_k=None, stat="average", logy_raw=True,
                                 max_k=None, spread="iqr", show_median=True):
    """Two-panel plot of union vs intersection k-mer cardinality vs k.

    Top panel shows the raw (non-normalized) cardinality ``C(k)`` (on a log
    y-axis by default, since it spans ~``4^k``); bottom panel shows the
    normalized ``C(k) / 4^k`` using ``num_distinct_canonical_kmers``. Both union
    and intersection are drawn over their own available k-range. ``opt_k`` (the
    best single k, defaulting to ``KNOWN_BEST_K[name]``) is marked with a red
    dotted vertical line. ``max_k`` (if given) caps the plotted k-range to
    ``k <= max_k``.

    ``spread`` shades the pairwise distribution around each central line:
    ``"iqr"`` (default, 25th-75th percentile), ``"minmax"``, ``"std"``, or
    ``None`` to draw only the central line.

    ``show_median`` (default True) overlays a dashed per-pair median line. This
    is useful because the per-pair distribution is right-skewed, so the mean can
    sit outside the IQR band while the median always lies within it.

    Returns the Matplotlib ``Figure``.
    """
    if plt is None:
        raise RuntimeError("matplotlib is not available")

    res_u = _aggregate_card_series(union_card_df, stat=stat, spread=spread)
    res_i = _aggregate_card_series(intersection_card_df, stat=stat, spread=spread)
    if spread is None:
        (k_u, norm_u), (k_i, norm_i) = res_u, res_i
        lo_u = hi_u = lo_i = hi_i = None
    else:
        k_u, norm_u, lo_u, hi_u = res_u
        k_i, norm_i, lo_i, hi_i = res_i

    central_name = "median" if stat in ("median", "q2", "2nd quantile") else "mean"
    add_median = show_median and central_name != "median"
    if add_median:
        _, med_u = _aggregate_card_series(union_card_df, stat="median")
        _, med_i = _aggregate_card_series(intersection_card_df, stat="median")
    else:
        med_u = med_i = None

    def _apply_max_k(k, *arrays):
        if max_k is None:
            return (k, *arrays)
        m = k <= max_k
        return (k[m], *(None if a is None else a[m] for a in arrays))

    k_u, norm_u, lo_u, hi_u, med_u = _apply_max_k(k_u, norm_u, lo_u, hi_u, med_u)
    k_i, norm_i, lo_i, hi_i, med_i = _apply_max_k(k_i, norm_i, lo_i, hi_i, med_i)

    factor_u = num_distinct_canonical_kmers(k_u)
    factor_i = num_distinct_canonical_kmers(k_i)
    raw_u = norm_u * factor_u
    raw_i = norm_i * factor_i

    if opt_k is None:
        opt_k = KNOWN_BEST_K.get(name)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    def _pos(a):
        # mask non-positive values so a log-scaled band has no -inf edges
        return np.where(np.asarray(a, dtype=float) > 0, a, np.nan)

    if spread is not None:
        ax_top.fill_between(k_u, _pos(lo_u * factor_u), _pos(hi_u * factor_u),
                            color="C0", alpha=0.2, lw=0)
        ax_top.fill_between(k_i, _pos(lo_i * factor_i), _pos(hi_i * factor_i),
                            color="C1", alpha=0.2, lw=0)
    ax_top.plot(k_u, raw_u, marker="o", markersize=3, lw=1.5, color="C0",
                label=f"union ({central_name})")
    ax_top.plot(k_i, raw_i, marker="s", markersize=3, lw=1.5, color="C1",
                label=f"intersection ({central_name})")
    if add_median:
        ax_top.plot(k_u, _pos(med_u * factor_u), ls="--", lw=1.2, color="C0",
                    label="union (median)")
        ax_top.plot(k_i, _pos(med_i * factor_i), ls="--", lw=1.2, color="C1",
                    label="intersection (median)")
    if logy_raw:
        ax_top.set_yscale("log")
    ax_top.set_ylabel("cardinality C(k)")

    if spread is not None:
        ax_bot.fill_between(k_u, lo_u, hi_u, color="C0", alpha=0.2, lw=0)
        ax_bot.fill_between(k_i, lo_i, hi_i, color="C1", alpha=0.2, lw=0)
    ax_bot.plot(k_u, norm_u, marker="o", markersize=3, lw=1.5, color="C0",
                label=f"union ({central_name})")
    ax_bot.plot(k_i, norm_i, marker="s", markersize=3, lw=1.5, color="C1",
                label=f"intersection ({central_name})")
    if add_median:
        ax_bot.plot(k_u, med_u, ls="--", lw=1.2, color="C0",
                    label="union (median)")
        ax_bot.plot(k_i, med_i, ls="--", lw=1.2, color="C1",
                    label="intersection (median)")
    ax_bot.set_ylabel("C(k) / 4^k")
    ax_bot.set_xlabel("k-mer length")

    spread_label = {"iqr": "IQR", "minmax": "min-max", "std": "+/-1 std"}.get(spread)
    lines_note = f"{central_name}+median" if add_median else central_name
    band_note = f" (lines={lines_note}, band={spread_label})" if spread else \
        f" ({lines_note})"
    ax_top.set_title(f"{name}: union vs intersection k-mer cardinality{band_note}")

    k_all = np.union1d(k_u, k_i).astype(int)
    for ax in (ax_top, ax_bot):
        if opt_k is not None:
            ax.axvline(opt_k, color="red", ls=":", lw=1.5,
                       label=f"best k={opt_k}")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(loc="best", frameon=False, fontsize=8)
    ax_bot.set_xticks(k_all)
    ax_bot.set_xticklabels(k_all, fontsize=7, rotation=90)

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
    """Fit effective genome size G from the linear tail of -log10(C(k) / 4^k).

    In the unsaturated regime ``C(k) / 4^k ~= G / (4**k / 2)`` so
    ``-log10(C(k) / 4^k) ~= k*log10(4) - log10(2G)``. Fitting the intercept over
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
    """Theoretical normalized union cardinality C(k) / 4^k = min(1, G / (4**k / 2))."""
    k = np.asarray(k, dtype=float)
    denom = (4.0 ** k) / 2.0
    return np.minimum(1.0, G / denom)


################################################################################
# Regenerate a missing stats TSV (e.g. Cyano)


def verify_pair_count(merged_card_df, expected_genomes=None):
    """Sanity-check that a merged union-cardinality frame is internally consistent.

    Guards against the cross-dataset mislabeling bug where a cached frame held
    another clade's data. Infers the genome count from the ``genome_1`` /
    ``genome_2`` columns, requires the row count to equal the complete
    upper-triangle pair count ``n*(n-1)/2``, and (optionally) checks that the
    genome count matches ``expected_genomes``.

    Returns ``(n_genomes, n_pairs)``. Raises ``ValueError`` on any mismatch.
    """
    if not {"genome_1", "genome_2"}.issubset(merged_card_df.columns):
        raise ValueError(
            "merged_card_df is missing 'genome_1'/'genome_2' columns; "
            "cannot verify pair count"
        )

    genomes = set(merged_card_df["genome_1"]) | set(merged_card_df["genome_2"])
    n = len(genomes)
    expected_pairs = n * (n - 1) // 2
    n_rows = len(merged_card_df)

    if n_rows != expected_pairs:
        raise ValueError(
            f"row count {n_rows} != complete pair count {expected_pairs} for "
            f"{n} genomes; the frame is incomplete or contains mismatched data"
        )

    if expected_genomes is not None and n != expected_genomes:
        raise ValueError(
            f"found {n} genomes but expected {expected_genomes}; the frame "
            f"likely contains the wrong dataset"
        )

    return n, n_rows


def compile_kmer_stats(merged_card_df, k_min=5, k_max=50, normalize=True):
    """Compile a stats DataFrame from a merged union-cardinality DataFrame.

    Mirrors the logic in ``explore_card.ipynb`` (cell 38) so a missing dataset
    TSV can be regenerated consistently.

    ``ok_k_mer.process_kmc_union_card`` already divides by the number of
    possible canonical k-mers, so the columns are the normalized ``C(k) / 4^k``.
    With ``normalize=False`` the compiled stats are de-normalized back to the
    raw cardinality ``C(k)`` via ``denormalize_stats_df``.
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
    if not normalize:
        stats_df = denormalize_stats_df(stats_df)
    return stats_df


def generate_kmer_stats_tsv(tree_path, union_dir_path, out_path,
                            k_min=5, k_max=50, expected_genomes=None,
                            normalize=True):
    """Build and save a ``<Dataset>_kmer_stats.tsv`` from raw KMC union data.

    Uses ``ok_k_mer.process_kmc_union_card`` so normalization matches the other
    datasets, then writes the compiled stats to ``out_path``. Set
    ``normalize=False`` to save the raw (un-normalized) cardinality C(k).

    Runs ``verify_pair_count`` before writing to catch the cross-dataset
    mislabeling bug; pass ``expected_genomes`` to also assert the clade size.
    """
    import ok_k_mer

    _, _, merged_card_df = ok_k_mer.process_kmc_union_card(tree_path,
                                                           union_dir_path)
    n, n_pairs = verify_pair_count(merged_card_df,
                                   expected_genomes=expected_genomes)
    print(f"pair-count check OK: {n} genomes, {n_pairs} pairs")
    stats_df = compile_kmer_stats(merged_card_df, k_min=k_min, k_max=k_max,
                                  normalize=normalize)
    stats_df.to_csv(out_path, sep="\t")
    return stats_df


def save_raw_stats_tsvs(in_dir, out_dir, pattern="*_kmer_stats.tsv"):
    """Write raw (un-normalized) copies of normalized stats TSVs.

    Loads every normalized ``*_kmer_stats.tsv`` in ``in_dir``, de-normalizes it
    to raw cardinality C(k) with ``denormalize_stats_df``, and writes the result
    to ``out_dir`` under the same filename. Returns the list of output paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, stats_df in load_all_kmer_stats(in_dir, pattern=pattern).items():
        out_path = os.path.join(out_dir, f"{name}_kmer_stats.tsv")
        denormalize_stats_df(stats_df).to_csv(out_path, sep="\t")
        written.append(out_path)
    return written
