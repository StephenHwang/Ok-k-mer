"""
ok_k_mer_stability_explore
==========================

Exploration helpers for asking how the jaccard distance metric and its
correlation with true tree distance behave as a function of k-mer length, and
whether there is a *range* of k that gives stable correlation (rather than a
single brittle optimum).

It also provides a ground-truth-free selection signal based purely on the shape
of the jaccard-mean-vs-k curve: its 1st/2nd derivatives and the
saturation->decline "elbow" (argmin of the 2nd derivative), analogous to the
cardinality elbow in ``ok_k_mer_card_explore``.

Everything here is derived directly from a per-dataset ``merged_df`` (one row
per genome pair) following the conventions in ``ok_k_mer_selection``:

* ``merged_df``: columns ``tree_distance`` and ``k{k}_distance`` (the jaccard
  metric) plus (optionally) a ``genome_pair`` column. The distance metric is
  similarity-like, so it correlates *negatively* with tree distance; we report
  the raw correlation and its magnitude (``abs_corr``) and pick the single-best
  k by ``argmax`` of ``abs_corr``.

Import in the notebook with ``import ok_k_mer_stability_explore``.
"""

import re

import numpy as np
import pandas as pd

import ok_k_mer_card_explore as ce

try:
    import matplotlib.pyplot as plt
except Exception:  # plotting is optional for headless use
    plt = None


################################################################################
# Column helpers


def _distance_k_columns(merged_df):
    """Return the sorted list of integer k that have a ``k{k}_distance`` column."""
    ks = []
    for col in merged_df.columns:
        m = re.fullmatch(r"k(\d+)_distance", str(col))
        if m:
            ks.append(int(m.group(1)))
    return sorted(ks)


def _as_dataset_pairs(datasets):
    """Normalize a {name: obj} dict or [(name, obj), ...] list to a list of pairs."""
    if isinstance(datasets, dict):
        return list(datasets.items())
    return list(datasets)


################################################################################
# Per-k signals


def compute_jaccard_vs_k(merged_df):
    """Summarize the jaccard distance across pairs for each k.

    For every ``k{k}_distance`` column, compute the mean, standard deviation and
    the 25/50/75% quantiles across genome pairs.

    Returns a DataFrame indexed by integer k with columns ``mean``, ``std``,
    ``q1``, ``median``, ``q3``.
    """
    ks = _distance_k_columns(merged_df)
    if not ks:
        raise ValueError("merged_df has no 'k{k}_distance' columns")

    cols = [f"k{k}_distance" for k in ks]
    sub = merged_df[cols].apply(pd.to_numeric, errors="coerce")

    out = pd.DataFrame(
        {
            "mean": sub.mean().to_numpy(dtype=float),
            "std": sub.std().to_numpy(dtype=float),
            "q1": sub.quantile(0.25).to_numpy(dtype=float),
            "median": sub.quantile(0.50).to_numpy(dtype=float),
            "q3": sub.quantile(0.75).to_numpy(dtype=float),
        },
        index=ks,
    )
    out.index.name = "k"
    return out


def compute_corr_vs_k(merged_df, method="spearman"):
    """Correlate each ``k{k}_distance`` column with ``tree_distance``.

    For each k, computes the (Spearman by default) correlation between the
    per-pair jaccard distance and the true tree distance.

    Returns a DataFrame indexed by integer k with columns ``corr``,
    ``abs_corr``, ``pval``.
    """
    from scipy.stats import pearsonr, spearmanr

    if "tree_distance" not in merged_df.columns:
        raise ValueError("merged_df must contain a 'tree_distance' column")
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")

    ks = _distance_k_columns(merged_df)
    if not ks:
        raise ValueError("merged_df has no 'k{k}_distance' columns")

    tree = pd.to_numeric(merged_df["tree_distance"], errors="coerce")
    corr_fn = spearmanr if method == "spearman" else pearsonr

    rows = {}
    for k in ks:
        dist = pd.to_numeric(merged_df[f"k{k}_distance"], errors="coerce")
        mask = dist.notna() & tree.notna()
        if mask.sum() < 2:
            corr, pval = np.nan, np.nan
        else:
            corr, pval = corr_fn(dist[mask], tree[mask])
        rows[k] = {"corr": corr, "abs_corr": abs(corr), "pval": pval}

    out = pd.DataFrame(rows).T
    out.index = out.index.astype(int)
    out.index.name = "k"
    return out


def best_k_from_corr(corr_df):
    """Single-best k = argmax of ``abs_corr`` (the strongest one-k correlation)."""
    return int(corr_df["abs_corr"].idxmax())


def find_stable_corr_range(corr_df, frac=0.99):
    """Contiguous k-interval around the peak where correlation stays high.

    Walks outward from the best k while ``abs_corr >= frac * max(abs_corr)``,
    stopping at the first k that drops below the threshold on each side.

    Returns ``(k_lo, k_hi)`` (inclusive). Degenerates to ``(best_k, best_k)``
    when no neighbours clear the threshold.
    """
    abs_corr = corr_df["abs_corr"].astype(float)
    ks = list(abs_corr.index)
    threshold = frac * np.nanmax(abs_corr.to_numpy())

    best_k = best_k_from_corr(corr_df)
    best_pos = ks.index(best_k)

    lo = best_pos
    while lo - 1 >= 0 and abs_corr.iloc[lo - 1] >= threshold:
        lo -= 1
    hi = best_pos
    while hi + 1 < len(ks) and abs_corr.iloc[hi + 1] >= threshold:
        hi += 1

    return int(ks[lo]), int(ks[hi])


def tightest_frac_for_width(corr_df, max_width=2, frac_grid=None):
    """Smallest ``frac`` whose stable range has width ``<= max_width``.

    Because ``find_stable_corr_range`` thresholds at ``frac * max(abs_corr)``,
    raising ``frac`` toward 1.0 tightens the threshold and the window width is
    monotonically non-increasing. Scanning an ascending ``frac`` grid, the first
    value meeting the target width is therefore the gentlest qualifying ``frac``.

    Returns ``(frac, k_lo, k_hi, width)``. If even the largest grid value cannot
    reach ``max_width`` (e.g. exact ties at the peak), falls back to that value.
    """
    if frac_grid is None:
        frac_grid = np.round(np.arange(0.90, 1.0001, 0.001), 4)

    for frac in frac_grid:
        k_lo, k_hi = find_stable_corr_range(corr_df, frac=float(frac))
        width = k_hi - k_lo + 1
        if width <= max_width:
            return float(frac), int(k_lo), int(k_hi), int(width)

    frac = float(frac_grid[-1])
    k_lo, k_hi = find_stable_corr_range(corr_df, frac=frac)
    return frac, int(k_lo), int(k_hi), int(k_hi - k_lo + 1)


def auto_tune_stability(merged_df_datasets, max_width=2, method="spearman",
                        plot=True):
    """Per-dataset auto-tune of ``frac`` to a target stable-window spread.

    For each dataset, find the smallest ``frac`` whose stable-correlation window
    is ``<= max_width`` wide (via ``tightest_frac_for_width``) and, when
    ``plot`` is set, render the jaccard/correlation figure at that tuned
    ``frac``.

    Returns a DataFrame indexed by dataset name with columns ``best_k``,
    ``tuned_frac``, ``stable_lo``, ``stable_hi``, ``stable_width`` and
    ``max_abs_corr``.
    """
    rows = {}
    for name, merged_df in _as_dataset_pairs(merged_df_datasets):
        corr = compute_corr_vs_k(merged_df, method=method)
        best_k = best_k_from_corr(corr)
        frac, k_lo, k_hi, width = tightest_frac_for_width(
            corr, max_width=max_width
        )
        rows[name] = {
            "best_k": best_k,
            "tuned_frac": frac,
            "stable_lo": k_lo,
            "stable_hi": k_hi,
            "stable_width": width,
            "max_abs_corr": float(corr["abs_corr"].max()),
        }
        if plot and plt is not None:
            plot_jaccard_stability(name, merged_df, frac=frac, method=method)
            plt.show()

    table = pd.DataFrame(rows).T
    table.index.name = "dataset"
    return table


################################################################################
# Ground-truth-free signal: jaccard-curve shape (derivatives + elbow)


def compute_jaccard_derivatives(merged_df, stat="mean", smooth=False,
                                smooth_window=5, smooth_polyorder=2,
                                method="gradient"):
    """1st/2nd derivatives of the (raw) jaccard ``stat`` curve vs k.

    Uses ``compute_jaccard_vs_k`` for the per-k curve and reuses
    ``ok_k_mer_card_explore._differentiate`` so the derivative conventions match
    the cardinality module. No ``-log10`` transform: derivatives are taken on the
    raw jaccard value in ``[0, 1]``.

    Parameters
    ----------
    stat : str
        Which jaccard column to differentiate (``mean``/``median``/``q1``/``q3``).
    method : {"gradient", "diff"}
        ``"gradient"`` (central differences, aligned to k) or ``"diff"``
        (forward differences; ``d1`` at midpoints, ``d2`` at interior k).
    smooth : bool
        Also return Savitzky-Golay-smoothed derivatives (``y_s``/``d1_s``/``d2_s``).

    Returns a dict with ``k``, ``y``, ``d1``, ``d2``, ``k_d1``, ``k_d2``.
    """
    jac = compute_jaccard_vs_k(merged_df)
    if stat not in jac.columns:
        raise KeyError(f"stat '{stat}' not in jaccard columns {list(jac.columns)}")

    k = np.asarray(jac.index, dtype=float)
    y = jac[stat].to_numpy(dtype=float)

    d1, d2, k_d1, k_d2 = ce._differentiate(y, k, method)
    out = {
        "k": k.astype(int),
        "y": y,
        "d1": d1,
        "d2": d2,
        "k_d1": k_d1,
        "k_d2": k_d2,
        "stat": stat,
        "method": method,
    }

    if smooth:
        y_s = ce._savgol(y, smooth_window, smooth_polyorder)
        d1_s, d2_s, _, _ = ce._differentiate(y_s, k, method)
        out.update({"y_s": y_s, "d1_s": d1_s, "d2_s": d2_s})

    return out


def estimate_k_jaccard_elbow(deriv):
    """k at the saturation->decline onset = argmin of the 2nd derivative.

    For the decreasing jaccard curve the most-concave bend (most negative d2)
    marks where the curve leaves the saturated ~1 plateau and starts dropping.
    """
    k_d2, d2 = deriv["k_d2"], deriv["d2"]
    return int(np.asarray(k_d2)[np.nanargmin(d2)])


def estimate_k_jaccard_elbow2(deriv):
    """k at the decline->floor knee = argmax of the 2nd derivative ("2nd elbow").

    The most-convex bend (most positive d2) marks where the falling jaccard
    curve flattens out toward its ~0 floor.
    """
    k_d2, d2 = deriv["k_d2"], deriv["d2"]
    return int(np.asarray(k_d2)[np.nanargmax(d2)])


def estimate_k_jaccard_settle(deriv, tol_frac=0.1):
    """Smallest k past the floor knee where the 2nd derivative settles at ~0.

    This is the "stable" point the jaccard curve stops varying: walking forward
    from the floor knee (``argmax`` d2), it returns the first k from which
    ``|d2|`` stays within ``tol_frac * max(|d2|)`` for all larger k (a flat
    plateau, not just a single zero crossing). Falls back to the last k if the
    curve never settles within tolerance.
    """
    k_d2 = np.asarray(deriv["k_d2"])
    d2 = np.asarray(deriv["d2"], dtype=float)
    if d2.size == 0:
        raise ValueError("no 2nd-derivative samples to assess settling")

    scale = np.nanmax(np.abs(d2))
    tol = tol_frac * scale if scale > 0 else 0.0
    peak = int(np.nanargmax(d2))

    abs_d2 = np.abs(d2)
    for i in range(peak, len(d2)):
        if np.all(abs_d2[i:] <= tol):
            return int(k_d2[i])
    return int(k_d2[-1])


################################################################################
# Ground-truth-free estimators (the 4 exploration options)


def _informative_mask(jac, lo=0.02, hi=0.98):
    """Boolean over k where the mean jaccard is neither saturated nor floored.

    Restricting argmax-based estimators to ``lo < mean < hi`` avoids the
    degenerate extremes where the curve sits at ~1 (std -> 0) or ~0.
    """
    mean = jac["mean"].to_numpy(dtype=float)
    mask = (mean > lo) & (mean < hi)
    if not mask.any():  # fall back to the whole range if everything is extreme
        mask = np.ones_like(mean, dtype=bool)
    return pd.Series(mask, index=jac.index)


# --- Option 1: discriminative spread -----------------------------------------


def compute_jaccard_spread(merged_df):
    """Across-pair spread of jaccard for each k.

    Returns a DataFrame indexed by k with columns ``iqr`` (``q3 - q1``),
    ``std`` and ``var`` (``std**2``). Larger spread = more resolving power
    between pairs.
    """
    jac = compute_jaccard_vs_k(merged_df)
    out = pd.DataFrame(
        {
            "iqr": (jac["q3"] - jac["q1"]).to_numpy(dtype=float),
            "std": jac["std"].to_numpy(dtype=float),
            "var": (jac["std"] ** 2).to_numpy(dtype=float),
        },
        index=jac.index,
    )
    out.index.name = "k"
    return out


def estimate_k_max_spread(merged_df, measure="iqr"):
    """k that maximizes the across-pair jaccard spread (``iqr``/``std``/``var``)."""
    spread = compute_jaccard_spread(merged_df)
    return int(spread[measure].idxmax())


# --- Option 2: rank stability across consecutive k ---------------------------


def compute_rank_stability(merged_df, method="spearman"):
    """Consecutive-k agreement of the per-pair distance ranking.

    For each adjacent pair (k, k+step) computes the Spearman correlation between
    the two distance vectors across genome pairs. High values mean the ranking
    of pairs is self-consistent between neighbouring k (a stability plateau that
    needs no ground truth). Constant columns (saturation/floor) yield ``NaN``.

    Returns a DataFrame indexed by the lower k of each pair with columns
    ``k_next`` and ``rank_corr``.
    """
    from scipy.stats import spearmanr

    ks = _distance_k_columns(merged_df)
    rows = {}
    for k_a, k_b in zip(ks[:-1], ks[1:]):
        a = pd.to_numeric(merged_df[f"k{k_a}_distance"], errors="coerce")
        b = pd.to_numeric(merged_df[f"k{k_b}_distance"], errors="coerce")
        mask = a.notna() & b.notna()
        if mask.sum() < 2 or a[mask].nunique() < 2 or b[mask].nunique() < 2:
            corr = np.nan
        else:
            corr, _ = spearmanr(a[mask], b[mask])
        rows[k_a] = {"k_next": k_b, "rank_corr": corr}

    out = pd.DataFrame(rows).T
    out.index = out.index.astype(int)
    out.index.name = "k"
    return out


def estimate_k_rank_stability(merged_df, method="spearman"):
    """k with the most self-consistent ranking vs its neighbour.

    Restricted to the informative region (mean jaccard in ``(0.02, 0.98)``) so
    the trivially-stable saturation/floor plateaus are excluded.
    """
    stab = compute_rank_stability(merged_df, method=method)
    jac = compute_jaccard_vs_k(merged_df)
    mask = _informative_mask(jac).reindex(stab.index).fillna(False)
    candidates = stab["rank_corr"][mask.to_numpy()]
    if candidates.dropna().empty:
        candidates = stab["rank_corr"]
    return int(candidates.idxmax())


# --- Option 3: signal-to-noise -----------------------------------------------


def compute_snr(merged_df):
    """Per-k signal-to-noise of the across-pair jaccard distribution.

    Returns a DataFrame indexed by k with ``snr`` (``mean/std``) and ``cv``
    (``std/mean``, the relative spread / resolving power).
    """
    jac = compute_jaccard_vs_k(merged_df)
    mean = jac["mean"].to_numpy(dtype=float)
    std = jac["std"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(std > 0, mean / std, np.nan)
        cv = np.where(mean > 0, std / mean, np.nan)
    out = pd.DataFrame({"snr": snr, "cv": cv}, index=jac.index)
    out.index.name = "k"
    return out


def estimate_k_max_snr(merged_df, measure="snr"):
    """k maximizing the chosen SNR measure within the informative region.

    ``measure="snr"`` (mean/std) follows the original suggestion; ``"cv"``
    (std/mean) maximizes relative spread. Both are restricted to
    ``0.02 < mean < 0.98`` so the std->0 saturation blow-up is excluded.
    """
    snr = compute_snr(merged_df)
    jac = compute_jaccard_vs_k(merged_df)
    mask = _informative_mask(jac).reindex(snr.index).fillna(False)
    candidates = snr[measure][mask.to_numpy()]
    if candidates.dropna().empty:
        candidates = snr[measure]
    return int(candidates.idxmax())


# --- Option 4: usable window then pick inside --------------------------------


def jaccard_usable_window(merged_df, stat="mean", method="gradient"):
    """Bracket k between the onset elbow and the floor knee of the jaccard curve.

    ``k_lo`` = argmin of the 2nd derivative (saturation->decline onset),
    ``k_hi`` = argmax of the 2nd derivative (decline->floor knee). Returned
    sorted so ``k_lo <= k_hi``.
    """
    deriv = compute_jaccard_derivatives(merged_df, stat=stat, method=method)
    k_d2 = np.asarray(deriv["k_d2"])
    d2 = deriv["d2"]
    onset = int(k_d2[np.nanargmin(d2)])
    floor = int(k_d2[np.nanargmax(d2)])
    return (onset, floor) if onset <= floor else (floor, onset)


def estimate_k_window_spread(merged_df, measure="iqr"):
    """k that maximizes across-pair spread *within* the usable window.

    Combines Option 4's bracket with Option 1's spread: restrict to
    ``[k_lo, k_hi]`` from ``jaccard_usable_window`` and pick the spread argmax
    inside it.
    """
    k_lo, k_hi = jaccard_usable_window(merged_df)
    spread = compute_jaccard_spread(merged_df)
    within = spread.loc[(spread.index >= k_lo) & (spread.index <= k_hi)]
    if within.empty:
        within = spread
    return int(within[measure].idxmax())


################################################################################
# Plotting


def plot_jaccard_stability(name, merged_df, ax=None, frac=0.99,
                           method="spearman"):
    """Jaccard (mean + IQR band) and one-k |corr| vs k for one dataset.

    Left y-axis: mean jaccard distance across pairs with a shaded 25-75%
    (IQR) band. Right y-axis: magnitude of the per-k correlation with tree
    distance. A red vertical line marks the single-best k (``argmax`` of
    ``abs_corr``) and a shaded vertical span marks the stable-correlation range
    (where ``abs_corr >= frac * max``).

    Returns the Matplotlib ``Figure``.
    """
    if plt is None:
        raise RuntimeError("matplotlib is not available")

    jac = compute_jaccard_vs_k(merged_df)
    corr = compute_corr_vs_k(merged_df, method=method)
    best_k = best_k_from_corr(corr)
    k_lo, k_hi = find_stable_corr_range(corr, frac=frac)

    k = np.asarray(jac.index, dtype=int)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure

    band = ax.fill_between(k, jac["q1"], jac["q3"], color="C0", alpha=0.2,
                           label="jaccard IQR (25-75%)")
    line_mean, = ax.plot(k, jac["mean"], color="C0", marker="o", ms=3, lw=1.5,
                         label="jaccard mean")
    ax.set_xlabel("k-mer length")
    ax.set_ylabel("jaccard distance")
    ax.set_xticks(k)
    ax.set_xticklabels(k, fontsize=7, rotation=90)
    ax.grid(True, ls=":", alpha=0.4)

    ax2 = ax.twinx()
    line_corr, = ax2.plot(corr.index, corr["abs_corr"], color="C2", marker="s",
                          ms=3, lw=1.5, label=f"|{method} corr| (one k)")
    ax2.set_ylabel("|correlation with tree distance|")

    span = ax.axvspan(k_lo, k_hi, color="C1", alpha=0.12,
                      label=f"stable range k={k_lo}-{k_hi} (>= {frac:g} of max)")
    vline = ax.axvline(best_k, color="C3", ls="-", lw=1.5,
                       label=f"best k={best_k}")

    handles = [line_mean, band, line_corr, span, vline]
    ax.legend(handles, [h.get_label() for h in handles], loc="best",
              frameon=False, fontsize=8)
    ax.set_title(f"{name}: jaccard & one-k |corr| vs k")
    fig.tight_layout()
    return fig


def plot_all_stability(merged_df_datasets, frac=0.99, method="spearman"):
    """Plot every dataset and return a per-dataset stability summary.

    ``merged_df_datasets`` is a list of ``(name, merged_df)`` (or a dict). One
    figure is produced per dataset via ``plot_jaccard_stability``.

    Returns a DataFrame indexed by dataset name with columns ``best_k``,
    ``stable_lo``, ``stable_hi``, ``stable_width`` and ``max_abs_corr`` so the
    "is there a stable range?" question is answered numerically too.
    """
    rows = {}
    for name, merged_df in _as_dataset_pairs(merged_df_datasets):
        corr = compute_corr_vs_k(merged_df, method=method)
        best_k = best_k_from_corr(corr)
        k_lo, k_hi = find_stable_corr_range(corr, frac=frac)
        rows[name] = {
            "best_k": best_k,
            "stable_lo": k_lo,
            "stable_hi": k_hi,
            "stable_width": k_hi - k_lo + 1,
            "max_abs_corr": float(corr["abs_corr"].max()),
        }
        if plt is not None:
            plot_jaccard_stability(name, merged_df, frac=frac, method=method)
            plt.show()

    table = pd.DataFrame(rows).T
    table.index.name = "dataset"
    return table


def plot_jaccard_derivatives(name, merged_df, stat="mean", smooth=False,
                             method="gradient", ax=None):
    """3-panel plot of the jaccard ``stat`` curve and its 1st/2nd derivatives.

    Marks the saturation->decline elbow (``estimate_k_jaccard_elbow``, argmin
    d2) with a solid red line, the decline->floor "2nd elbow"
    (``estimate_k_jaccard_elbow2``, argmax d2) with a dashed purple line, and
    the settle point (``estimate_k_jaccard_settle``, where d2 returns to ~0 and
    stays) with a dotted green line on every panel. Returns the Matplotlib
    ``Figure``.
    """
    if plt is None:
        raise RuntimeError("matplotlib is not available")

    deriv = compute_jaccard_derivatives(merged_df, stat=stat, smooth=smooth,
                                        method=method)
    elbow_k = estimate_k_jaccard_elbow(deriv)
    elbow2_k = estimate_k_jaccard_elbow2(deriv)
    settle_k = estimate_k_jaccard_settle(deriv)
    k = deriv["k"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    panels = [
        (f"jaccard {stat}", "y", "y_s", "k"),
        ("d/dk  (1st derivative)", "d1", "d1_s", "k_d1"),
        ("d2/dk2  (2nd derivative)", "d2", "d2_s", "k_d2"),
    ]
    for ax_i, (title, raw_key, smooth_key, kx) in zip(axes, panels):
        ax_i.plot(deriv[kx], deriv[raw_key], marker="o", markersize=3, lw=1.5,
                  color="C0", label="raw")
        if smooth and smooth_key in deriv:
            ax_i.plot(deriv["k"], deriv[smooth_key], lw=1.5, color="C1",
                      label="smoothed")
        if raw_key == "d2":
            ax_i.axhline(0.0, color="grey", ls=":", lw=1)
        ax_i.axvline(elbow_k, color="C3", ls="-", lw=1.5,
                     label=f"elbow (argmin d2) k={elbow_k}")
        ax_i.axvline(elbow2_k, color="C4", ls="--", lw=1.5,
                     label=f"2nd elbow (argmax d2) k={elbow2_k}")
        ax_i.axvline(settle_k, color="C2", ls=":", lw=1.5,
                     label=f"settle (d2~0 stable) k={settle_k}")
        ax_i.set_ylabel(title)
        ax_i.grid(True, ls=":", alpha=0.4)
        ax_i.legend(loc="best", frameon=False, fontsize=8)

    axes[-1].set_xlabel("k-mer length")
    axes[-1].set_xticks(k)
    axes[-1].set_xticklabels(k, fontsize=7, rotation=90)
    axes[0].set_title(f"{name}: jaccard {stat} dynamics, elbows & settle")
    fig.tight_layout()
    return fig


def plot_all_jaccard_derivatives(merged_df_datasets, stat="mean", smooth=False,
                                 method="gradient", plot=True):
    """Ground-truth-free jaccard-elbow selection across every dataset.

    For each dataset, find the jaccard-curve elbow (``argmin`` of the 2nd
    derivative), the "2nd elbow" (``argmax`` of the 2nd derivative, the
    decline->floor knee) and the settle point (where d2 returns to ~0 and stays
    stable), and when ``plot`` is set render the 3-panel derivative figure.

    Returns a DataFrame indexed by dataset name with ``elbow_k``,
    ``jaccard_at_elbow``, ``elbow2_k``, ``jaccard_at_elbow2``, ``settle_k``,
    ``jaccard_at_settle`` and, for context, ``corr_best_k`` (the ground-truth
    argmax of |corr|) so the no-ground-truth estimates can be compared against
    it.
    """
    rows = {}
    for name, merged_df in _as_dataset_pairs(merged_df_datasets):
        deriv = compute_jaccard_derivatives(merged_df, stat=stat, smooth=smooth,
                                            method=method)
        elbow_k = estimate_k_jaccard_elbow(deriv)
        elbow2_k = estimate_k_jaccard_elbow2(deriv)
        settle_k = estimate_k_jaccard_settle(deriv)
        k_list = list(deriv["k"])
        jac_at_elbow = float(deriv["y"][k_list.index(elbow_k)])
        jac_at_elbow2 = float(deriv["y"][k_list.index(elbow2_k)])
        jac_at_settle = float(deriv["y"][k_list.index(settle_k)])
        corr = compute_corr_vs_k(merged_df)
        rows[name] = {
            "elbow_k": elbow_k,
            "jaccard_at_elbow": jac_at_elbow,
            "elbow2_k": elbow2_k,
            "jaccard_at_elbow2": jac_at_elbow2,
            "settle_k": settle_k,
            "jaccard_at_settle": jac_at_settle,
            "corr_best_k": best_k_from_corr(corr),
        }
        if plot and plt is not None:
            plot_jaccard_derivatives(name, merged_df, stat=stat, smooth=smooth,
                                     method=method)
            plt.show()

    table = pd.DataFrame(rows).T
    table.index.name = "dataset"
    return table


def _unsupervised_estimates(merged_df):
    """All ground-truth-free k estimates for one dataset, plus ``corr_best_k``."""
    deriv = compute_jaccard_derivatives(merged_df)
    k_lo, k_hi = jaccard_usable_window(merged_df)
    corr = compute_corr_vs_k(merged_df)
    return {
        "onset_elbow": estimate_k_jaccard_elbow(deriv),
        "floor_knee": int(np.asarray(deriv["k_d2"])[np.nanargmax(deriv["d2"])]),
        "max_iqr": estimate_k_max_spread(merged_df, measure="iqr"),
        "max_var": estimate_k_max_spread(merged_df, measure="var"),
        "rank_stable": estimate_k_rank_stability(merged_df),
        "max_snr": estimate_k_max_snr(merged_df, measure="snr"),
        "max_cv": estimate_k_max_snr(merged_df, measure="cv"),
        "window": f"{k_lo}-{k_hi}",
        "window_spread": estimate_k_window_spread(merged_df, measure="iqr"),
        "corr_best_k": best_k_from_corr(corr),
    }


def plot_unsupervised_signals(name, merged_df):
    """4-panel view of every ground-truth-free signal vs k for one dataset.

    Panels: (A) jaccard mean + IQR band with the usable window shaded; (B)
    across-pair spread (IQR/std); (C) consecutive-k rank stability; (D) SNR and
    CV. Each estimator's chosen k is drawn as a vertical line, and the
    ground-truth ``corr_best_k`` is shown (dashed black) for reference.

    Returns the Matplotlib ``Figure``.
    """
    if plt is None:
        raise RuntimeError("matplotlib is not available")

    jac = compute_jaccard_vs_k(merged_df)
    spread = compute_jaccard_spread(merged_df)
    stab = compute_rank_stability(merged_df)
    snr = compute_snr(merged_df)
    est = _unsupervised_estimates(merged_df)
    k_lo, k_hi = jaccard_usable_window(merged_df)
    k = np.asarray(jac.index, dtype=int)
    best = est["corr_best_k"]

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)

    ax = axes[0]
    ax.fill_between(k, jac["q1"], jac["q3"], color="C0", alpha=0.2,
                    label="IQR (25-75%)")
    ax.plot(k, jac["mean"], color="C0", marker="o", ms=3, lw=1.5, label="mean")
    ax.axvspan(k_lo, k_hi, color="C1", alpha=0.12,
               label=f"usable window {k_lo}-{k_hi}")
    ax.axvline(est["onset_elbow"], color="C1", ls="-", lw=1,
               label=f"onset elbow={est['onset_elbow']}")
    ax.set_ylabel("jaccard mean")

    ax = axes[1]
    ax.plot(spread.index, spread["iqr"], color="C0", marker="o", ms=3,
            label="IQR")
    ax.plot(spread.index, spread["std"], color="C4", marker="^", ms=3,
            label="std")
    ax.axvline(est["max_iqr"], color="C0", ls="-", lw=1,
               label=f"max IQR k={est['max_iqr']}")
    ax.set_ylabel("across-pair spread")

    ax = axes[2]
    ax.plot(stab.index, stab["rank_corr"], color="C2", marker="s", ms=3,
            label="consecutive-k rank corr")
    ax.axvline(est["rank_stable"], color="C2", ls="-", lw=1,
               label=f"rank-stable k={est['rank_stable']}")
    ax.set_ylabel("rank stability")

    ax = axes[3]
    ax.plot(snr.index, snr["snr"], color="C5", marker="o", ms=3, label="snr=mean/std")
    ax.axvline(est["max_snr"], color="C5", ls="-", lw=1,
               label=f"max snr k={est['max_snr']}")
    ax2 = ax.twinx()
    ax2.plot(snr.index, snr["cv"], color="C6", marker="x", ms=3, label="cv=std/mean")
    ax2.axvline(est["max_cv"], color="C6", ls=":", lw=1,
                label=f"max cv k={est['max_cv']}")
    ax2.set_ylabel("cv")
    ax.set_ylabel("snr")

    for ax_i in axes:
        ax_i.axvline(best, color="k", ls="--", lw=1.5, alpha=0.7,
                     label=f"corr best k={best}")
        ax_i.grid(True, ls=":", alpha=0.4)
        ax_i.legend(loc="best", frameon=False, fontsize=7)

    axes[-1].set_xlabel("k-mer length")
    axes[-1].set_xticks(k)
    axes[-1].set_xticklabels(k, fontsize=7, rotation=90)
    axes[0].set_title(f"{name}: ground-truth-free k signals")
    fig.tight_layout()
    return fig


def compare_unsupervised_estimators(merged_df_datasets, plot=True):
    """Compare all 4 ground-truth-free estimators against ``corr_best_k``.

    For each dataset computes every estimator's k (Option 1 spread, Option 2
    rank stability, Option 3 SNR/CV, Option 4 usable-window spread, plus the
    onset elbow and floor knee) and, when ``plot`` is set, renders the 4-panel
    signal figure.

    Returns a DataFrame indexed by dataset with one column per estimator, the
    ground-truth ``corr_best_k``, and ``abs_err_*`` columns giving each
    estimator's absolute distance from ``corr_best_k`` (with a final ``MAE`` row
    of mean absolute error per estimator).
    """
    rows = {}
    for name, merged_df in _as_dataset_pairs(merged_df_datasets):
        rows[name] = _unsupervised_estimates(merged_df)
        if plot and plt is not None:
            plot_unsupervised_signals(name, merged_df)
            plt.show()

    table = pd.DataFrame(rows).T
    table.index.name = "dataset"

    est_cols = ["onset_elbow", "floor_knee", "max_iqr", "max_var",
                "rank_stable", "max_snr", "max_cv", "window_spread"]
    best = pd.to_numeric(table["corr_best_k"])
    for col in est_cols:
        table[f"abs_err_{col}"] = (pd.to_numeric(table[col]) - best).abs()

    mae = {c: np.nan for c in table.columns}
    for col in est_cols:
        mae[f"abs_err_{col}"] = table[f"abs_err_{col}"].mean()
    table.loc["MAE"] = mae
    return table
