"""
ok_k_mer_selection
==================

A derivative-based k-mer length selection scheme and its evaluation.

The central idea: the 2nd derivative of the -log10 normalized union
cardinality ``-log10(C(k) / 4^k)`` peaks at the saturation -> linear transition
("elbow"). We take that argmax as ``k1`` and add geometric multiples ``2*k1``
and ``3*k1`` as additional k-mer lengths. Selected k-mer lengths are then
evaluated by averaging the per-pair distance ranks (evenly weighted) and
correlating that average against the true tree distance with Spearman's rho.

Two variants are provided:

* **global** - one ``k1`` per dataset, derived from the dataset-wide all-pairs
  ``average`` cardinality curve (the ``<Dataset>_kmer_stats.tsv`` ``average``
  row, as loaded by ``ok_k_mer_card_explore``).
* **pairwise** - each genome pair gets its own ``k1`` from its own cardinality
  curve, so different pairs are evaluated at different (multiplied) k.

Data conventions (matching ``explore_card.ipynb``):

* ``stats_df``: columns = integer k, rows = summary stats; values are the
  normalized ``C(k) / 4^k``.
* ``merged_df``: one row per genome pair with columns ``k{k}_distance`` and
  ``tree_distance`` plus a ``genome_pair`` column. The distance metric is
  similarity-like, so it correlates *negatively* with tree distance and the
  grid search prefers the most-negative Spearman; we therefore report the raw
  correlation and its magnitude (``abs_corr``).
* ``union_card_df``: one row per genome pair with ``k{k}_cardinality`` columns
  (normalized ``C(k) / 4^k``) and a ``genome_pair`` column.

Import in the notebook with ``import ok_k_mer_selection``.
"""

import re
import warnings

import numpy as np
import pandas as pd

import ok_k_mer_card_explore as ce


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


def _cardinality_k_columns(union_card_df):
    """Return the sorted list of integer k that have a ``k{k}_cardinality`` column."""
    ks = []
    for col in union_card_df.columns:
        m = re.fullmatch(r"k(\d+)_cardinality", str(col))
        if m:
            ks.append(int(m.group(1)))
    return sorted(ks)


def _clamp_k(k, available, label=""):
    """Clamp ``k`` to the nearest value present in ``available`` (sorted list).

    Emits a warning when clamping is required so silently-shifted k are visible.
    """
    if k in available:
        return k
    nearest = min(available, key=lambda a: abs(a - k))
    warnings.warn(
        f"requested k={k}{(' for ' + label) if label else ''} is not available; "
        f"clamping to nearest available k={nearest}",
        stacklevel=2,
    )
    return nearest


################################################################################
# Global scheme (one k1 per dataset)


def select_global_ks(stats_df, stat="average", n_kmers=3, multipliers=(1, 2, 3)):
    """Select k-mer lengths from the dataset-wide cardinality dynamics.

    ``k1`` is the argmax of the 2nd derivative of ``-log10(C(k) / 4^k)`` (the
    transition elbow), and the remaining lengths are geometric multiples of it.

    Returns the list ``[m * k1 for m in multipliers][:n_kmers]``.
    """
    deriv = ce.compute_logcard_derivatives(stats_df, stat=stat)
    k1 = ce.estimate_k_elbow(deriv)
    ks = [int(m * k1) for m in multipliers]
    return ks[:n_kmers]


def evaluate_weighted_rank_corr(merged_df, ks, weights=None):
    """Evenly-weighted average of per-pair distance ranks vs tree distance.

    For each k in ``ks`` the column ``k{k}_distance`` is ranked across pairs;
    the (weighted) average rank is correlated with ``tree_distance`` using
    Spearman's rho. ``weights`` defaults to equal weights summing to 1.

    Returns ``(corr, pval)``.
    """
    from scipy.stats import spearmanr

    available = _distance_k_columns(merged_df)
    ks = [_clamp_k(int(k), available, label="distance") for k in ks]

    if weights is None:
        weights = [1.0 / len(ks)] * len(ks)
    if len(weights) != len(ks):
        raise ValueError(f"weights {weights} length != ks {ks} length")

    avg_rank = None
    for w, k in zip(weights, ks):
        rank = merged_df[f"k{k}_distance"].rank()
        avg_rank = w * rank if avg_rank is None else avg_rank + w * rank

    corr, pval = spearmanr(avg_rank, merged_df["tree_distance"])
    return corr, pval


def run_global_selection(stats_df, merged_df, stat="average",
                         w_2kmer=(0.33, 0.66)):
    """Run the global 2-kmer and 3-kmer selection + evaluation for one dataset.

    The 2-kmer average uses ``w_2kmer`` (default ``(0.33, 0.66)`` -> k1 weighted
    0.33, k2 weighted 0.66); the 3-kmer average is evenly weighted.

    Returns a dict with the selected k, the Spearman correlation (and p-value)
    for the 2-kmer and 3-kmer averages, and the correlation magnitudes.
    """
    k1, k2, k3 = select_global_ks(stats_df, stat=stat, n_kmers=3)

    corr2, pval2 = evaluate_weighted_rank_corr(merged_df, [k1, k2],
                                               weights=list(w_2kmer))
    corr3, pval3 = evaluate_weighted_rank_corr(merged_df, [k1, k2, k3])

    return {
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "ks_2kmer": (k1, k2),
        "ks_3kmer": (k1, k2, k3),
        "corr_2kmer": corr2,
        "pval_2kmer": pval2,
        "abs_corr_2kmer": abs(corr2),
        "corr_3kmer": corr3,
        "pval_3kmer": pval3,
        "abs_corr_3kmer": abs(corr3),
    }


################################################################################
# Pairwise scheme (one k1 per genome pair)


def pairwise_elbow_k(union_card_df):
    """Per-pair argmax of the 2nd derivative of ``-log10(C(k) / 4^k)``.

    For each genome pair (row), the normalized cardinality curve across k is
    -log10 transformed and differentiated twice with ``np.gradient`` (central
    differences, aligned to k). The k at the maximum 2nd derivative is the
    pair's ``k1``.

    Returns a Series of integer k indexed by ``genome_pair``.
    """
    ks = _cardinality_k_columns(union_card_df)
    if len(ks) < 3:
        raise ValueError(
            f"need >=3 cardinality columns to take a 2nd derivative, found {ks}"
        )
    k_arr = np.asarray(ks, dtype=float)
    cols = [f"k{k}_cardinality" for k in ks]

    card = union_card_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = -np.log10(card)

    # 2nd derivative per row (axis=1), aligned to k.
    d1 = np.gradient(y, k_arr, axis=1)
    d2 = np.gradient(d1, k_arr, axis=1)

    elbow_idx = np.nanargmax(d2, axis=1)
    elbow_k = k_arr[elbow_idx].astype(int)

    index = (
        union_card_df["genome_pair"]
        if "genome_pair" in union_card_df.columns
        else union_card_df.index
    )
    return pd.Series(elbow_k, index=np.asarray(index), name="k1")


def evaluate_pairwise_weighted_rank_corr(union_card_df, merged_df, multipliers,
                                         weights=None):
    """Evaluate the pairwise scheme for a given set of k multipliers.

    Each pair selects its own ``k1`` (from ``pairwise_elbow_k``); for every
    multiplier ``m`` the pair contributes its distance at ``k = m * k1`` (clamped
    into the available ``k{k}_distance`` columns). The per-multiplier distance
    vectors are ranked across pairs, weighted-averaged (``weights`` defaults to
    equal weights), and correlated with the true tree distance via Spearman's
    rho. ``weights`` applies to every individual pairwise comparison.

    Returns ``(corr, pval)``.
    """
    from scipy.stats import spearmanr

    if "genome_pair" not in merged_df.columns:
        raise ValueError("merged_df must contain a 'genome_pair' column")

    available = _distance_k_columns(merged_df)
    eval_df = merged_df.set_index("genome_pair")

    k1_by_pair = pairwise_elbow_k(union_card_df)
    # Align to the pairs present in the evaluation frame.
    k1_by_pair = k1_by_pair.reindex(eval_df.index)
    if k1_by_pair.isna().any():
        missing = int(k1_by_pair.isna().sum())
        raise ValueError(
            f"{missing} pair(s) in merged_df have no cardinality-derived k1; "
            "union_card_df and merged_df pairs are misaligned"
        )
    k1_by_pair = k1_by_pair.astype(int)

    if weights is None:
        weights = [1.0 / len(multipliers)] * len(multipliers)
    if len(weights) != len(multipliers):
        raise ValueError(
            f"weights {weights} length != multipliers {multipliers} length"
        )

    avg_rank = None
    for w, m in zip(weights, multipliers):
        selected_k = k1_by_pair.apply(
            lambda k1: _clamp_k(int(m * k1), available, label="distance")
        )
        # Gather the distance value at each pair's selected k.
        values = pd.Series(index=eval_df.index, dtype=float)
        for k, pairs in selected_k.groupby(selected_k):
            values.loc[pairs.index] = eval_df.loc[pairs.index, f"k{int(k)}_distance"]
        rank = values.rank()
        avg_rank = w * rank if avg_rank is None else avg_rank + w * rank

    corr, pval = spearmanr(avg_rank, eval_df["tree_distance"])
    return corr, pval


def run_pairwise_selection(union_card_df, merged_df, w_2kmer=(0.33, 0.66)):
    """Run the pairwise 2-kmer and 3-kmer selection + evaluation for one dataset.

    The 2-kmer average uses ``w_2kmer`` (default ``(0.33, 0.66)`` -> k1 weighted
    0.33, k2 = 2*k1 weighted 0.66) on each individual pairwise comparison; the
    3-kmer average is evenly weighted.

    Returns a dict with the Spearman correlation (and p-value / magnitude) for
    the 2-kmer and 3-kmer averages, plus the median/mean of the per-pair k1.
    """
    k1_by_pair = pairwise_elbow_k(union_card_df)

    corr2, pval2 = evaluate_pairwise_weighted_rank_corr(
        union_card_df, merged_df, multipliers=(1, 2), weights=list(w_2kmer)
    )
    corr3, pval3 = evaluate_pairwise_weighted_rank_corr(
        union_card_df, merged_df, multipliers=(1, 2, 3)
    )

    return {
        "k1_median": float(np.median(k1_by_pair.to_numpy())),
        "k1_mean": float(np.mean(k1_by_pair.to_numpy())),
        "corr_2kmer": corr2,
        "pval_2kmer": pval2,
        "abs_corr_2kmer": abs(corr2),
        "corr_3kmer": corr3,
        "pval_3kmer": pval3,
        "abs_corr_3kmer": abs(corr3),
    }


################################################################################
# Aggregators across all datasets


def _as_dataset_pairs(datasets):
    """Normalize a {name: obj} dict or [(name, obj), ...] list to a list of pairs."""
    if isinstance(datasets, dict):
        return list(datasets.items())
    return list(datasets)


def run_global_all(card_stats, merged_df_datasets, stat="average"):
    """Run the global scheme across every dataset.

    ``card_stats`` is ``{name: stats_df}`` and ``merged_df_datasets`` is a list
    of ``(name, merged_df)`` (or a dict). Datasets are matched by name.

    Returns a DataFrame indexed by dataset name.
    """
    stats_by_name = dict(_as_dataset_pairs(card_stats))
    merged_by_name = dict(_as_dataset_pairs(merged_df_datasets))

    rows = {}
    for name, stats_df in stats_by_name.items():
        if name not in merged_by_name:
            warnings.warn(f"no merged_df for dataset {name!r}; skipping", stacklevel=2)
            continue
        rows[name] = run_global_selection(stats_df, merged_by_name[name], stat=stat)

    table = pd.DataFrame(rows).T
    table.index.name = "dataset"
    return table


def run_pairwise_all(union_cardinality_datasets, merged_df_datasets):
    """Run the pairwise scheme across every dataset.

    ``union_cardinality_datasets`` is a list of ``(name, union_card_df)`` (or a
    dict) and ``merged_df_datasets`` a list of ``(name, merged_df)`` (or a dict).
    Datasets are matched by name.

    Returns a DataFrame indexed by dataset name.
    """
    card_by_name = dict(_as_dataset_pairs(union_cardinality_datasets))
    merged_by_name = dict(_as_dataset_pairs(merged_df_datasets))

    rows = {}
    for name, union_card_df in card_by_name.items():
        if name not in merged_by_name:
            warnings.warn(f"no merged_df for dataset {name!r}; skipping", stacklevel=2)
            continue
        rows[name] = run_pairwise_selection(union_card_df, merged_by_name[name])

    table = pd.DataFrame(rows).T
    table.index.name = "dataset"
    return table
