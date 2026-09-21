"""Helper functions for plotting phylogenetic trees with ete3, optionally
annotating each internal node with its best k-mer / tree-distance correlation
and the two-k weighted average that achieves it.

This mirrors the prototype logic in `II_IO_trees.ipynb` (cells 19-29), but is
organized into reusable functions so the same plots can be produced for any
dataset.

Data expectations (see `ok_k_mer.build_merged_df`):
  - Tree leaf `.name` equals the genome id (e.g. ``GCA_030146285.1``) and
    matches `genome_1` / `genome_2` in `merged_df`.
  - `merged_df` has a `tree_distance` column and one `k{K}_distance` column per
    k-mer length, plus a `genome_pair` column equal to
    ``'_'.join(sorted([genome_1, genome_2]))``.
"""

import os
import pickle

# ete3 needs a Qt platform; on a headless cluster use the offscreen backend.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from scipy.stats import spearmanr, kendalltau

from ete3 import Tree, TreeStyle, NodeStyle, faces

try:
    from IPython.display import Image, display as _ipy_display
    _HAVE_IPYTHON = True
except Exception:  # pragma: no cover - IPython not always available
    _HAVE_IPYTHON = False


################################################################################
# Tree loading / colors

def load_named_tree(tree_path):
    """Load a Newick tree and give every internal node a unique name.

    Internal (non-leaf) nodes are renamed ``internal_node_{i}`` so they can be
    used as dictionary keys for per-node annotations.

    Args:
        tree_path: Path to the ``.nwk`` file.

    Returns:
        ete3.Tree
    """
    tree = Tree(tree_path, format=1)
    i = 0
    for node in tree.traverse():
        if not node.is_leaf():
            node.name = f"internal_node_{i}"
            i += 1
    return tree


def make_k_color(k_low=6, k_high=40, cmap="viridis"):
    """Map each k value in ``[k_low, k_high]`` to a hex color via a colormap.

    Args:
        k_low: Lowest k value (inclusive).
        k_high: Highest k value (inclusive).
        cmap: Name of a matplotlib colormap (default ``viridis``).

    Returns:
        dict: ``{k: '#rrggbb'}``.
    """
    cmap_obj = plt.cm.get_cmap(cmap)
    span = max(k_high - k_low, 1)
    return {k: mcolors.rgb2hex(cmap_obj((k - k_low) / span))
            for k in range(k_low, k_high + 1)}


################################################################################
# Rendering helpers

def _basic_tree_style(scale=2000, show_leaf_name=True, mode="r"):
    """Build a basic rectangular TreeStyle."""
    ts = TreeStyle()
    ts.show_leaf_name = show_leaf_name
    ts.mode = mode                  # 'r' rectangular, 'c' circular
    ts.scale = scale                # scale branch lengths
    ts.branch_vertical_margin = 10  # spacing between branches
    ts.show_scale = True
    return ts


def _render(tree, outpath, tree_style, width=1500, display=True, dpi=None):
    """Render `tree` to `outpath` and optionally show it inline."""
    if dpi is None:
        tree.render(outpath, tree_style=tree_style, w=width)
    else:
        tree.render(outpath, tree_style=tree_style, w=width, dpi=dpi)
    if display and _HAVE_IPYTHON and str(outpath).lower().endswith(".png"):
        _ipy_display(Image(filename=outpath))
    return outpath


def _corr_colorbar_face(cmap=plt.cm.YlOrRd, vmin=0.0, vmax=1.0, tmp_path="colorbar_corr.png"):
    """Build an ImgFace colorbar for correlation values."""
    fig, ax = plt.subplots(figsize=(12, 2))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=ax, orientation="horizontal",
                      ticks=np.round(np.arange(vmin, vmax + 0.01, 0.1), 1))
    cb.set_label("Correlation")  # correlations are negative; plot -corr
    ax.set_visible(False)
    plt.savefig(tmp_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    face = faces.ImgFace(tmp_path)
    face.margin_top = 100
    face.margin_bottom = 100
    face.margin_right = 150  # space between the Correlation and K-mer legends
    return face


def _k_colorbar_png(k_color, tmp_path="colorbar_k.png"):
    """Save a horizontal k colorbar (tightly cropped) to `tmp_path`."""
    fig, ax = plt.subplots(figsize=(12, 2))
    ks = sorted(k_color.keys())
    colors = [k_color[k] for k in ks]
    bounds = np.arange(len(colors) + 1)
    cmap = plt.cm.colors.ListedColormap(colors)
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=ax, orientation="horizontal", ticks=bounds[:-1] + 0.5)
    cb.set_label("K-mer length")
    cb.ax.minorticks_off()  # keep only the integer (per-k) ticks, no sub-ticks
    # Label every other tick so the numbers don't overlap.
    cb.set_ticklabels([str(k) if i % 2 == 0 else "" for i, k in enumerate(ks)])
    ax.set_visible(False)
    plt.savefig(tmp_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return tmp_path


def _k_colorbar_face(k_color, tmp_path="colorbar_k.png"):
    """Build an ImgFace colorbar for k values from a `k_color` dict."""
    face = faces.ImgFace(_k_colorbar_png(k_color, tmp_path))
    face.margin_top = 100
    face.margin_bottom = 100
    return face


def _paste_centered_legend(image_path, legend_path, target_frac=0.4, pad=30,
                           bg=(255, 255, 255, 255), dpi=None):
    """Paste `legend_path` horizontally centered under the tree in `image_path`.

    ete3's built-in legend is corner-anchored and rescaled, so instead we
    composite the colorbar onto the rendered PNG for reliable centering/size.
    ``target_frac`` sets the legend width as a fraction of the image width.
    """
    from PIL import Image as _PILImage
    base = _PILImage.open(image_path).convert("RGBA")
    leg = _PILImage.open(legend_path).convert("RGBA")
    target_w = max(1, int(base.width * target_frac))
    scale = target_w / leg.width
    leg = leg.resize((target_w, max(1, int(leg.height * scale))),
                     _PILImage.LANCZOS)
    canvas = _PILImage.new("RGBA", (base.width, base.height + pad + leg.height), bg)
    canvas.paste(base, (0, 0), base)
    canvas.paste(leg, ((base.width - leg.width) // 2, base.height + pad), leg)
    save_kwargs = {} if dpi is None else {"dpi": (dpi, dpi)}
    canvas.convert("RGB").save(image_path, **save_kwargs)
    return image_path


################################################################################
# Plain tree plot

def plot_tree(tree_path, name=None, outpath=None, width=1500, scale=2000,
              show_leaf_name=True, display=True):
    """Render a tree with no annotations.

    Args:
        tree_path: Path to the ``.nwk`` file.
        name: Optional dataset name; used for the default output filename.
        outpath: Output image path (defaults to ``{name or 'tree'}_tree.png``).
        width: Render width in pixels.
        scale: Branch-length scale.
        show_leaf_name: Whether to show leaf labels.
        display: Show the rendered image inline (in a notebook).

    Returns:
        ete3.Tree: the (named) loaded tree.
    """
    tree = load_named_tree(tree_path)
    if outpath is None:
        outpath = f"{name or 'tree'}_tree.png"
    ts = _basic_tree_style(scale=scale, show_leaf_name=show_leaf_name)
    _render(tree, outpath, ts, width=width, display=display)
    return tree


################################################################################
# Per-node correlation / weighting

def _fast_rank(values):
    """Dense competition-free ranks via argsort (ties broken by position).

    Much faster than `scipy.stats.rankdata`; ties are negligible for the
    continuous weighted rank-sums swept here.
    """
    order = values.argsort()
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _sweep_best_corr(sub_df, k_low=6, k_high=40, w_step=0.05):
    """Find the globally best single-k and two-k weighted correlation.

    Sweeps every single k in ``[k_low, k_high]`` and **every** pair of ks across
    **every** weight in ``(0, 1)`` (step `w_step`), returning the most-negative
    (Spearman) correlation against ``tree_distance``. This is an exhaustive
    global search, not a search restricted to single-k local minima.

    Args:
        sub_df: Rows of `merged_df` (one per genome pair) with a `tree_distance`
            column and `k{K}_distance` columns.
        k_low, k_high: Inclusive k-sweep range.
        w_step: Step for the weight search in ``(0, 1)``.

    Returns:
        tuple: ``(local_minima, weighted_minima, best_corr)``.
            - ``local_minima``: list of single-k ``(k, corr)`` sorted best (most
              negative) first (reported for reference).
            - ``weighted_minima``: the globally best ``(k1, k2, w, corr)`` over
              all k-pairs and weights, or all-``None`` if none.
            - ``best_corr``: the better of the best single-k and the best
              weighted pair; ``None``, a ``(k, corr)`` tuple, or a
              ``(k1, k2, w, corr)`` tuple.

    Note:
        The exhaustive pair sweep is ``O(K^2 * W)`` correlations per call (vs.
        ``O(W)`` for a single pair), so this is markedly slower on large k ranges
        / many genome pairs. Increase `w_step` to trade resolution for speed.
    """
    if len(sub_df) < 3:  # too few comparisons to be meaningful
        return [], (None, None, None, None), None

    tree_dist = sub_df["tree_distance"]
    ks = [k for k in range(k_low, k_high + 1)
          if f"k{k}_distance" in sub_df.columns]

    # Precompute each k's rank vector once; collect single-k correlations.
    rank_cache = {}
    correlations = []
    for k in ks:
        col = sub_df[f"k{k}_distance"]
        rank_cache[k] = col.rank().to_numpy()
        corr, _ = spearmanr(col, tree_dist)
        correlations.append((k, corr))

    # Single-k local minima (reported for reference only).
    local_minima = []
    for i in range(1, len(correlations) - 1):
        k, corr = correlations[i]
        if corr < correlations[i - 1][1] and corr < correlations[i + 1][1]:
            local_minima.append((k, corr))
    local_minima = sorted(local_minima, key=lambda x: x[1])  # best (most neg) first

    # Globally best single k.
    valid_single = [(k, c) for k, c in correlations if not np.isnan(c)]
    best_single = min(valid_single, key=lambda x: x[1]) if valid_single else None

    # Fast Spearman vs tree_distance: rank + center the target once, then for
    # each candidate only its ranks change.
    t_centered = tree_dist.rank().to_numpy()
    t_centered = t_centered - t_centered.mean()
    t_norm = np.sqrt((t_centered ** 2).sum())

    def fast_spearman(values):
        vr = _fast_rank(values)
        vr = vr - vr.mean()
        denom = np.sqrt((vr ** 2).sum()) * t_norm
        if denom == 0:
            return np.nan
        return float((vr * t_centered).sum() / denom)

    # Exhaustive sweep over every k-pair and every weight.
    weights = [round(w, 2) for w in np.arange(w_step, 1.0, w_step)]
    best_pair = None  # (k1, k2, w, corr)
    for i in range(len(ks)):
        r1 = rank_cache[ks[i]]
        for j in range(i + 1, len(ks)):
            r2 = rank_cache[ks[j]]
            for w in weights:
                corr = fast_spearman(w * r1 + (1 - w) * r2)
                if not np.isnan(corr) and (best_pair is None or corr < best_pair[3]):
                    best_pair = (ks[i], ks[j], w, corr)
    weighted_minima = best_pair if best_pair is not None else (None, None, None, None)

    # Best overall: globally best weighted pair vs globally best single k.
    if best_pair is not None and (best_single is None or best_pair[3] < best_single[1]):
        best_corr = best_pair
    elif best_single is not None:
        best_corr = best_single
    else:
        best_corr = None

    return local_minima, weighted_minima, best_corr


def compute_node_annotations(tree, merged_df, k_low=6, k_high=40, w_step=0.05,
                             masked=None):
    """Sweep correlations for every internal node of `tree`.

    For each internal node, the II set is every pair of leaves under it. The
    correlation between each ``k{k}_distance`` (ranked) and ``tree_distance`` is
    swept over ``[k_low, k_high]``, and an exhaustive global search over every
    pair of ks and every weight looks for the most-negative weighted-rank
    correlation. ``best_corr`` is whichever is better (most negative) between the
    globally best single k and the globally best weighted pair (see
    `_sweep_best_corr`).

    Args:
        tree: A tree from `load_named_tree` (internal nodes must be named).
        merged_df: DataFrame with `genome_pair`, `tree_distance`, and
            `k{K}_distance` columns.
        k_low, k_high: Inclusive k-sweep range.
        w_step: Step for the weight search in ``(0, 1)``.
        masked: Optional set of genome (leaf) names to exclude. Each node's II
            set is restricted to pairs of un-masked genomes; the tree topology
            and node names are unchanged.

    Returns:
        dict: ``{node_name: {'local_minima': [...], 'weighted_minima': (...),
        'best_corr': ...}}``. ``best_corr`` is ``None`` for nodes with too few
        leaves / no local minima; a ``(k, corr)`` tuple for a single k; or a
        ``(k1, k2, w, corr)`` tuple for a weighted pair.
    """
    masked = set() if masked is None else set(masked)

    # Index by genome_pair for fast subsetting.
    if merged_df.index.name == "genome_pair":
        df = merged_df
    else:
        df = merged_df.set_index("genome_pair", drop=False)

    internal_nodes = [n for n in tree.traverse() if not n.is_leaf()]
    node_info = {}

    for node in internal_nodes:
        leaf_names = [l for l in node.get_leaf_names() if l not in masked]
        pairs = ["_".join(sorted(pair)) for pair in combinations(leaf_names, 2)]
        pairs = [p for p in pairs if p in df.index]

        if len(pairs) < 3:  # too few comparisons to be meaningful
            node_info[node.name] = {
                "local_minima": [], "weighted_minima": (None, None, None, None),
                "best_corr": None}
            continue

        local_minima, weighted_minima, best_corr = _sweep_best_corr(
            df.loc[pairs], k_low=k_low, k_high=k_high, w_step=w_step)
        node_info[node.name] = {"local_minima": local_minima,
                                "weighted_minima": weighted_minima,
                                "best_corr": best_corr}

    # Verify the dataset-level direction of the signal. The k-mer distances used
    # here decrease as phylogenetic distance grows, so the correlation between a
    # k-mer distance and `tree_distance` is negative. We assert on the
    # whole-tree (root) node, whose sweep uses every genome pair; tiny subtrees
    # can have noisy near-zero correlations of either sign and are not checked.
    # Skip the check when masking, since a masked subsample can flip the sign.
    if not masked:
        root_info = node_info.get(tree.name)
        if root_info is not None and root_info["best_corr"] is not None:
            root_corr = root_info["best_corr"][-1]
            assert root_corr < 0, (
                "Expected a negative k-mer / tree-distance correlation, but the "
                f"whole-tree node {tree.name!r} has correlation {root_corr}."
            )

    return node_info


################################################################################
# Genome masking and recommendation summaries

def mask_genomes(tree, frac, seed=None):
    """Randomly choose a fraction of the tree's leaves (genomes) to mask.

    Args:
        tree: An ete3 tree (or any object with `get_leaf_names`).
        frac: Fraction of genomes to mask, in ``[0, 1)``.
        seed: Optional seed for the random generator (reproducibility).

    Returns:
        set: The masked genome (leaf) names.
    """
    if not (0 <= frac < 1):
        raise ValueError(f"frac must be in [0, 1), got {frac}")
    leaves = tree.get_leaf_names()
    n_mask = int(np.floor(frac * len(leaves)))
    if n_mask == 0:
        return set()
    rng = np.random.default_rng(seed)
    return set(rng.choice(np.asarray(leaves), size=n_mask, replace=False).tolist())


def recommendation_summary(best_corr):
    """Summarize a node's `best_corr` into scalar recommendation features.

    Args:
        best_corr: ``None``, a ``(k, corr)`` single-k tuple, or a
            ``(k1, k2, w, corr)`` weighted-pair tuple.

    Returns:
        dict: ``{ks, n_ks, k_eff, corr}``. ``k_eff`` is the weighted effective
        k (``w*k1 + (1-w)*k2`` for a pair, ``k`` for a single k). All values are
        ``None`` / ``nan`` when ``best_corr`` is ``None``.
    """
    if best_corr is None:
        return {"ks": None, "n_ks": 0, "k_eff": np.nan, "corr": np.nan}
    if len(best_corr) == 4:
        k1, k2, w, corr = best_corr
        return {"ks": (k1, k2), "n_ks": 2,
                "k_eff": w * k1 + (1 - w) * k2, "corr": corr}
    k, corr = best_corr
    return {"ks": (k,), "n_ks": 1, "k_eff": float(k), "corr": corr}


def _best_corr_ks_weight(best_corr):
    """Return ``(k1, k2, w)`` from a `best_corr` tuple; NaN for unused fields."""
    if best_corr is None:
        return np.nan, np.nan, np.nan
    if len(best_corr) == 4:
        k1, k2, w, _ = best_corr
        return k1, k2, w
    k, _ = best_corr
    return k, np.nan, np.nan


def _same_ks_and_weight(full_best, masked_best):
    """True when masked recommendation uses the same k-mer(s) and weight as full."""
    if full_best is None or masked_best is None:
        return False
    fs = recommendation_summary(full_best)
    ms = recommendation_summary(masked_best)
    if fs["ks"] != ms["ks"]:
        return False
    if len(full_best) == 2:
        return True
    return full_best[2] == masked_best[2]


def _recommendation_match_row(full_best, masked_best, tol):
    """Build per-run match features comparing masked vs full recommendations."""
    fs = recommendation_summary(full_best)
    ms = recommendation_summary(masked_best)
    k_eff = ms["k_eff"]
    k_eff_global = fs["k_eff"]
    matched_k_eff = (bool(abs(k_eff - k_eff_global) <= tol)
                     if not np.isnan(k_eff) else False)
    matched_ks = (bool(ms["ks"] == fs["ks"])
                  if ms["ks"] is not None and fs["ks"] is not None else False)
    matched_ks_w = matched_ks and _same_ks_and_weight(full_best, masked_best)
    k1, k2, w = _best_corr_ks_weight(masked_best)
    k_eff_dev = k_eff - k_eff_global if not np.isnan(k_eff) else np.nan
    return {
        "k_eff": k_eff,
        "n_ks": ms["n_ks"],
        "k1": k1,
        "k2": k2,
        "w": w,
        "matched_k_eff": matched_k_eff,
        "matched_ks": matched_ks,
        "matched_ks_w": matched_ks_w,
        "k_eff_dev": k_eff_dev,
    }


class GlobalEffectiveMatchResults(object):
    """Simulation output for whole-tree effective-match robustness."""

    def __init__(self, runs, full, full_best_corr, meta):
        self.runs = runs
        self.full = full
        self.full_best_corr = full_best_corr
        self.meta = meta


def save_global_effective_match_results(results, path):
    """Pickle `GlobalEffectiveMatchResults` to `path`."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)


def load_global_effective_match_results(path):
    """Load `GlobalEffectiveMatchResults` from a pickle written by `save_...`."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _normalize_effective_match_results(results, tol=None):
    """Accept dataclass or legacy DataFrame; return (runs_df, full, meta, tol)."""
    if isinstance(results, GlobalEffectiveMatchResults):
        df = results.runs
        full = results.full
        meta = results.meta
        if tol is None:
            tol = meta.get("tol", 1)
        return df, full, meta, tol

    df = results
    full = df.attrs.get("full", {})
    meta = {}
    if tol is None:
        tol = 1
    if "matched_k_eff" not in df.columns and "matched" in df.columns:
        df = df.copy()
        df["matched_k_eff"] = df["matched"]
    if "k_eff_dev" not in df.columns and "k_eff" in df.columns and full:
        df = df.copy()
        df["k_eff_dev"] = df["k_eff"] - full.get("k_eff", np.nan)
    return df, full, meta, tol


def _as_global_consistency_df(results):
    """Return a `global_consistency` runs DataFrame; reject effective-match results."""
    if isinstance(results, GlobalEffectiveMatchResults):
        raise TypeError(
            "plot_global_consistency expects output from global_consistency "
            "(pd.DataFrame), not GlobalEffectiveMatchResults. "
            "Use plot_global_effective_match_consistency instead.")
    return results


def node_recommendation_table(tree, node_info):
    """Build a per-node table of recommendation features and node geometry.

    Args:
        tree: Named tree (from `load_named_tree`).
        node_info: Output of `compute_node_annotations`.

    Returns:
        pd.DataFrame: indexed by node_name with columns ``k_eff, n_ks, corr,
        dist_from_root`` (branch-length distance), ``depth_from_root``
        (topological depth) and ``subtree_size`` (number of descendant leaves).
        Nodes without a recommendation (``best_corr is None``) are dropped.
    """
    rows = {}
    for node in tree.traverse():
        if node.is_leaf():
            continue
        info = node_info.get(node.name)
        if info is None or info["best_corr"] is None:
            continue
        summ = recommendation_summary(info["best_corr"])
        rows[node.name] = {
            "k_eff": summ["k_eff"],
            "n_ks": summ["n_ks"],
            "corr": summ["corr"],
            "dist_from_root": node.get_distance(tree),
            "depth_from_root": len(node.get_ancestors()),
            "subtree_size": len(node.get_leaf_names()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def recommendation_pair_scores(sub_df, best_corr):
    """Per-pair scores defining the ordering a recommendation induces.

    Single-k recommendations rank pairs by that k's distance; two-k pairs rank
    by the weighted rank ``w*rank(k1) + (1-w)*rank(k2)``.

    Args:
        sub_df: Rows of `merged_df` for the pairs of interest (index =
            genome_pair).
        best_corr: A `best_corr` tuple (single-k or weighted pair).

    Returns:
        pd.Series: per-pair score (index aligned with `sub_df`), or ``None`` if
        `best_corr` is ``None``.
    """
    if best_corr is None:
        return None
    if len(best_corr) == 4:
        k1, k2, w, _ = best_corr
        return (w * sub_df[f"k{k1}_distance"].rank()
                + (1 - w) * sub_df[f"k{k2}_distance"].rank())
    k, _ = best_corr
    return sub_df[f"k{k}_distance"].rank()


def kendall_tau_distance(scores_a, scores_b):
    """Normalized Kendall tau distance in ``[0, 1]`` between two score vectors.

    Derived from `scipy.stats.kendalltau` as ``(1 - tau) / 2`` (tie-robust):
    0 means identical ordering, 1 means fully reversed. This equals the
    fraction of discordant pairs, i.e. the normalized number of adjacent-swap
    edits to turn one ranking into the other.

    Returns:
        float: distance, or ``nan`` if undefined (e.g. fewer than 2 items).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return np.nan
    tau, _ = kendalltau(a, b)
    if tau is None or np.isnan(tau):
        return np.nan
    return (1.0 - tau) / 2.0


################################################################################
# Annotated tree plot

def plot_tree_annotated(tree, node_info, name=None, k_low=6, k_high=40,
                        k_color=None, show_corr=True, show_weight=True,
                        masked=None, outpath=None, width=1500, scale=2000,
                        display=True, dpi=None):
    """Render a tree, annotating internal nodes with their best correlation.

    Each internal node with a non-``None`` ``best_corr`` gets:
      - a `PieChartFace` of the k weight(s) colored via `k_color` (when
        `show_weight`),
      - a `TextFace` label summarizing k / w (and ``corr`` when `show_corr`).
        The stored correlation is negative, so it is reported sign-flipped
        (i.e. ``corr = -stored``) as a positive value.

    Args:
        tree: Named tree (from `load_named_tree`).
        node_info: Output of `compute_node_annotations`.
        name: Optional dataset name for the default output filename.
        k_low, k_high: k range (for the legend colorbar and `k_color`).
        k_color: Optional precomputed ``{k: color}`` dict; built if ``None``.
        show_corr: Include the (sign-flipped) correlation in the text label.
        show_weight: Draw the k pie chart + text label.
        masked: Optional set of masked genome (leaf) names; masked leaves are
            grayed out in the rendering.
        outpath: Output image path.
        width: Render width in pixels.
        scale: Branch-length scale.
        display: Show the rendered image inline.

    Returns:
        str: the output image path.
    """
    if k_color is None:
        k_color = make_k_color(k_low, k_high)
    if outpath is None:
        outpath = f"{name or 'tree'}_tree_annotated.png"
    masked = set() if masked is None else set(masked)

    ts = _basic_tree_style(scale=scale)
    ts.show_scale = False        # entirely remove the branch-length scale bar
    ts.show_leaf_name = False    # leaf names drawn in layout for a larger font

    # Legend: k colorbar. For PNG output it is composited on afterwards so it can
    # be centered and sized precisely; otherwise fall back to ete3's legend.
    composite_legend = show_weight and str(outpath).lower().endswith(".png")
    if show_weight and not composite_legend:
        ts.legend.add_face(_k_colorbar_face(k_color), column=0)
        ts.legend_position = 3  # bottom-left (1=TL, 2=TR, 3=BL, 4=BR)

    def layout(node):
        nstyle = NodeStyle()
        nstyle["vt_line_width"] = 2
        nstyle["hz_line_width"] = 2

        if node.is_leaf():
            if node.name in masked:  # gray out masked genomes
                nstyle["fgcolor"] = "#cccccc"
                nstyle["hz_line_color"] = "#cccccc"
                nstyle["size"] = 4
            leaf_color = "#cccccc" if node.name in masked else "black"
            name_face = faces.AttrFace("name", fsize=13, fgcolor=leaf_color)
            faces.add_face_to_node(name_face, node, column=0,
                                   position="branch-right")

        # Skip annotations on small clades (<= 4 descendant genomes).
        best = None if node.is_leaf() else node_info.get(node.name, {}).get("best_corr")
        if best is not None and len(node.get_leaves()) > 4:
            if len(best) == 4:                     # weighted pair (k1, k2, w, corr)
                k1, k2, w, corr = best
                colors = [k_color.get(k1, "#808080"), k_color.get(k2, "#808080")]
                weights = [w * 100, (1 - w) * 100]
                label = f"k={k1},{k2}\nw={w}"
            else:                                  # single k (k, corr) -> w=1.0
                k, corr = best
                colors = [k_color.get(k, "#808080")]
                weights = [100]
                label = f"k={k}\nw=1.0"

            # Stored corr is negative; report it sign-flipped as positive.
            if show_corr:
                label += f"\ncorr={round(-corr, 3)}"

            if show_weight:
                pie = faces.PieChartFace(weights, colors=colors, width=50, height=50)
                pie.border.width = None
                faces.add_face_to_node(pie, node, 0, position="branch-right")

                text = faces.TextFace(label, fsize=15)
                faces.add_face_to_node(text, node, 1, position="branch-bottom")

        node.set_style(nstyle)

    ts.layout_fn = layout
    # Render first (no inline display yet); composite the centered legend, then show.
    _render(tree, outpath, ts, width=width, display=False, dpi=dpi)
    if composite_legend:
        _paste_centered_legend(outpath, _k_colorbar_png(k_color), dpi=dpi)
    if display and _HAVE_IPYTHON and str(outpath).lower().endswith(".png"):
        _ipy_display(Image(filename=outpath))
    return outpath


def _annotation_matches(best_corr, k1, k2, w, corr, tol_corr=0.002):
    """True when `best_corr` matches the displayed annotation fields."""
    if best_corr is None:
        return False
    if len(best_corr) == 4:
        bk1, bk2, bw, bcorr = best_corr
        if bk1 != k1 or bk2 != k2:
            return False
        if w is not None and abs(bw - w) > 1e-9:
            return False
        if corr is not None and abs(-bcorr - corr) > tol_corr:
            return False
        return True
    bk, bcorr = best_corr
    if k2 is not None:
        return False
    if bk != k1:
        return False
    if w is not None and abs(w - 1.0) > 1e-9:
        return False
    if corr is not None and abs(-bcorr - corr) > tol_corr:
        return False
    return True


def find_node_by_annotation(node_info, k1, k2, w=None, corr=None, tol_corr=0.002):
    """Return the internal-node name whose annotation matches the given fields.

    Displayed correlations are positive (``corr``); stored values are negative.
    For two-k recommendations, ``k1`` and ``k2`` are required. For single-k,
    pass ``k2=None``.
    """
    matches = []
    for name, info in node_info.items():
        best = info.get("best_corr")
        if _annotation_matches(best, k1, k2, w, corr, tol_corr=tol_corr):
            matches.append(name)
    if not matches:
        raise ValueError(
            "No node matches k1={k1}, k2={k2}, w={w}, corr={corr}".format(
                k1=k1, k2=k2, w=w, corr=corr))
    if len(matches) > 1:
        raise ValueError(
            "Multiple nodes match the annotation ({n}): {names}".format(
                n=len(matches), names=matches))
    return matches[0]


def subtree_at_node(tree, node_name):
    """Return a copy of the subtree rooted at `node_name`."""
    nodes = tree.search_nodes(name=node_name)
    if not nodes:
        raise ValueError("Node not found: {0}".format(node_name))
    return nodes[0].copy()


def subset_node_info(node_info, subtree):
    """Restrict `node_info` to internal nodes present in `subtree`."""
    names = {node.name for node in subtree.traverse() if not node.is_leaf()}
    return {name: node_info[name] for name in names if name in node_info}


def plot_subtree_annotated(tree, node_info, root_node_name, name=None,
                           k_low=6, k_high=40, k_color=None, show_corr=True,
                           show_weight=True, outpath=None, width=1500,
                           scale=2000, display=True, dpi=None):
    """Plot only the subtree rooted at `root_node_name` with its annotations."""
    subtree = subtree_at_node(tree, root_node_name)
    sub_info = subset_node_info(node_info, subtree)
    if outpath is None:
        suffix = root_node_name.replace("internal_node_", "node")
        outpath = "{base}_subtree_{suffix}_annotated.png".format(
            base=name or "tree", suffix=suffix)
    if outpath:  # ensure the destination directory exists
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plot_tree_annotated(subtree, sub_info, name=name, k_low=k_low, k_high=k_high,
                        k_color=k_color, show_corr=show_corr,
                        show_weight=show_weight, outpath=outpath, width=width,
                        scale=scale, display=display, dpi=dpi)
    return subtree, sub_info, outpath


################################################################################
# Convenience wrapper

def plot_dataset_tree(name, tree_path, merged_df=None, annotate=False,
                      k_low=6, k_high=40, w_step=0.05, outdir=".",
                      width=1500, scale=2000, display=True, **kwargs):
    """Plot one dataset's tree, plain or annotated.

    Args:
        name: Dataset name (used in output filenames).
        tree_path: Path to the ``.nwk`` file.
        merged_df: Required when ``annotate=True``.
        annotate: If ``True``, compute and draw per-node annotations.
        k_low, k_high, w_step: Passed to `compute_node_annotations`.
        outdir: Directory for the output PNG.
        width, scale, display: Render options.
        **kwargs: Forwarded to `plot_tree_annotated` (e.g. ``show_corr``,
            ``show_weight``).

    Returns:
        tuple: ``(tree, node_info)``. ``node_info`` is ``None`` when not
        annotating.
    """
    tree = load_named_tree(tree_path)

    if not annotate:
        outpath = os.path.join(outdir, f"{name}_tree.png")
        ts = _basic_tree_style(scale=scale)
        _render(tree, outpath, ts, width=width, display=display)
        return tree, None

    if merged_df is None:
        raise ValueError("merged_df is required when annotate=True")

    node_info = compute_node_annotations(
        tree, merged_df, k_low=k_low, k_high=k_high, w_step=w_step)
    outpath = os.path.join(outdir, f"{name}_tree_annotated.png")
    plot_tree_annotated(tree, node_info, name=name, k_low=k_low, k_high=k_high,
                        outpath=outpath, width=width, scale=scale,
                        display=display, **kwargs)
    return tree, node_info


def plot_masked_tree(name, tree_path, merged_df, frac, seed=None,
                     k_low=6, k_high=40, w_step=0.05, outdir=".",
                     width=1500, scale=2000, display=True, **kwargs):
    """Mask a fraction of genomes, then plot the annotated tree on the rest.

    The full tree topology is kept; masked genomes are excluded from each
    internal node's correlation so the annotations reflect the un-masked subset.

    Args:
        name: Dataset name (used in the output filename).
        tree_path: Path to the ``.nwk`` file.
        merged_df: DataFrame of pairwise distances for this dataset.
        frac: Fraction of genomes to mask, in ``[0, 1)``.
        seed: Optional RNG seed for reproducible masking.
        k_low, k_high, w_step: Passed to `compute_node_annotations`.
        outdir: Directory for the output PNG.
        width, scale, display: Render options.
        **kwargs: Forwarded to `plot_tree_annotated`.

    Returns:
        tuple: ``(tree, node_info, masked)``.
    """
    tree = load_named_tree(tree_path)
    masked = mask_genomes(tree, frac, seed=seed)
    node_info = compute_node_annotations(
        tree, merged_df, k_low=k_low, k_high=k_high, w_step=w_step, masked=masked)
    pct = int(round(frac * 100))
    outpath = os.path.join(outdir, f"{name}_tree_annotated_mask{pct}.png")
    plot_tree_annotated(tree, node_info, name=name, k_low=k_low, k_high=k_high,
                        masked=masked, outpath=outpath, width=width, scale=scale,
                        display=display, **kwargs)
    return tree, node_info, masked


################################################################################
# Robustness analyses

def _as_pair_indexed(merged_df):
    """Return `merged_df` indexed by `genome_pair` (without copying if already)."""
    if merged_df.index.name == "genome_pair":
        return merged_df
    return merged_df.set_index("genome_pair", drop=False)


def global_consistency(tree, merged_df, fracs, n_runs=10, k_low=6, k_high=40,
                       w_step=0.05, seed=0, tol=1):
    """Whole-tree recommendation robustness vs masking fraction.

    For each masking fraction, masks `n_runs` random genome subsets and
    recomputes the whole-tree (root) recommendation on the remaining genomes,
    comparing it to the full-data recommendation.

    Args:
        tree: Named tree (used only to draw masked genome names).
        merged_df: Dataset distances (`genome_pair`/`genome_1`/`genome_2`,
            `tree_distance`, `k{K}_distance`).
        fracs: Iterable of masking fractions in ``[0, 1)``.
        n_runs: Random masking repeats per fraction.
        k_low, k_high, w_step: Sweep parameters.
        seed: Master RNG seed.
        tol: ``|k_eff|`` tolerance (k units) for the match-rate.

    Returns:
        pd.DataFrame: one row per (frac, run) with columns ``frac, run, k_eff,
        corr, matched, abs_dev, corr_retention, ktau_full_vs_masked,
        ktau_degradation``. ``df.attrs['full']`` holds the full-data summary.
    """
    df = _as_pair_indexed(merged_df)
    _, _, best_full = _sweep_best_corr(df, k_low, k_high, w_step)
    full = recommendation_summary(best_full)

    rng_master = np.random.default_rng(seed)
    rows = []
    for frac in fracs:
        for run in range(n_runs):
            run_seed = int(rng_master.integers(0, 2**31 - 1))
            masked = mask_genomes(tree, frac, seed=run_seed)
            keep = ~(df["genome_1"].isin(masked) | df["genome_2"].isin(masked))
            sub = df[keep]

            _, _, best_m = _sweep_best_corr(sub, k_low, k_high, w_step)
            summ = recommendation_summary(best_m)

            # Genome-pair ordering comparison on the shared (un-masked) pairs.
            full_scores = recommendation_pair_scores(sub, best_full)
            masked_scores = recommendation_pair_scores(sub, best_m)
            ktau_fm = kendall_tau_distance(full_scores, masked_scores)
            ktau_ft = kendall_tau_distance(full_scores, sub["tree_distance"])
            ktau_mt = kendall_tau_distance(masked_scores, sub["tree_distance"])

            k_eff = summ["k_eff"]
            rows.append({
                "frac": float(frac), "run": run,
                "k_eff": k_eff, "corr": summ["corr"],
                "matched": bool(abs(k_eff - full["k_eff"]) <= tol)
                if not np.isnan(k_eff) else False,
                "abs_dev": abs(k_eff - full["k_eff"]),
                "corr_retention": abs(summ["corr"]) / abs(full["corr"])
                if full["corr"] else np.nan,
                "ktau_full_vs_masked": ktau_fm,
                "ktau_degradation": ktau_mt - ktau_ft,
            })

    out = pd.DataFrame(rows)
    out.attrs["full"] = full
    return out


def global_effective_match_consistency(tree, merged_df, fracs, n_runs=10,
                                       k_low=6, k_high=40, w_step=0.05, seed=0,
                                       tol=1):
    """Whole-tree effective-k match rate vs masking fraction.

    A lighter variant of `global_consistency` that computes only the effective
    match rate: for each masking fraction it masks `n_runs` random genome
    subsets, recomputes the whole-tree (root) recommendation on the remaining
    genomes, and flags a match when the masked ``k_eff`` is within `tol` of the
    full-data ``k_eff``. The Kendall-tau / dispersion / correlation-retention
    measures are skipped, so this is considerably faster.

    Args:
        tree: Named tree.
        merged_df: Dataset distances (`genome_pair`/`genome_1`/`genome_2`,
            `tree_distance`, `k{K}_distance`).
        fracs: Iterable of masking fractions in ``[0, 1)``.
        n_runs: Random masking repeats per fraction.
        k_low, k_high, w_step: Sweep parameters.
        seed: Master RNG seed.
        tol: ``|k_eff|`` tolerance (k units) for the match flag.

    Returns:
        GlobalEffectiveMatchResults: ``runs`` has one row per (frac, run) with
        columns ``frac, run, k_eff, n_ks, k1, k2, w, matched_k_eff,
        matched_ks, matched_ks_w, k_eff_dev``. ``full`` holds the full-data
        summary; ``full_best_corr`` is the raw best_corr tuple.
    """
    df = _as_pair_indexed(merged_df)
    fracs = list(fracs)
    _, _, best_full = _sweep_best_corr(df, k_low, k_high, w_step)
    full = recommendation_summary(best_full)

    rng_master = np.random.default_rng(seed)
    rows = []
    for frac in fracs:
        for run in range(n_runs):
            run_seed = int(rng_master.integers(0, 2**31 - 1))
            masked = mask_genomes(tree, frac, seed=run_seed)
            keep = ~(df["genome_1"].isin(masked) | df["genome_2"].isin(masked))
            sub = df[keep]

            _, _, best_m = _sweep_best_corr(sub, k_low, k_high, w_step)
            row = _recommendation_match_row(best_full, best_m, tol)
            row.update({"frac": float(frac), "run": run})
            rows.append(row)

    meta = {
        "tol": tol,
        "k_low": k_low,
        "k_high": k_high,
        "w_step": w_step,
        "seed": seed,
        "fracs": fracs,
        "n_runs": n_runs,
    }
    return GlobalEffectiveMatchResults(
        runs=pd.DataFrame(rows),
        full=full,
        full_best_corr=best_full,
        meta=meta,
    )


def node_consistency(tree, merged_df, fracs, n_runs=5, k_low=6, k_high=40,
                     w_step=0.05, seed=0, tol=1):
    """Per-internal-node recommendation robustness vs masking fraction.

    The full tree topology (and node identities) is preserved, so each node's
    masked recommendation is compared to its full-data recommendation directly.

    Args:
        tree: Named tree.
        merged_df: Dataset distances.
        fracs: Iterable of masking fractions in ``[0, 1)``.
        n_runs: Random masking repeats per fraction.
        k_low, k_high, w_step: Sweep parameters.
        seed: Master RNG seed.
        tol: ``|k_eff|`` tolerance (k units) for the per-node match flag.

    Returns:
        pd.DataFrame: one row per (frac, run, node) that has a recommendation in
        both the full and masked runs, with columns ``frac, run, node,
        k_eff_full, k_eff_masked, matched, n_ks_match, ktau_full_vs_masked,
        ktau_degradation``.
    """
    df = _as_pair_indexed(merged_df)
    base = compute_node_annotations(tree, df, k_low=k_low, k_high=k_high,
                                    w_step=w_step)
    base_summ = {n: recommendation_summary(v["best_corr"]) for n, v in base.items()}

    internal_nodes = [n for n in tree.traverse() if not n.is_leaf()]
    node_leaves = {n.name: n.get_leaf_names() for n in internal_nodes}

    rng_master = np.random.default_rng(seed)
    rows = []
    for frac in fracs:
        for run in range(n_runs):
            run_seed = int(rng_master.integers(0, 2**31 - 1))
            masked = mask_genomes(tree, frac, seed=run_seed)
            ni = compute_node_annotations(tree, df, k_low=k_low, k_high=k_high,
                                          w_step=w_step, masked=masked)
            for name, info in ni.items():
                base_best = base[name]["best_corr"]
                run_best = info["best_corr"]
                if base_best is None or run_best is None:
                    continue
                bs, rs = base_summ[name], recommendation_summary(run_best)

                # Shared (un-masked) pairs under this node.
                leaves = [l for l in node_leaves[name] if l not in masked]
                pairs = ["_".join(sorted(p)) for p in combinations(leaves, 2)]
                pairs = [p for p in pairs if p in df.index]
                ktau_fm = ktau_deg = np.nan
                if len(pairs) >= 2:
                    sub = df.loc[pairs]
                    full_scores = recommendation_pair_scores(sub, base_best)
                    masked_scores = recommendation_pair_scores(sub, run_best)
                    ktau_fm = kendall_tau_distance(full_scores, masked_scores)
                    ktau_ft = kendall_tau_distance(full_scores, sub["tree_distance"])
                    ktau_mt = kendall_tau_distance(masked_scores, sub["tree_distance"])
                    ktau_deg = ktau_mt - ktau_ft

                rows.append({
                    "frac": float(frac), "run": run, "node": name,
                    "k_eff_full": bs["k_eff"], "k_eff_masked": rs["k_eff"],
                    "matched": bool(abs(rs["k_eff"] - bs["k_eff"]) <= tol),
                    "n_ks_match": bool(rs["n_ks"] == bs["n_ks"]),
                    "ktau_full_vs_masked": ktau_fm,
                    "ktau_degradation": ktau_deg,
                })

    return pd.DataFrame(rows)


def _plot_mean_band(ax, results_df, value, label=None, color="C0"):
    """Plot mean +/- 1 std of `value` over runs vs masking fraction on `ax`."""
    grp = results_df.groupby("frac")[value]
    mean, std = grp.mean(), grp.std()
    fracs = mean.index.to_numpy(dtype=float)
    ax.plot(fracs, mean.to_numpy(), marker="o", color=color, label=label)
    ax.fill_between(fracs, (mean - std).to_numpy(), (mean + std).to_numpy(),
                    color=color, alpha=0.2)
    ax.set_xlabel("masking fraction")
    if label:
        ax.set_ylabel(label)


def _plot_frac_scatter(ax, df, y_col, x_jitter=0.012, alpha=0.4, s=12, color="C0"):
    """Scatter `y_col` vs masking fraction with reproducible x jitter."""
    x = df["frac"].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    rng = np.random.default_rng(0)
    x_plot = x + rng.uniform(-x_jitter, x_jitter, size=len(x))
    ax.scatter(x_plot, y, alpha=alpha, s=s, color=color, edgecolors="none")
    ax.set_xlabel("masking fraction")


def _plot_match_strip(ax, df, value_col, ylabel, color="C0"):
    """Jittered binary outcomes plus mean match rate per masking fraction."""
    _plot_frac_scatter(ax, df, value_col, alpha=0.35, s=10, color=color)
    grp = df.groupby("frac")[value_col]
    mean = grp.mean()
    fracs = mean.index.to_numpy(dtype=float)
    ax.plot(fracs, mean.to_numpy(), marker="o", color=color, linewidth=2, zorder=5)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.05)


def plot_global_consistency(results_df, name, display=True):
    """Plot whole-tree robustness metrics vs masking fraction (4 panels)."""
    results_df = _as_global_consistency_df(results_df)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    _plot_mean_band(axes[0], results_df, "matched", "match rate (+/-1 k)")
    axes[0].set_ylim(-0.05, 1.05)
    _plot_mean_band(axes[1], results_df, "abs_dev", "mean |dk_eff|", color="C1")
    _plot_mean_band(axes[2], results_df, "corr_retention",
                    "correlation retention", color="C2")
    axes[3].set_title("genome-pair ordering")
    _plot_mean_band(axes[3], results_df, "ktau_full_vs_masked",
                    "Kendall tau distance", color="C3")
    _plot_mean_band(axes[3], results_df, "ktau_degradation",
                    "ktau degradation vs truth", color="C4")
    axes[3].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{name}: whole-tree recommendation robustness")
    fig.tight_layout()
    if display:
        plt.show()
    return fig


def plot_global_effective_match_consistency(results, name, display=True, tol=None):
    """Plot whole-tree recommendation robustness vs masking fraction (2x2)."""
    df, full, meta, tol = _normalize_effective_match_results(results, tol=tol)
    k_eff_global = full.get("k_eff", np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Panel 1: per-run k_eff values.
    _plot_frac_scatter(ax1, df, "k_eff")
    ax1.axhline(k_eff_global, color="C3", linestyle="--", linewidth=1.5,
                label=f"global k_eff = {k_eff_global:.2f}")
    mean = df.groupby("frac")["k_eff"].mean()
    fracs = mean.index.to_numpy(dtype=float)
    ax1.plot(fracs, mean.to_numpy(), color="C0", linewidth=1, alpha=0.5, label="mean")
    ax1.set_ylabel("Recommended effective k (k_eff)")
    ax1.legend(frameon=False, fontsize=8)

    # Panel 2: exact k-mer(s) and weight match.
    if "matched_ks_w" in df.columns:
        _plot_match_strip(ax2, df, "matched_ks_w",
                          "Match (same k-mer(s) and weight)")
    else:
        ax2.text(0.5, 0.5, "Re-run simulation for\nmatched_ks_w data",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_xlabel("masking fraction")

    # Panel 3: exact k-mer(s) match (weight may differ).
    if "matched_ks" in df.columns:
        _plot_match_strip(ax3, df, "matched_ks",
                          "Match (same k-mer(s), any weight)", color="C1")
    else:
        ax3.text(0.5, 0.5, "Re-run simulation for\nmatched_ks data",
                 ha="center", va="center", transform=ax3.transAxes)
        ax3.set_xlabel("masking fraction")

    # Panel 4: deviation from global k_eff baseline.
    _plot_frac_scatter(ax4, df, "k_eff_dev", color="C2")
    ax4.axhline(0, color="C3", linestyle="--", linewidth=1.5)
    ax4.set_ylabel("Δk_eff (masked − global)")
    ax4.set_title(f"global k_eff = {k_eff_global:.2f}, tol = ±{tol} k", fontsize=9)

    fig.suptitle(f"{name}: whole-tree recommendation robustness")
    fig.tight_layout()
    if display:
        plt.show()
    return fig


def plot_effective_length_vs_masking(results_by_name, display=True, show_spread=True,
                                      ax=None):
    """Overlay mean effective k vs masking fraction for several datasets.

    Args:
        results_by_name: Mapping ``{dataset_name: results}`` where each value is
            a `GlobalEffectiveMatchResults` (or legacy DataFrame) from
            `global_effective_match_consistency`.
        display: Show the figure inline.
        show_spread: Shade mean +/- 1 std of k_eff across runs per fraction.
        ax: Optional existing axis to draw on.

    Returns:
        matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    else:
        fig = ax.figure

    label_overrides = {
        "Aquificota": "p_Aquificota (132)",
        "Bacillales": "o_Bacillales_A (249)",
        "Caldisericia": "c_Caldisericia (49)",
        "Chlamydia": "o_Chlamydiales (288)",
        "Cyano": "o_Cyanobacteriales (585)",
        "Neisseriaceae": "f_Neisseriaceae (145)",
    }

    color_overrides = {"Cyano": "C2", "Caldisericia": "C4"}

    for i, (name, results) in enumerate(results_by_name.items()):
        df, _full, _meta, _tol = _normalize_effective_match_results(results)
        grp = df.groupby("frac")["k_eff"]
        mean = grp.mean()
        fracs = mean.index.to_numpy(dtype=float)
        color = color_overrides.get(name, f"C{i % 10}")
        ax.plot(fracs, mean.to_numpy(), marker="o", color=color,
                label=label_overrides.get(name, name))
        if show_spread:
            std = grp.std()
            ax.fill_between(fracs, (mean - std).to_numpy(), (mean + std).to_numpy(),
                            color=color, alpha=0.15)

    ax.set_xlabel("Masking fraction")
    ax.set_ylabel("Weighted composite rank")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _pos: f"{x * 100:.0f}%"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    fig.tight_layout()
    if display:
        plt.show()
    return fig


def plot_node_consistency(results_df, name, display=True):
    """Plot per-node robustness metrics vs masking fraction (3 panels)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _plot_mean_band(axes[0], results_df, "matched",
                    "node match rate (+/-1 k)")
    axes[0].set_ylim(-0.05, 1.05)
    _plot_mean_band(axes[1], results_df, "ktau_full_vs_masked",
                    "Kendall tau distance\n(full vs masked)", color="C3")
    _plot_mean_band(axes[2], results_df, "ktau_degradation",
                    "ktau degradation vs truth", color="C4")
    fig.suptitle(f"{name}: per-node recommendation robustness")
    fig.tight_layout()
    if display:
        plt.show()
    return fig


def plot_within_tree_correlations(table, name, display=True):
    """Scatter recommended k / number of ks vs node geometry, with Spearman rho.

    Panels:
      1. distance-from-root vs k_eff
      2. depth-from-root vs k_eff
      3. subtree size vs k_eff
      4. subtree size vs number of ks

    Args:
        table: Output of `node_recommendation_table`.
        name: Dataset name for the title.
        display: Show the figure inline.

    Returns:
        dict: Spearman ``{(x, y): (rho, pval)}`` for each panel.
    """
    panels = [
        ("dist_from_root", "k_eff"),
        ("depth_from_root", "k_eff"),
        ("subtree_size", "k_eff"),
        ("subtree_size", "n_ks"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    corrs = {}
    for ax, (xcol, ycol) in zip(axes, panels):
        x = table[xcol].to_numpy(dtype=float)
        y = table[ycol].to_numpy(dtype=float)
        ok = ~(np.isnan(x) | np.isnan(y))
        rho, pval = (spearmanr(x[ok], y[ok]) if ok.sum() >= 3 else (np.nan, np.nan))
        corrs[(xcol, ycol)] = (rho, pval)
        sc = ax.scatter(x, y, c=table["n_ks"].to_numpy(dtype=float),
                        cmap="viridis", s=18, alpha=0.7)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(f"rho={rho:.2f}, p={pval:.1e}")
    fig.colorbar(sc, ax=axes[-1], label="n_ks")
    fig.suptitle(f"{name}: within-tree recommendation correlations")
    fig.tight_layout()
    if display:
        plt.show()
    return corrs
