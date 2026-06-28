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

# ete3 needs a Qt platform; on a headless cluster use the offscreen backend.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr

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


def _render(tree, outpath, tree_style, width=1500, display=True):
    """Render `tree` to `outpath` and optionally show it inline."""
    tree.render(outpath, tree_style=tree_style, w=width)
    if display and _HAVE_IPYTHON:
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


def _k_colorbar_face(k_color, tmp_path="colorbar_k.png"):
    """Build an ImgFace colorbar for k values from a `k_color` dict."""
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
    face = faces.ImgFace(tmp_path)
    face.margin_top = 100
    face.margin_bottom = 100
    return face


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

def compute_node_annotations(tree, merged_df, k_low=6, k_high=40, w_step=0.05):
    """Sweep correlations for every internal node of `tree`.

    For each internal node, the II set is every pair of leaves under it. The
    correlation between each ``k{k}_distance`` (ranked) and ``tree_distance`` is
    swept over ``[k_low, k_high]``. Local minima (most-negative correlations) are
    found, and the two best minima are combined via a weighted average of their
    ranks to look for a stronger correlation. ``best_corr`` is whichever is
    better (most negative) between the best single-k minimum and the best
    weighted pair.

    Args:
        tree: A tree from `load_named_tree` (internal nodes must be named).
        merged_df: DataFrame with `genome_pair`, `tree_distance`, and
            `k{K}_distance` columns.
        k_low, k_high: Inclusive k-sweep range.
        w_step: Step for the weight search in ``(0, 1)``.

    Returns:
        dict: ``{node_name: {'local_minima': [...], 'weighted_minima': (...),
        'best_corr': ...}}``. ``best_corr`` is ``None`` for nodes with too few
        leaves / no local minima; a ``(k, corr)`` tuple for a single k; or a
        ``(k1, k2, w, corr)`` tuple for a weighted pair.
    """
    # Index by genome_pair for fast subsetting.
    if merged_df.index.name == "genome_pair":
        df = merged_df
    else:
        df = merged_df.set_index("genome_pair", drop=False)

    internal_nodes = [n for n in tree.traverse() if not n.is_leaf()]
    node_info = {}

    for node in internal_nodes:
        leaf_names = node.get_leaf_names()
        pairs = ["_".join(sorted(pair)) for pair in combinations(leaf_names, 2)]
        pairs = [p for p in pairs if p in df.index]

        info = {"local_minima": [], "weighted_minima": (None, None, None, None),
                "best_corr": None}

        if len(pairs) < 3:  # too few comparisons to be meaningful
            node_info[node.name] = info
            continue

        sub = df.loc[pairs]
        tree_dist = sub["tree_distance"]

        # Sweep single-k correlations.
        correlations = []
        for k in range(k_low, k_high + 1):
            col = f"k{k}_distance"
            if col not in sub.columns:
                continue
            corr, _ = spearmanr(sub[col], tree_dist)
            correlations.append((k, corr))

        # Local minima (more negative than both neighbors).
        local_minima = []
        for i in range(1, len(correlations) - 1):
            k, corr = correlations[i]
            if corr < correlations[i - 1][1] and corr < correlations[i + 1][1]:
                local_minima.append((k, corr))
        local_minima = sorted(local_minima, key=lambda x: x[1])  # best (most neg) first
        info["local_minima"] = local_minima

        # Weighted average of the two best local minima.
        k1 = k2 = best_w = best_w_corr = None
        if len(local_minima) >= 2:
            k1 = local_minima[0][0]
            k2 = local_minima[1][0]
            k1_ranks = sub[f"k{k1}_distance"].rank()
            k2_ranks = sub[f"k{k2}_distance"].rank()
            weighted_corrs = []
            for w in np.arange(w_step, 1.0, w_step):
                w = round(w, 2)
                weighted_ranks = w * k1_ranks + (1 - w) * k2_ranks
                corr, _ = spearmanr(weighted_ranks, tree_dist)
                weighted_corrs.append((w, corr))
            best_w, best_w_corr = min(weighted_corrs, key=lambda x: x[1])
        info["weighted_minima"] = (k1, k2, best_w, best_w_corr)

        # Best overall: weighted pair vs best single-k local minimum.
        if best_w_corr is not None and best_w_corr < local_minima[0][1]:
            info["best_corr"] = (k1, k2, best_w, best_w_corr)
        elif local_minima:
            info["best_corr"] = local_minima[0]
        else:
            info["best_corr"] = None

        node_info[node.name] = info

    # Verify the dataset-level direction of the signal. The k-mer distances used
    # here decrease as phylogenetic distance grows, so the correlation between a
    # k-mer distance and `tree_distance` is negative. We assert on the
    # whole-tree (root) node, whose sweep uses every genome pair; tiny subtrees
    # can have noisy near-zero correlations of either sign and are not checked.
    root_info = node_info.get(tree.name)
    if root_info is not None and root_info["best_corr"] is not None:
        root_corr = root_info["best_corr"][-1]
        assert root_corr < 0, (
            "Expected a negative k-mer / tree-distance correlation, but the "
            f"whole-tree node {tree.name!r} has correlation {root_corr}."
        )

    return node_info


################################################################################
# Annotated tree plot

def plot_tree_annotated(tree, node_info, name=None, k_low=6, k_high=40,
                        k_color=None, show_corr=True, show_weight=True,
                        outpath=None, width=1500, scale=2000, display=True):
    """Render a tree, annotating internal nodes with their best correlation.

    Each internal node with a non-``None`` ``best_corr`` gets:
      - a `PieChartFace` of the k weight(s) colored via `k_color` (when
        `show_weight`),
      - a `CircleFace` colored by ``-corr`` on `YlOrRd` over ``[0, 1]``
        (correlations are negative; stronger = darker red) (when `show_corr`),
      - a `TextFace` label summarizing k / w / corr.

    Args:
        tree: Named tree (from `load_named_tree`).
        node_info: Output of `compute_node_annotations`.
        name: Optional dataset name for the default output filename.
        k_low, k_high: k range (for the legend colorbar and `k_color`).
        k_color: Optional precomputed ``{k: color}`` dict; built if ``None``.
        show_corr: Draw the correlation circle.
        show_weight: Draw the k pie chart + text label.
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

    corr_cmap = plt.cm.YlOrRd

    ts = _basic_tree_style(scale=scale)

    # Legends: correlation colorbar + k colorbar.
    if show_corr:
        ts.legend.add_face(_corr_colorbar_face(cmap=corr_cmap), column=0)
    if show_weight:
        ts.legend.add_face(_k_colorbar_face(k_color), column=1)
    ts.legend_position = 3  # bottom-left (1=TL, 2=TR, 3=BL, 4=BR)

    def layout(node):
        nstyle = NodeStyle()
        nstyle["vt_line_width"] = 2
        nstyle["hz_line_width"] = 2

        best = None if node.is_leaf() else node_info.get(node.name, {}).get("best_corr")
        if best is not None:
            if len(best) == 4:                     # weighted pair (k1, k2, w, corr)
                k1, k2, w, corr = best
                colors = [k_color.get(k1, "#808080"), k_color.get(k2, "#808080")]
                weights = [w * 100, (1 - w) * 100]
                label = f"k={k1},{k2}\nw={w}\ncorr={round(corr, 3)}"
            else:                                  # single k (k, corr)
                k, corr = best
                colors = [k_color.get(k, "#808080")]
                weights = [100]
                label = f"k={k}\ncorr={round(corr, 3)}"

            if show_weight:
                pie = faces.PieChartFace(weights, colors=colors, width=50, height=50)
                pie.border.width = None
                faces.add_face_to_node(pie, node, 0, position="branch-right")

            if show_corr:
                # Correlations are negative; map -corr in [0, 1] to color
                # strength (clip guards rare noisy positive-corr subtrees).
                strength = float(np.clip(-corr, 0.0, 1.0))
                circle = faces.CircleFace(
                    radius=25, color=mcolors.rgb2hex(corr_cmap(strength)))
                faces.add_face_to_node(circle, node, 0, position="branch-right")

            if show_weight:
                text = faces.TextFace(label, fsize=15)
                faces.add_face_to_node(text, node, 1, position="branch-bottom")

        node.set_style(nstyle)

    ts.layout_fn = layout
    _render(tree, outpath, ts, width=width, display=display)
    return outpath


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
