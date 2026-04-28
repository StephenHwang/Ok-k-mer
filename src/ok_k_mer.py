# imports
import os
from random import randint
import multiprocessing as mp
from random import sample
from random import sample, seed as set_seed, randint
import numpy as np
import pandas as pd
from collections import Counter
from ete3 import Tree
from itertools import combinations, product
from scipy.stats import binom, geom, spearmanr, pearsonr


# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
################################################################################


# import plotnine
from plotnine import \
    ggplot, aes, theme, themes, element_blank, element_line, element_text, mapping, \
    geom_point, geom_jitter, geom_line, geom_bar, geom_boxplot, geom_violin, geom_histogram, \
    ggtitle, xlab, ylab, geom_vline, geom_hline, geom_abline, geom_col, geom_tile, \
    scale_x_discrete, scale_y_discrete, scale_x_continuous, scale_y_continuous, \
    scale_color_discrete, scale_fill_manual, scale_fill_hue, scale_fill_gradient, scale_color_manual, scale_color_hue, \
    guides, after_stat, \
    xlim, ylim
from plotnine.options import figure_size
from mizani.formatters import percent_format

# plotnine theme
def theme_tufte_func(base_size=19, base_family="sans", rotate=True, figure_size=None):
    ''' Plotnine plotting theme. '''
    thm = themes.theme_bw(base_size=base_size, base_family = base_family) + \
        theme(
            legend_background = element_blank(),
            legend_key = element_blank(),
            panel_background = element_blank(),
            panel_border = element_blank(),
            strip_background = element_blank(),
            plot_background = element_blank(),
            panel_grid = element_blank(),
            axis_line = element_line(colour = "black", size = 1),
            axis_text_y = element_text(colour = "black")
        )

    if rotate:
        thm += theme(axis_text_x=element_text(rotation=45, hjust=1))
    if figure_size is not None:
        thm += theme(figure_size=figure_size)
    return thm


################################################################################
# helper functions

def yield_file_lines(path):
    ''' Read file from path line-by-line. '''
    with open(path, 'r') as in_file:
        for line in in_file:
            yield line.strip()

def calc_corr(l1, l2, pearson=False, prnt=False):
    if pearson:
        corr, pval = pearsonr(l1, l2)
    else:
        corr, pval = spearmanr(l1, l2)
    if prnt:
        print("Correlation coefficient:", corr)
        print("p-value:", pval)
    return corr, pval

def dist_tree_calc(filename1):
    ''' Calculate tree distance. '''
    tree1= Tree(filename1,format=1)
    # print('Tree from', filename1, 'len:', len(tree1))
    dist_tree={}
    samples = [i.name for i in tree1.get_leaves()]
    num_sample = len(samples)
    for i in range(num_sample):
        tax_i = samples[i]
        for j in range(i+1,num_sample):
            tax_j = samples[j]
            dist_tree[(min(tax_i, tax_j)+".fa", max(tax_i, tax_j)+".fa")] = tree1.get_distance(tax_i, tax_j, topology_only=False)
    # print('num samples:', num_sample)
    # print('xxx:', num_sample * (num_sample-1)/2)
    # print('tree len:', len(dist_tree))
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
        g1_clean = g1.rstrip('.fa')
        g2_clean = g2.rstrip('.fa')
        names.add(g1_clean)
        names.add(g2_clean)
        processed_pairs.append((g1_clean, g2_clean, float(dist)))

    tree_dict = {name: {other: 0.0 for other in names} for name in names}
    for g1, g2, d in processed_pairs:
        tree_dict[g1][g2] = d
        tree_dict[g2][g1] = d  # Ensure symmetry

    return tree_dict



def get_random_samples(sample_names, sample_fraction=0.1, seed=None):
    '''
    # Example usage:
    cyano_random_samples = get_random_samples(cyano_sample_names, sample_fraction=0.1, seed=42)
    '''
    if seed is not None:
        set_seed(seed)
    num_samples = max(1, int(len(sample_names) * sample_fraction))
    random_samples = sample(sample_names, num_samples)
    return random_samples


def parse_dashing_dfs(dir_path):
    """
    Parses all distance_* files in the given directory into a dictionary of pandas DataFrames.
    Each DataFrame skips the first line, assumes tab-separation and adjusts indices and columns as expected.

    Args:
        dir_path (str): Path to the directory containing distance_* files.

    Returns:
        dict: Dictionary mapping filename to corresponding DataFrame.
    """

    # Get list of files in directory that start with 'distance_'
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.startswith('distance_')]

    # Create a dictionary to store dataframes
    dashing_dfs = {}

    # Read each file into a dataframe, skipping first line
    for file in files:
        file_path = os.path.join(dir_path, file)
        try:
            # Skip first line and read tab-separated data
            df = pd.read_csv(file_path, sep='\t', header=None, skiprows=1)
            df.index = df.iloc[:, 0].str.replace('genomes_ncbi/', '')
            df = df.iloc[:, 1:]
            df.columns = df.index
            dashing_dfs[file] = df
            # print(f"Successfully read {file}")
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")

    # Print summary of loaded dataframes
    print(f"\nLoaded {len(dashing_dfs)} files")
    for file, df in dashing_dfs.items():
        print(f"{file}: {df.shape}")
    return dashing_dfs


def build_merged_df(tree_dict, dashing_dfs):
    """
    Build merged dataframe for all genome pairs among `genomes`,
    including tree distances and distances for each k from `dashing_dfs`.

    Parameters:
    -----------
    tree_dict : dict
        Nested dictionary of tree-based distances; tree_dict[g1][g2] gives distance.
    dashing_dfs : dict
        Mapping from dashing file name to pairwise distance DataFrame.

    Returns:
    --------
    merged_df : pd.DataFrame
        DataFrame indexed by sorted genome pairs, with metadata and k-mer distances.
    """

    genomes = sorted(tree_dict.keys())

    # Create all pairs of genomes (upper triangle only, sorted lexicographically)
    pairs = []
    for i in range(len(genomes)):
        for j in range(i+1, len(genomes)):
            pair = '_'.join(sorted([genomes[i], genomes[j]]))
            pairs.append(pair)

    # Initialize dataframe with pairs as index
    merged_df = pd.DataFrame(index=pairs)

    # Add metadata columns to merged_df
    genome_pairs = []
    genomes_1s = []
    genomes_2s = []
    tree_distances = []
    for index in merged_df.index:
        # Split the genome pair
        genome1 = "_".join(index.split("_", 2)[:2])
        genome2 = "_".join(index.split("_", 2)[2:])

        # Get distance from tree_dict
        genome_pairs.append(index)
        genomes_1s.append(genome1)
        genomes_2s.append(genome2)
        tree_distances.append(tree_dict[genome1][genome2])

    merged_df['genome_pair'] = genome_pairs
    merged_df['genome_1'] = genomes_1s
    merged_df['genome_2'] = genomes_2s
    merged_df['tree_distance'] = tree_distances

    print(f"Final dataframe shape: {merged_df.shape}")

    # For each k, add a column to merged_df with the pairwise distances
    for fname, df in dashing_dfs.items():
        k_val = int(str(fname).replace('distance_k', '').replace('.txt', ''))
        # Build a dictionary mapping sorted genome pair -> distance
        pair_to_dist = {}
        for i in range(len(df)):
            g1 = df.iloc[i].name
            for g2, d in dict(df.iloc[i]).items():
                if g1 == g2: continue
                pair = '_'.join(sorted([g1, g2]))
                pair_to_dist[pair] = d

        # Map the distances to the merged_df for this k
        colname = f'k{k_val}_distance'
        merged_df[colname] = merged_df.index.map(pair_to_dist.get)
        print(f"Added column '{colname}' to merged_df.")

    return merged_df

def process_data(tree_path, dashing_dir_path):
    tree_dict = make_tree_dict(tree_path)
    dashing_dfs = parse_dashing_dfs(dashing_dir_path)
    merged_df = build_merged_df(tree_dict, dashing_dfs)
    return tree_dict, dashing_dfs, merged_df


################################################################################
# Grid sreach

## one k-mer
# Find single best k-mer length

def initial_corr(merged_df, iteration=0, k_low=5, k_high=32):
    ''' Find best single value k. Output correlation in clipped direction '''
    results_matrix = np.zeros((k_high - k_low + 1, 3 + iteration))
    for i, k in enumerate(range(k_low, k_high + 1)):
        k_rank = merged_df[f'k{k}_distance'].rank()
        corr, pval = spearmanr(k_rank, merged_df['tree_distance'])
        results_matrix[i, :] = [k, -1*corr, pval]
    min_idx = np.nanargmax(results_matrix[:,-2])
    return results_matrix, (int(results_matrix[min_idx][0]), *results_matrix[min_idx][1:3])

# Plot single best k-mer length
def plot_1k_colorbar_corr(name, df, k_low=6, k_high=40, cmap_vmax=None):
    """Plot a single heatmap row with one colorbar for k-mer correlations."""
    arr = np.asarray(df)
    k_vals = arr[:, 0].astype(int)
    corr_vals = arr[:, 1].astype(float)
    keep = (k_vals >= k_low) & (k_vals <= k_high) # filter to k-mers between k_low and k_high
    k_vals = k_vals[keep]
    corr_vals = corr_vals[keep]
    heatmap = corr_vals[np.newaxis, :]  # shape (1, n_k)
    fig, ax = plt.subplots(1, 1, figsize=(5,5), constrained_layout=True) # square heatmap plot

    if cmap_vmax:
        cmap_shared, vmax = cmap_vmax
        im = ax.imshow(
            heatmap,
            aspect="auto",
            origin="lower",
            cmap=cmap_shared,
            vmin=0,
            vmax=vmax
        )
    else:
        im = ax.imshow(
            heatmap,
            aspect="auto",
            origin="lower",
            cmap="YlOrRd",
            vmin=0
        )


    xtick_pos = np.arange(0, len(k_vals), 2)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(k_vals[xtick_pos], fontsize=8)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("k-mer length")
    ax.set_title(f"{name}: Spearman correlation for one k-mer, k={k_low}..{k_high}")

    # Draw a black edge around every location where the correlation is maximal (including ties)
    max_corr = np.nanmax(corr_vals)
    best_xs = np.where(np.isclose(corr_vals, max_corr))[0]
    for best_x in best_xs:
        rect = plt.Rectangle(
            (best_x - 0.5, -0.5),
            1,
            1,
            linewidth=2.5,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)

    fig.colorbar(im, ax=ax, label="Spearman corr")
    plt.show()

################################################################################
# Full Grid search: unfixed k-mers; with and without weights

def run_full_grid_corr(merged_df, k_low=5, k_high=32, w_low=0, w_high=1, w_step=0.05):
    ''' Full grid search with all degrees of freedom: k1 low-> high, k2 low -> high, with all weights. '''
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
                corr, pval = spearmanr(avg_k1_k2_w_rank, merged_df['tree_distance'])
                results_matrix[k1_idx, k2_idx, w_idx] = corr

    # find global min
    results_matrix = np.nan_to_num(results_matrix, nan=0.0)
    min_val = np.min(results_matrix)
    min_idx = np.unravel_index(np.argmin(results_matrix), results_matrix.shape)
    k1_min, k2_min, w_min = ks[min_idx[0]], ks[min_idx[1]], ws[min_idx[2]]
    print(f"Grid Ks: {k1_min, k2_min} with weight {w_min}; corr: {min_val}")
    print(min_idx)
    return results_matrix, min_val, k1_min, k2_min, w_min, min_val

def plot_2k_grid_corr(name, results_matrix, k_low, k_high, cmap_vmax=None):
    """
    Plot a 2D heatmap using only the first two dimensions (k1, k2) of results_matrix.
    Averages (or finds the min/max) over the remaining dimension (e.g., ws).
    Spearman correlation strength is shown; the axes cover all ks in the range.
    """
    aggregate_matrix = np.max(results_matrix, axis=2)  # shape [k1, k2]
    ks = list(range(k_low, k_high + 1))
    n_ks = len(ks)
    fig, ax = plt.subplots(figsize=(8, 6))

    if cmap_vmax:
        cmap_shared, vmax = cmap_vmax
        im = ax.imshow(
            aggregate_matrix,
            origin="lower",
            aspect="equal",
            vmin=0,
            cmap=cmap_shared,
            vmax=vmax
        )
    else:
        im = ax.imshow(
            aggregate_matrix,
            origin="lower",
            aspect="equal",
            vmin=0,
            cmap="YlOrRd")

    # Label axes as k1 (y), k2 (x):
    ax.set_xlabel("k2")
    ax.set_ylabel("k1")
    ax.set_title(f"{name}: Spearman correlation for two k-mers, k={k_low}..{k_high}")

    xticks = yticks = list(range(n_ks))
    ax.set_xticks(xticks)
    ax.set_xticklabels(ks, fontsize=8, rotation=90)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ks, fontsize=8)

    # Optional: mark the globally min (best) cell
    # max_val = np.nanmax(aggregate_matrix)
    max_val = np.nanmax(aggregate_matrix[np.isfinite(aggregate_matrix)])
    min_indices = np.where(aggregate_matrix == max_val)
    print(min_indices, max_val)
    if len(min_indices[0]) > 0:
        y, x = min_indices[0][0], min_indices[1][0]
        ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, fill=False, edgecolor='black', lw=2.5))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman correlation")
    plt.tight_layout()
    plt.show()

def plot_2k_grid_weights(name, results_matrix, k_low, k_high, w_low=0, w_high=1, w_step=0.05):
    """
    Plot a 2D heatmap using only the first two dimensions (k1, k2) of results_matrix.
    Averages (or finds the min/max) over the remaining dimension (e.g., ws).
    Spearman correlation strength is shown; the axes cover all ks in the range.
    """
    indexes = np.nanargmax(results_matrix, axis=2)  # shape [k1, k2]
    values = np.round(np.arange(w_low, w_high + w_step, w_step), 3)
    aggregate_matrix = values[indexes]

    ks = list(range(k_low, k_high + 1))
    n_ks = len(ks)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Discrete colormap + boundaries so the legend uses stepped weight bins.
    cmap = plt.get_cmap("Greys", len(values))
    boundaries = np.arange(0, 1 + 0.05, 0.05)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
    im = ax.imshow(aggregate_matrix, origin="lower", aspect="equal", cmap=cmap, norm=norm)

    # Label axes as k1 (y), k2 (x):
    ax.set_xlabel("k2")
    ax.set_ylabel("k1")
    ax.set_title(f"{name}: Two k-mers, k1 weight, k={k_low}..{k_high}")

    xticks = yticks = list(range(n_ks))
    ax.set_xticks(xticks)
    ax.set_xticklabels(ks, fontsize=8, rotation=90)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ks, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="k1 weight", boundaries=boundaries, ticks=values)
    cbar.ax.set_yticklabels([f"{v:.2f}" for v in values])
    plt.tight_layout()
    plt.show()


################################################################################

def plot_3k_grid_corr(name, grid_df, k, s, k_low, k_high, cmap_vmax=None):
    """
    Plot one heatmap per k1 in [k−s, …, k+s] (each 35×35, k2 × k3);
    collapse weight dims by min Spearman (matches grid objective), then negate
    so stronger associations plot as larger positive values (for YlOrRd).
    Only show every other tick for the x and y axes.
    Tick labels use a smaller font.
    In each subplot, color the best (largest) cell red.
    Across all subplots, color the overall best cell black.
    """
    planes = grid_df.max(axis=(3, 4))  # (n_k1, 35, 35); flip sign so min ρ → positive
    n_k1_full = planes.shape[0]
    half = (n_k1_full - 1) // 2
    if s > half:
        raise ValueError(f"s={s} exceeds k1 half-width {half} in grid")
    planes = planes[half - s : half + s + 1]
    n_panels = planes.shape[0]
    fig_w = 24 * n_panels / 7
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 3.5), constrained_layout=True, squeeze=False)
    axes = np.ravel(axes)
    pfin = planes[np.isfinite(planes)]
    vmin, vmax = float(np.min(pfin)), float(np.max(pfin))

    # k1 titles aligned with sliced panels (length must equal n_panels)
    k_subplot_labels = [k - s + i for i in range(n_panels)]
    k_axis_labels = list(range(k_low, k_high + 1))

    # Only select every other tick for the axes
    n_labels = len(k_axis_labels)
    xtick_positions = [i for i in range(n_labels) if i % 2 == 0]  # every other tick
    ytick_positions = [i for i in range(n_labels) if i % 2 == 0]
    xtick_labels = [k_axis_labels[i] for i in xtick_positions]
    ytick_labels = [k_axis_labels[i] for i in ytick_positions]

    # Overall best cell across all planes (largest after negation)
    global_max = np.nanmax(planes)
    g_indices = np.where(planes == global_max)
    if len(g_indices[0]) > 0:
        global_best_subplot = int(g_indices[0][0])
        global_best_y = int(g_indices[1][0])
        global_best_x = int(g_indices[2][0])
    else:
        global_best_subplot = global_best_y = global_best_x = None

    im = None
    tick_label_fontsize = 8  # Make tick labels smaller
    for i, ax in enumerate(axes):
        data = planes[i]

        if cmap_vmax:
            cmap_shared, vmax = cmap_vmax
            im = ax.imshow(
                data,
                aspect="equal",
                origin="lower",
                vmin=0,
                cmap=cmap_shared,
                vmax=vmax
            )
        else:
            im = ax.imshow(data,
                           aspect="equal",
                           origin="lower",
                           vmin=0,
                           cmap="YlOrRd")

        ax.set_xlabel("k3")
        ax.set_ylabel("k2")
        ax.set_title(f"k1={k_subplot_labels[i]}")

        # Set x/y ticks and labels at every other tick, use smaller font
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, fontsize=tick_label_fontsize)
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, fontsize=tick_label_fontsize)

        # Highlight the best (largest) cell in the plane with a red rectangle or marker,
        # but only if there is at least one finite value.
        if np.any(np.isfinite(data)):
            max_val = np.nanmax(data)
            ys, xs = np.where(data == max_val)
            if len(ys) > 0:
                y, x = ys[0], xs[0]
                # If this cell is ALSO the global best, color it black. Otherwise, red.
                if i == global_best_subplot and y == global_best_y and x == global_best_x:
                    rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                        linewidth=2.5, edgecolor='#000000', facecolor='none')       # #00BFFF for blue
                    ax.add_patch(rect)
                # else:
                    # rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                    #                     linewidth=2.5, edgecolor='red', facecolor='none')
                    # ax.add_patch(rect)

    # fig.suptitle(f"{name}: Spearman correlation for three k-mers, k={k_low}..{k_high}", fontsize=18)
    fig.suptitle(f"{name}: Spearman correlation for three k-mers, k={k_low}..{k_high}")

    # Move the colorbar (legend) more to the left by reducing the pad
    # You can use the ax argument as a list, but use the 'fraction' and 'pad' kwargs to move it left
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, location="right", label="Spearman corr", pad=0.02)
    plt.show()


################################################################################


def parse_union_cardinality_dashing_dfs(dir_path):
    """
    Parses all distance_* files in the given directory into a dictionary of pandas DataFrames.
    Each DataFrame skips the first line, assumes tab-separation and adjusts indices and columns as expected.

    Args:
        dir_path (str): Path to the directory containing distance_* files.

    Returns:
        dict: Dictionary mapping filename to corresponding DataFrame.
    """

    # Get list of files in directory that start with 'distance_'
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.startswith('union_')]

    # Create a dictionary to store dataframes
    dashing_dfs = {}

    # Read each file into a dataframe, skipping first line
    for file in files:
        file_path = os.path.join(dir_path, file)
        try:
            # Skip first line and read tab-separated data
            df = pd.read_csv(file_path, sep='\t', header=None, skiprows=3)
            df.index = df.iloc[:, 0].str.replace('genomes_ncbi/', '')
            df = df.iloc[:, 1:]
            df.columns = df.index
            dashing_dfs[file] = df
            # print(f"Successfully read {file}")
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")

    # Print summary of loaded dataframes
    print(f"\nLoaded {len(dashing_dfs)} files")
    for file, df in dashing_dfs.items():
        print(f"{file}: {df.shape}")
    return dashing_dfs


def build_merged_card_df(tree_dict, dashing_card_dfs):
    """
    Build merged dataframe for all genome pairs among `genomes`,
    including tree distances and distances for each k from `dashing_card_dfs`.

    Parameters:
    -----------
    tree_dict : dict
        Nested dictionary of tree-based distances; tree_dict[g1][g2] gives distance.
    dashing_card_dfs : dict
        Mapping from dashing file name to pairwise distance DataFrame.

    Returns:
    --------
    merged_card_df : pd.DataFrame
        DataFrame indexed by sorted genome pairs, with metadata and k-mer distances.
    """

    genomes = sorted(tree_dict.keys())

    # Create all pairs of genomes (upper triangle only, sorted lexicographically)
    pairs = []
    for i in range(len(genomes)):
        for j in range(i+1, len(genomes)):
            pair = '_'.join(sorted([genomes[i], genomes[j]]))
            pairs.append(pair)

    # Initialize dataframe with pairs as index
    merged_card_df = pd.DataFrame(index=pairs)

    # Add metadata columns to merged_card_df
    genome_pairs = []
    genomes_1s = []
    genomes_2s = []
    tree_distances = []
    for index in merged_card_df.index:
        # Split the genome pair
        genome1 = "_".join(index.split("_", 2)[:2])
        genome2 = "_".join(index.split("_", 2)[2:])

        # Get distance from tree_dict
        genome_pairs.append(index)
        genomes_1s.append(genome1)
        genomes_2s.append(genome2)
        tree_distances.append(tree_dict[genome1][genome2])

    merged_card_df['genome_pair'] = genome_pairs
    merged_card_df['genome_1'] = genomes_1s
    merged_card_df['genome_2'] = genomes_2s
    merged_card_df['tree_distance'] = tree_distances

    print(f"Final dataframe shape: {merged_card_df.shape}")

    # For each k, add a column to merged_card_df with the pairwise distances
    for fname, df in dashing_card_dfs.items():
        k_val = int(str(fname).replace('union_k', '').replace('.txt', ''))
        # Build a dictionary mapping sorted genome pair -> distance
        pair_to_dist = {}
        for i in range(len(df)):
            g1 = df.iloc[i].name
            for g2, d in dict(df.iloc[i]).items():
                if g1 == g2: continue
                pair = '_'.join(sorted([g1, g2]))
                pair_to_dist[pair] = d

        # Map the distances to the merged_card_df for this k
        colname = f'k{k_val}_cardinality'
        merged_card_df[colname] = merged_card_df.index.map(pair_to_dist.get)

        print(f"Added column '{colname}' to merged_card_df.")
        if k_val % 2 == 0: # k-mer is even
            num_pal = 4**(k_val/2)
            num_distinct_kmers = (4**k_val - num_pal) / 2 + num_pal
        else:
            num_distinct_kmers = 4**k_val / 2

        merged_card_df[colname] = merged_card_df[colname] / num_distinct_kmers

    return merged_card_df

def process_card_data(tree_path, dashing_dir_path):
    tree_dict = make_tree_dict(tree_path)
    dashing_card_dfs = parse_union_cardinality_dashing_dfs(dashing_dir_path)
    merged_card_df = build_merged_card_df(tree_dict, dashing_card_dfs)
    return tree_dict, dashing_card_dfs, merged_card_df



def plot_kmer_dashing_card(merged_card_df, name, k_min=5, k_max=64, log=False):
    """
    Plot k-mer cardinality boxplots or their -log10 transforms for columns k{k}_cardinality in merged_card_df.

    Parameters
    ----------
    merged_card_df : pandas.DataFrame
        DataFrame containing columns like 'k5_cardinality', ..., 'k64_cardinality'.
    name : str
        Title prefix for plot.
    k_min : int, optional
        Minimum k to start from (default is 5).
    k_max : int, optional
        Maximum k to end at (default is 64).
    log : bool, optional
        If True, plot -1 * log10(cardinality) instead of cardinality. Default is False.
    """

    data = []
    labels = []
    medians = []
    for k in range(k_min, k_max + 1):
        col = f"k{k}_cardinality"
        if col in merged_card_df.columns:
            col_values = merged_card_df[col].dropna()
            if log:
                values = -1 * np.log10(col_values)
            else:
                values = col_values
            data.append(values)
            labels.append(f"k{k}")
            if len(values) > 0:
                medians.append(np.median(values))
            else:
                medians.append(np.nan)

    plt.figure(figsize=(max(12, len(labels) * 0.4), 6))
    box = plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=90)
    if log:
        plt.ylabel(r"-log$_{10}$(k-mer cardinality / number of k-mers)")
        plt.title(f"{name}: -log10 k-mer cardinality (k = {k_min}..{k_max})")
    else:
        plt.ylabel("k-mer cardinality / number of k-mers")
        plt.title(f"{name}: k-mer cardinality (k = {k_min}..{k_max})")
    plt.tight_layout()

    # Label each boxplot with the median
    for i, median in enumerate(medians, start=1):
        if not np.isnan(median):
            if log:
                plt.text(i, median, f"{median:.3f}", ha='center', va='bottom', color='midnightblue',
                        fontsize=8, fontweight='bold', rotation=90)
            else:
                plt.text(i, median, f"{median:.2e}", ha='center', va='bottom', color='midnightblue',
                        fontsize=8, fontweight='bold', rotation=90)

    plt.show()



################################################################################
# colorscheme stuff

def min_max_3k(grid_df, s):
    """
    Given a grid_df, output the min and max value of each subplot (panel)
    in the planes sliced as in the original 3k grid plotting function.
    Also prints and returns the global min and max across all panels.
    """
    planes = grid_df.max(axis=(3, 4))  # (n_k1, 35, 35)
    n_k1_full = planes.shape[0]
    half = (n_k1_full - 1) // 2
    if s > half:
        raise ValueError(f"s={s} exceeds k1 half-width {half} in grid")
    planes = planes[half - s : half + s + 1]
    n_panels = planes.shape[0]
    min_max_list = []
    all_finite_vals = []
    for i in range(n_panels):
        data = planes[i]
        finite_vals = data[np.isfinite(data)]
        if finite_vals.size > 0:
            min_val = float(np.min(finite_vals))
            max_val = float(np.max(finite_vals))
            all_finite_vals.append(finite_vals)
        else:
            min_val = float('nan')
            max_val = float('nan')
        min_max_list.append((min_val, max_val))
        # print(f"Panel {i}: min={min_val}, max={max_val}")

    # Calculate global min/max across all panels
    if all_finite_vals:
        all_data = np.concatenate(all_finite_vals)
        global_min = float(np.min(all_data))
        global_max = float(np.max(all_data))
        # print(f"Global across panels: min={global_min}, max={global_max}")
    else:
        global_min = float('nan')
        global_max = float('nan')
        # print("Global across panels: min=nan, max=nan")
    return global_min, global_max


def make_custom_k_cmap_norm(k_one, k_two, k_three):
    """
    Construct a piecewise colormap and normalization for k-mer heatmaps.
    Returns: (colormap, norm)
    """
    k_one_l, k_one_h = k_one
    k_two_l, k_two_h = k_two
    k_three_l, k_three_h = k_three

    # Segment boundaries
    vmin = 0 #min(k_one_l, k_two_l, k_three_l)
    vmax = max(k_one_h, k_two_h, k_three_h)
    scale = vmax - vmin

    color_1_start = vmin
    color_1_end_color_2_start =  k_one_h
    # color_1_end_color_2_start =  min(k_one_h, k_two_h)
    color_2_end_color_3_start = max(k_one_h, k_two_h)
    # color_2_end_color_3_start = k_two_h        # if error, max(k_one_h, k_two_h)
    color_3_end = vmax

    # Build ordered list of (x, color) tuples
    color_points = [
        # color 1
        ((color_1_start - vmin) / scale, plt.cm.viridis(0.00)),     # normalized start of k-one, dark viridis
        ((color_1_end_color_2_start - vmin) / scale, plt.cm.viridis(0.50)),     # normalized end of k-one, yellow
        # color 2
        ((color_1_end_color_2_start - vmin) / scale, plt.cm.YlOrRd(0.25)),    # start (normalized) of k-two, lighter blue
        ((color_2_end_color_3_start - vmin) / scale, plt.cm.YlOrRd(1.00)),      # end (normalized) of k-two, deep blue
        # color 3
        ((color_2_end_color_3_start - vmin) / scale, plt.cm.RdBu(0.35)),    # start k-three, lighter green
        ((color_3_end - vmin) / scale, plt.cm.RdBu(0.95)),      # end k-three, deep green
    ]

    custom_k_cmap = mcolors.LinearSegmentedColormap.from_list("custom_k_cmap", color_points)
    custom_k_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    print(color_points)
    return custom_k_cmap, custom_k_norm, vmax



def build_cmap_dict(datasets_1k, datasets_2k, datasets_3k):
    cmap_dict = dict()
    for (name, k1_df), (_, k2_df), (_, k3_df, *_) in zip(datasets_1k, datasets_2k, datasets_3k):
        k1_corr =  k1_df[:, 1]
        k1_min, k1_max = np.nanmin(k1_corr[np.isfinite(k1_corr)]), np.nanmax(k1_corr[np.isfinite(k1_corr)])    # good
        k2_agg = np.max(k2_df, axis=2)  # shape [k1, k2]
        k2_min, k2_max =  np.nanmin(k2_agg[np.isfinite(k2_agg)]), np.nanmax(k2_agg[np.isfinite(k2_agg)])
        k3_min, k3_max = min_max_3k(k3_df, s=2)
        print(f"{name}:")
        print(f"  k1_df min: {k1_min}, max: {k1_max}")
        print(f"  k2_df min: {k2_min}, max: {k2_max}")
        print(f"  k3_df min: {k3_min}, max: {k3_max}")
        custom_k_cmap, custom_k_norm, vmax = make_custom_k_cmap_norm((k1_min, k1_max), (k2_min, k2_max), (k3_min, k3_max))
        cmap_dict[name] = custom_k_cmap, vmax
    return cmap_dict