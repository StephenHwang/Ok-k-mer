#!/usr/bin/env python3
"""Prepare validated new-dataset inputs and manifests; export manuscript artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
from ete3 import Tree
from scipy.stats import spearmanr
import stability

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
SOURCES = [('o_Anaerolineales', 'anaerolineales', 'o__Anaerolineales.nwk', 'run_20261027T232554Z_h128Lf'), ('c_Dethiobacteria', 'dethiobacteria', 'c__Dethiobacteria.nwk', 'run_20261027T232607Z_2oE3Jg')]

def accession(value):
    match = re.match(r'(GC[AF]_\d+\.\d+)', Path(value).name)
    if not match:
        raise ValueError(f'Unrecognized genome: {value}')
    return match[1]

def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Output directory must be empty')
    template = json.loads((DATA / 'stability_3k/manifest.json').read_text())
    templates = [t for t in template['tasks'] if t['dataset'] == template['tasks'][0]['dataset']]
    tasks = []
    provenance = []
    for name, folder, treefile, run in SOURCES:
        treepath = (DATA / folder / treefile).resolve()
        tree = Tree(str(treepath), format=1)
        excluded = ['GCF_000174415.1', 'GCF_022558445.1'] if name == 'c_Dethiobacteria' else []
        leaves = sorted(set(tree.get_leaf_names()) - set(excluded))
        # User-confirmed RefSeq/GenBank mappings, including the version change.
        aliases = {}
        if name == 'o_Anaerolineales':
            for leaf in leaves:
                if leaf.startswith('GCF_'):
                    matrix_id = leaf.replace('GCF_', 'GCA_', 1)
                    if leaf == 'GCF_001050235.1':
                        matrix_id = 'GCA_001050235.2'
                    aliases[matrix_id] = leaf
        (out / f'{name}_accession_mapping.json').write_text(json.dumps(aliases, indent=2))
        assert len(leaves) == len(set(leaves))
        i, j = np.triu_indices(len(leaves), 1)
        df = pd.DataFrame({'genome_1': np.array(leaves)[i], 'genome_2': np.array(leaves)[j]})
        nodes = {leaf.name: leaf for leaf in tree}
        df['tree_distance'] = [nodes[a].get_distance(nodes[b]) for a, b in zip(df.genome_1, df.genome_2)]
        for k in range(6, 41):
            path = DATA / folder / 'genomes_timing_test/distances' / run / f'jaccard_k{k}.tsv'
            with path.open() as f:
                f.readline()
                columns = [aliases.get(accession(x), accession(x)) for x in f.readline().strip().split('\t')[1:]]
            matrix = pd.read_csv(path, sep='\t', comment='#', header=None, index_col=0)
            matrix.index = [aliases.get(accession(x), accession(x)) for x in matrix.index]
            assert len(columns) == len(set(columns)) and matrix.index.is_unique
            assert set(columns) == set(leaves) == set(matrix.index)
            matrix.columns = columns
            values = matrix.loc[leaves, leaves].to_numpy(float)
            assert np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))
            assert np.allclose(values, values.T) and np.allclose(values.diagonal(), 0)
            # Original optimizers minimize rho on similarities; reflect distance ranks.
            df[f'k{k}_distance'] = -values[i, j]
            provenance.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        csv = out / f'{name}_merged_df.csv'
        df.to_csv(csv, index=False)
        for t in templates:
            t = dict(t)
            pct, sim = t['masking_percentage'], t['simulation']
            t.update(task_id=len(tasks), excluded_genomes=excluded, dataset=name, tree_path=str(treepath), merged_csv=str(csv.resolve()), seed=int.from_bytes(hashlib.sha256(f"{t['master_seed']}:{name}:{pct}:{sim}".encode()).digest()[:8], 'big'))
            tasks.append(t)
        print(name, len(leaves), 'genomes;', len(df), 'pairs', flush=True)
    for maximum in (2, 3):
        directory = out / f'{maximum}k'
        for sub in ('results', 'logs', 'submissions'):
            (directory / sub).mkdir(parents=True)
        selected = [dict(t) for t in tasks if maximum == 3 or t['masking_percentage'] == 0]
        for idx, t in enumerate(selected):
            t.update(task_id=idx, max_k=maximum, save_tensor=False, json_out=str((directory / 'results' / f'{idx:04d}.json').resolve()))
        (directory / 'manifest.json').write_text(json.dumps({'schema_version': 1, 'max_k': maximum, 'tasks': selected}, indent=2))
        stability.write_csv(directory / 'tasks.csv', selected)
    (out / 'input_provenance.json').write_text(json.dumps(provenance, indent=2))

def plot_figures(out, combined=None, only=None):
    if combined is None:
        combined = pd.read_csv(out / "all_eight_simulations.csv")
    labels = {
        'o_Anaerolineales': 'o__Anaerolineales (525)',
        'Aquificota': 'p__Aquificota (132)',
        'Bacillales': 'o__Bacillales_A (249)',
        'Caldisericia': 'c__Caldisericia (49)',
        'Chlamydia': 'o__Chlamydiales (288)',
        'Cyano': 'o__Cyanobacteriales (585)',
        'c_Dethiobacteria': 'c__Dethiobacteria (102)',
        'Neisseriaceae': 'f__Neisseriaceae (145)',
    }
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for column, label, stem in [('rho', r'Spearman correlation, $\rho$', 'fig4_spearman'), ('k_eff', 'Weighted composite k-mer length', 'figA1_weighted_k')]:
        if only is not None and column != only:
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        for name, group in combined[combined.masking_percentage <= 60].groupby('dataset', sort=False):
            stats = group.groupby('masking_percentage')[column].agg(['mean', 'std'])
            x = stats.index.to_numpy()
            y = stats['mean'].to_numpy()
            sd = stats['std'].fillna(0).to_numpy()
            line, = ax.plot(x, y, marker='o', label=labels[name])
            ax.fill_between(x, y-sd, y+sd, color=line.get_color(), alpha=.12)
        ax.set(xlabel='Genomes masked (%)', ylabel=label)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.tick_params(axis='both', direction='in')
        ax.set_xticks(range(0, 61, 10), [f'{x}%' for x in range(0, 61, 10)])
        if column == 'rho':
            ax.set_ylim(0.45, 1.0)
            ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        handles, texts = ax.get_legend_handles_labels()
        by_label = dict(zip(texts, handles))
        ordered = [label for label in labels.values() if label in by_label]
        if column == 'rho':
            # Recover the panel width from the original 9-inch, right-legend layout.
            old_legend = ax.legend([by_label[label] for label in ordered], ordered,
                                   loc='center left', bbox_to_anchor=(1.02, 0.5),
                                   frameon=False, fontsize=8)
            fig.tight_layout()
            panel_width_inches = ax.get_position().width * fig.get_figwidth()
            old_legend.remove()
            fig.set_size_inches(12, 6)
            # Matplotlib fills columns first; preserve the requested row order.
            legend_order = [ordered[i] for i in (0, 4, 1, 5, 2, 6, 3, 7)]
            ax.legend([by_label[label] for label in legend_order], legend_order,
                      loc='lower center', bbox_to_anchor=(0.5, 1.01),
                      ncol=4, frameon=False, fontsize=11,
                      columnspacing=1.2, handlelength=1.8, borderaxespad=0)
        else:
            ax.legend([by_label[label] for label in ordered], ordered,
                      loc='center left', bbox_to_anchor=(1.02, 0.5),
                      frameon=False, fontsize=11)
        fig.tight_layout()
        # Apply one uniform scale to labels, tick labels, and legend text.
        from matplotlib.text import Text
        for text in fig.findobj(match=Text):
            text.set_fontsize(text.get_fontsize() * 2.25)
        if column == 'rho':
            position = ax.get_position()
            width = panel_width_inches / fig.get_figwidth()
            ax.set_position([(1 - width) / 2, position.y0, width, position.height])
            # Match the legend width and reduce the reference panel height by 25%.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            panel = ax.get_window_extent(renderer)
            legend = ax.get_legend().get_window_extent(renderer)
            aspect = panel.width / panel.height
            panel_width = legend.width / fig.dpi
            panel_height = 0.75 * panel_width / aspect
            left, right, bottom = 0.8, 0.2, 0.6
            top = legend.height / fig.dpi + 0.01 * panel_height + 0.2
            figure_width = left + panel_width + right
            figure_height = bottom + panel_height + top
            fig.set_size_inches(figure_width, figure_height)
            ax.set_position([left / figure_width, bottom / figure_height,
                             panel_width / figure_width, panel_height / figure_height])
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            panel = ax.get_window_extent(renderer)
            legend = ax.get_legend().get_window_extent(renderer)
            assert abs(panel.width - legend.width) < 1
            assert abs(panel.width / panel.height - aspect / 0.75) < 1e-6
            print(f'Fig 4 panel: {panel_width:.3f} x {panel_height:.3f} inches; '
                  f'legend width: {legend.width / fig.dpi:.3f} inches')
        for extension in ('pdf', 'png'):
            fig.savefig(out / f'{stem}.{extension}', dpi=300, bbox_inches='tight')
        plt.close(fig)

def collect(out):
    for maximum in (2, 3):
        stability.collect(argparse.Namespace(manifest=out / f'{maximum}k/manifest.json', tol=1., plots=True))
    three = pd.read_csv(out / '3k/stability_simulations.csv')
    two = pd.read_csv(out / '2k/stability_simulations.csv')
    original = pd.read_csv(DATA / 'stability_3k/stability_simulations.csv')
    combined = pd.concat([original, three], ignore_index=True)
    combined['rho'] = -combined.correlation
    combined.to_csv(out / 'all_eight_simulations.csv', index=False)
    trials = combined[combined.simulation < 3]
    trials.to_csv(out / 'table3_all_fractions.csv', index=False)
    trials[trials.masking_percentage == 30].to_csv(out / 'table3_30percent.csv', index=False)
    plot_figures(out, combined)
    rows = []
    for name, group in three.groupby('dataset'):
        base = group[group.masking_fraction == 0].iloc[0]
        df = pd.read_csv(out / f'{name}_merged_df.csv')
        correlations = {k: -spearmanr(df[f'k{k}_distance'], df.tree_distance).statistic for k in range(6, 41)}
        best = max(correlations, key=correlations.get)
        row = {'dataset': name, 'single_k': best, 'single_rho': correlations[best], 'rho21': correlations[21], 'rho31': correlations[31]}
        for n, result in [(2, two[two.dataset == name].iloc[0]), (3, base)]:
            row[f'rho_{n}k'] = -result.correlation
            for col in ('k1','k2','k3','w1','w2','w3'):
                row[f'{n}k_{col}'] = result[col]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / 'table2_and_A1.csv', index=False)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'collect', 'plot_figures'])
    p.add_argument('--out', type=Path, default=ROOT / 'results/new_datasets_20261028')
    args = p.parse_args()
    globals()[args.command](args.out.resolve())
